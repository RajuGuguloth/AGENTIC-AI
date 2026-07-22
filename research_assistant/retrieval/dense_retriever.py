"""
Retrieval — Dense Retriever
FAISS-based vector search using SentenceTransformer for text and CLIP for images.
Text and image embeddings share a unified 512-dim FAISS index.
"""

from typing import List, Optional, Tuple

import faiss
import numpy as np
import os
import pickle
import torch
from langchain_core.documents import Document
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from config import Config
from retrieval.score_filter import filter_by_score

UNIFIED_DIM = 512
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

_clip_model: Optional[CLIPModel] = None
_clip_processor: Optional[CLIPProcessor] = None


def _get_clip() -> Tuple[CLIPModel, CLIPProcessor]:
    """Lazy-load CLIP model and processor."""
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        print(f"[dense_retriever] Loading CLIP model: {CLIP_MODEL_NAME}")
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _project_to_512(embedding: np.ndarray) -> np.ndarray:
    """Project an embedding vector to 512 dimensions and L2-normalize."""
    vec = np.asarray(embedding, dtype=np.float32).flatten()
    if vec.shape[0] == UNIFIED_DIM:
        out = vec.copy()
    elif vec.shape[0] < UNIFIED_DIM:
        out = np.pad(vec, (0, UNIFIED_DIM - vec.shape[0]))
    else:
        out = vec[:UNIFIED_DIM]

    norm = np.linalg.norm(out)
    if norm > 0:
        out = out / norm
    return out


def _l2_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """L2-normalize a feature tensor."""
    return torch.nn.functional.normalize(tensor, p=2, dim=-1)


def _clip_features_to_tensor(features) -> torch.Tensor:
    """Convert CLIP model output to a 2D float tensor."""
    if isinstance(features, torch.Tensor):
        return features

    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state"):
        return features.last_hidden_state[:, 0, :]

    raise TypeError(f"Unexpected CLIP feature type: {type(features)}")


def embed_query_text_for_clip(query: str) -> np.ndarray:
    """
    Embed a text query with CLIP's text encoder for image retrieval.

    Args:
        query: Natural-language search query.

    Returns:
        L2-normalised 512-dim embedding as float32 numpy array.
    """
    model, processor = _get_clip()
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = _l2_normalize_tensor(_clip_features_to_tensor(features))
    return _project_to_512(features.squeeze(0).cpu().numpy())


def _embed_clip_image(image_path: str) -> np.ndarray:
    """Embed an image file with CLIP's image encoder."""
    model, processor = _get_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = _l2_normalize_tensor(_clip_features_to_tensor(features))
    return _project_to_512(features.squeeze(0).cpu().numpy())


class DenseRetriever:
    """
    FAISS IndexFlatIP with L2-normalised embeddings = cosine similarity search.
    Candidate pool size: TOP_K_DENSE (default 20) for RRF input.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        print(f"[dense_retriever] Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.index: faiss.Index = None
        self.documents: List[Document] = []
        self.dimension: int = None

    def _embed_text_document(self, doc: Document) -> np.ndarray:
        """Embed a text/table document with SentenceTransformer, projected to 512 dims."""
        embedding = self.model.encode(
            [doc.page_content],
            normalize_embeddings=True,
        )[0]
        return _project_to_512(embedding)

    def _embed_image_document(self, doc: Document) -> np.ndarray:
        """Embed an image document with CLIP."""
        image_path = doc.metadata.get("image_path") or doc.metadata.get("file_path")
        if not image_path:
            raise ValueError(
                f"Image document '{doc.metadata.get('source', 'unknown')}' "
                "is missing image_path/file_path metadata."
            )
        return _embed_clip_image(image_path)

    def build_index(self, documents: List[Document]) -> None:
        """
        Embed all document chunks and build FAISS index.

        Args:
            documents: List of LangChain Documents (chunks).
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list.")

        self.documents = documents
        image_count = sum(
            1 for doc in documents if doc.metadata.get("content_type") == "image"
        )
        text_count = len(documents) - image_count

        print(
            f"[dense_retriever] Embedding {len(documents)} chunks "
            f"({text_count} text/table, {image_count} image)..."
        )

        embeddings_list: List[np.ndarray] = []
        for doc in documents:
            if doc.metadata.get("content_type") == "image":
                embeddings_list.append(self._embed_image_document(doc))
            else:
                embeddings_list.append(self._embed_text_document(doc))

        embeddings_np = np.vstack(embeddings_list).astype(np.float32)
        self.dimension = UNIFIED_DIM

        # Inner product on normalised vectors = cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)

        print(f"[dense_retriever] Index built: {self.index.ntotal} vectors, dim={self.dimension}")

    def _merge_search_results(
        self,
        *result_sets: List[Tuple[int, float]],
        k: int,
    ) -> List[Tuple[int, float]]:
        """Merge multiple ranked result lists by max score per document index."""
        combined: dict[int, float] = {}
        for results in result_sets:
            for idx, score in results:
                combined[idx] = max(combined.get(idx, float("-inf")), score)

        return sorted(combined.items(), key=lambda item: item[1], reverse=True)[:k]

    def _apply_score_filter(
        self,
        results: List[Tuple[int, float]],
        query: str = "",
    ) -> List[Tuple[int, float]]:
        """Drop results below effective MIN_RETRIEVAL_SCORE."""
        threshold = Config.get_effective_retrieval_threshold(query)
        filtered_results, min_score_found = filter_by_score(results, threshold, query=query)
        if not filtered_results:
            print(
                f"[dense_retriever] No results above threshold {threshold:.3f}. "
                f"Lowest score: {min_score_found:.3f}"
            )
            return []
        return filtered_results

    def search(
        self,
        query: str,
        k: int = None,
    ) -> List[Tuple[int, float]]:
        """
        Search for top-k most similar chunks.

        Uses SentenceTransformer for text/table matches and CLIP text embeddings
        for image matches, then merges both result sets.

        Args:
            query: Query string.
            k: Number of candidates to return (default: Config.TOP_K_DENSE).

        Returns:
            List of (doc_index, cosine_score) sorted by score descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = k or Config.TOP_K_DENSE
        k = min(k, self.index.ntotal)

        text_query = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        text_query_np = _project_to_512(text_query[0]).reshape(1, -1).astype(np.float32)
        text_scores, text_indices = self.index.search(text_query_np, k)
        text_results = [
            (int(idx), float(score))
            for idx, score in zip(text_indices[0], text_scores[0])
            if idx >= 0
        ]

        # Score filtering
        text_results = self._apply_score_filter(text_results, query=query)
        if not text_results:
            return []

        has_image_docs = any(
            doc.metadata.get("content_type") == "image" for doc in self.documents
        )
        if not has_image_docs:
            return text_results[:k]

        try:
            clip_query_np = embed_query_text_for_clip(query).reshape(1, -1).astype(np.float32)
            clip_scores, clip_indices = self.index.search(clip_query_np, k)
            clip_results = [
                (int(idx), float(score))
                for idx, score in zip(clip_indices[0], clip_scores[0])
                if idx >= 0
            ]
            merged = self._merge_search_results(text_results, clip_results, k=k)
            return self._apply_score_filter(merged, query=query)[:k]
        except Exception as e:
            print(f"[dense_retriever] CLIP search skipped: {e}")
            return text_results[:k]

    def get_document(self, idx: int) -> Document:
        """Return document by index."""
        return self.documents[idx]

    @property
    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def get_stats(self) -> dict:
        return {
            "total_chunks": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "model": self.model_name,
            "clip_model": CLIP_MODEL_NAME,
            "unified_dimension": UNIFIED_DIM,
        }

    def save_index(self, path: str) -> None:
        """Persist FAISS index and document list to disk."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        print(f"[dense_retriever] Saved index to {path}")

    def load_index(self, path: str) -> bool:
        """Load FAISS index and document list from disk."""
        faiss_path = os.path.join(path, "faiss.index")
        docs_path = os.path.join(path, "documents.pkl")
        if not os.path.exists(faiss_path) or not os.path.exists(docs_path):
            return False

        self.index = faiss.read_index(faiss_path)
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        self.dimension = self.index.d
        print(f"[dense_retriever] Loaded index from {path}: {self.index.ntotal} vectors")
        return True
