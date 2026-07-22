"""
Multi-modal retrieval extending HybridRetriever with CLIP fusion search.
Text and image queries share the unified 512-dim FAISS vector store.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from langchain_core.documents import Document
from PIL import Image

from config import Config
from retrieval.dense_retriever import (
    UNIFIED_DIM,
    DenseRetriever,
    _get_clip,
    _l2_normalize_tensor,
    _project_to_512,
    embed_query_text_for_clip,
)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.score_filter import filter_by_score
from retrieval.sparse_retriever import SparseRetriever

import torch


TEXT_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4


class MultimodalRetriever(HybridRetriever):
    """
    Hybrid retriever with CLIP-based multimodal query fusion.
    Supports text-only, image-only, and fused text+image queries.
    """

    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever):
        super().__init__(dense_retriever, sparse_retriever)

    def _encode_clip_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """Encode PIL Image or file path with CLIP image encoder."""
        model, processor = _get_clip()
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = _l2_normalize_tensor(features)
        return _project_to_512(features.squeeze(0).cpu().numpy())

    def _encode_clip_text(self, query: str) -> np.ndarray:
        return embed_query_text_for_clip(query)

    def _fuse_embeddings(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
        text_weight: float = TEXT_WEIGHT,
        image_weight: float = IMAGE_WEIGHT,
    ) -> np.ndarray:
        fused = text_weight * text_embedding + image_weight * image_embedding
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        return fused.astype(np.float32)

    def _vector_search(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[Tuple[int, float]]:
        if self.dense.index is None:
            raise RuntimeError("Dense index not built.")

        k = min(k, self.dense.index.ntotal)
        query_np = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.dense.index.search(query_np, k)
        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0
        ]
        filtered, _ = filter_by_score(results, Config.MIN_RETRIEVAL_SCORE)
        return filtered[:k]

    def search_multimodal(
        self,
        query: str,
        image_query: Optional[Union[Image.Image, str]] = None,
        k: int = None,
    ) -> List[Document]:
        """
        Multimodal search with optional image query fusion.

        Args:
            query: Text query string.
            image_query: Optional PIL Image or path for visual search.
            k: Final number of parent documents to return.

        Returns:
            List of parent Documents (same as hybrid search output).
        """
        k_final = k or Config.TOP_K_FINAL
        k_dense = Config.TOP_K_DENSE

        if image_query is not None:
            text_emb = self._encode_clip_text(query or "image search")
            image_emb = self._encode_clip_image(image_query)
            query_embedding = self._fuse_embeddings(text_emb, image_emb)
            dense_results = self._vector_search(query_embedding, k=k_dense)
            print(f"[multimodal_retriever] Fused CLIP search: {len(dense_results)} hits")
        else:
            dense_results = self.dense.search(query, k=k_dense)

        if not dense_results and image_query is None:
            return super().search(query, k=k_final)

        sparse_results = self.sparse.search(query, k=Config.TOP_K_SPARSE) if query else []

        rrf_constant = Config.RRF_K
        rrf_scores: dict[int, float] = {}
        for rank, (doc_idx, _) in enumerate(dense_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_constant + rank + 1))
        for rank, (doc_idx, _) in enumerate(sparse_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_constant + rank + 1))

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        unique_parents: dict[str, Document] = {}

        for doc_idx, score in sorted_indices:
            child_doc = self.dense.get_document(doc_idx)
            pid = child_doc.metadata.get("parent_id")
            if pid and pid in self.parent_store and pid not in unique_parents:
                parent = self.parent_store[pid]
                meta = {
                    **parent.metadata,
                    "child_rrf_score": score,
                    "retrieval_score": score,
                    "multimodal": image_query is not None,
                }
                unique_parents[pid] = Document(page_content=parent.page_content, metadata=meta)
            if len(unique_parents) >= k_final * 2:
                break

        candidates = list(unique_parents.values())
        if self.reranker and candidates and query:
            pairs = [[query, doc.page_content] for doc in candidates]
            scores = self.reranker.predict(pairs)
            for doc, score in zip(candidates, scores):
                doc.metadata["reranker_score"] = float(score)
            candidates.sort(key=lambda x: x.metadata.get("reranker_score", 0), reverse=True)
        else:
            candidates.sort(key=lambda x: x.metadata.get("child_rrf_score", 0), reverse=True)

        return candidates[:k_final]

    def search(self, query: str, k: int = None) -> List[Document]:
        """Default to standard hybrid search; use search_multimodal for images."""
        from retrieval.query_cache import get_query_cache

        cache = get_query_cache()
        if cache.enabled:
            cached = cache.get(query, extra="hybrid")
            if cached is not None:
                print(f"[multimodal_retriever] Cache hit for '{query[:40]}...'")
                return cached

        docs = super().search(query, k=k)
        if cache.enabled:
            serializable = [
                {"page_content": d.page_content, "metadata": d.metadata} for d in docs
            ]
            cache.set(query, serializable, extra="hybrid")
            docs = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in serializable
            ]
        return docs
