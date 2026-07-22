"""
Retrieval — Sparse Retriever (BM25)
BM25 keyword-based retrieval using rank_bm25.
Produces a ranked list of (doc_index, score) pairs for RRF fusion.

BM25 complements dense retrieval by excelling at:
 - Exact keyword matches (model names, acronyms, technical terms)
 - Out-of-vocabulary tokens not captured by dense embeddings
 - Queries with rare but critical terms
"""

from typing import List, Tuple
import numpy as np
import os
import pickle
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from config import Config


def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace + lowercase tokenizer for BM25.
    Strips punctuation for cleaner matching.
    """
    import re
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return tokens


class SparseRetriever:
    """
    BM25-based retrieval over document chunks.
    Uses BM25Okapi from rank_bm25 with default k1=1.5, b=0.75.
    """

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []

    def build_index(self, documents: List[Document]) -> None:
        """
        Tokenize documents and build BM25 index.

        Args:
            documents: List of LangChain Documents (same list as dense retriever).
        """
        if not documents:
            raise ValueError("Cannot build BM25 index from empty document list.")

        self.documents = documents
        print(f"[sparse_retriever] Building BM25 index over {len(documents)} chunks...")

        self.tokenized_corpus = [
            _tokenize(doc.page_content) for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[sparse_retriever] BM25 index built with {len(documents)} documents.")

    def search(
        self,
        query: str,
        k: int = None,
    ) -> List[Tuple[int, float]]:
        """
        Search BM25 index for top-k relevant chunks.

        Args:
            query: Query string.
            k: Number of candidates (default: Config.TOP_K_SPARSE).

        Returns:
            List of (doc_index, bm25_score) sorted by score descending.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        k = k or Config.TOP_K_SPARSE
        k = min(k, len(self.documents))

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)  # shape: (n_docs,)

        # Get top-k indices by score
        top_k_indices = np.argsort(scores)[::-1][:k]

        results = [
            (int(idx), float(scores[idx]))
            for idx in top_k_indices
            if scores[idx] > 0  # filter zero-score docs
        ]
        return results

    def get_document(self, idx: int) -> Document:
        """Return document by index."""
        return self.documents[idx]

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None

    def get_stats(self) -> dict:
        return {
            "total_chunks": len(self.documents),
            "avg_doc_length": (
                float(np.mean([len(t) for t in self.tokenized_corpus]))
                if self.tokenized_corpus else 0
            ),
        }

    def save_index(self, path: str) -> None:
        """Persist BM25 index and tokenized corpus to disk."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "sparse_bm25.pkl"), "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "tokenized_corpus": self.tokenized_corpus},
                f,
            )
        print(f"[sparse_retriever] Saved index to {path}")

    def load_index(self, path: str) -> bool:
        """Load BM25 index and tokenized corpus from disk."""
        sparse_path = os.path.join(path, "sparse_bm25.pkl")
        if not os.path.exists(sparse_path):
            return False

        with open(sparse_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.tokenized_corpus = data["tokenized_corpus"]
        print(f"[sparse_retriever] Loaded index from {path}: {len(self.tokenized_corpus)} documents")
        return True
