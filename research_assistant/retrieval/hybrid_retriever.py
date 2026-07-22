"""
Retrieval — Hybrid Retriever (FAISS + BM25 + RRF) + Cross-Encoder Reranker
Fuses results from dense and sparse retrievers using Reciprocal Rank Fusion (RRF) on child chunks.
Fetches Parent documents and applies Cross-Encoder reranking.
"""

from typing import List, Dict
import os
import pickle
from langchain_core.documents import Document
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from config import Config

# Try to import CrossEncoder, handle if not installed
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


class HybridRetriever:
    """
    Combines FAISS (Dense) and BM25 (Sparse) using RRF on Child chunks,
    fetches Parent documents, and reranks them using a Cross-Encoder.
    """

    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.parent_store: Dict[str, Document] = {}
        
        if HAS_CROSS_ENCODER:
            print("[hybrid_retriever] Initializing CrossEncoder (BAAI/bge-reranker-large)...")
            self.reranker = CrossEncoder('BAAI/bge-reranker-large', max_length=512)
        else:
            self.reranker = None
            print("[hybrid_retriever] Warning: sentence-transformers not found, reranking disabled.")

    def add_parents(self, parent_docs: List[Document]):
        """Store parent documents for retrieval by ID."""
        for doc in parent_docs:
            pid = doc.metadata.get("parent_id")
            if pid:
                self.parent_store[pid] = doc

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Execute hybrid search with Parent-Child mapping and Reranking.
        
        Algorithm:
        1. Query dense/sparse on children -> RRF fuse -> top child chunks.
        2. Map child chunks to Parent documents (deduplicating).
        3. Cross-Encoder rerank the Parent documents against the query.
        """
        if not self.dense.is_ready or not self.sparse.is_ready:
            raise RuntimeError("Retrievers are not ready. Build index first.")

        k_final = k or Config.TOP_K_FINAL
        k_dense = Config.TOP_K_DENSE
        k_sparse = Config.TOP_K_SPARSE
        rrf_constant = Config.RRF_K

        print(f"[hybrid_retriever] Query: '{query}'")
        
        # 1. Get child chunk rankings (dense scores are cosine similarity in [0, 1])
        dense_results = self.dense.search(query, k=k_dense)
        dense_cosine_map = {idx: score for idx, score in dense_results}
        sparse_results = self.sparse.search(query, k=k_sparse)
        
        # 2. Calculate RRF scores for children
        rrf_scores: Dict[int, float] = {}
        for rank, (doc_idx, _) in enumerate(dense_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_constant + rank + 1))
        for rank, (doc_idx, _) in enumerate(sparse_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (rrf_constant + rank + 1))

        # Sort children by RRF
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Map to Parent Documents (Deduplicate)
        unique_parents: Dict[str, Document] = {}
        
        for doc_idx, score in sorted_indices:
            child_doc = self.dense.get_document(doc_idx)
            pid = child_doc.metadata.get("parent_id")
            
            if pid and pid in self.parent_store:
                if pid not in unique_parents:
                    parent_doc = self.parent_store[pid]
                    cosine = dense_cosine_map.get(doc_idx)
                    cloned_parent = Document(
                        page_content=parent_doc.page_content,
                        metadata={
                            **parent_doc.metadata,
                            "child_rrf_score": score,
                            "retrieval_score": cosine,
                        },
                    )
                    unique_parents[pid] = cloned_parent
                elif doc_idx in dense_cosine_map:
                    existing = unique_parents[pid]
                    prev = existing.metadata.get("retrieval_score") or 0.0
                    existing.metadata["retrieval_score"] = max(
                        prev, dense_cosine_map[doc_idx]
                    )
            
            # Optimization: If we have enough unique parents, we can stop early
            if len(unique_parents) >= k_final * 2: # Fetch 2x for reranking
                break
                
        candidate_parents = list(unique_parents.values())
        print(f"[hybrid_retriever] Mapped to {len(candidate_parents)} unique Parent documents.")

        # 4. Cross-Encoder Reranking
        if self.reranker and candidate_parents:
            print("[hybrid_retriever] Reranking Parents with CrossEncoder...")
            # Prepare pairs: (query, document_text)
            pairs = [[query, doc.page_content] for doc in candidate_parents]
            scores = self.reranker.predict(pairs)
            
            # Inject scores into metadata and sort
            for doc, score in zip(candidate_parents, scores):
                doc.metadata["reranker_score"] = float(score)
                
            candidate_parents.sort(key=lambda x: x.metadata["reranker_score"], reverse=True)
        else:
            # Fallback to sorting by child_rrf_score if no reranker
            candidate_parents.sort(key=lambda x: x.metadata.get("child_rrf_score", 0), reverse=True)

        # 5. Return top k final Parent documents
        final_docs = candidate_parents[:k_final]
        print(f"[hybrid_retriever] Returned top {len(final_docs)} reranked Parent documents.")
        return final_docs

    def save(self, dir: str) -> None:
        """Persist dense, sparse, and parent store indexes to disk."""
        os.makedirs(dir, exist_ok=True)
        self.dense.save_index(dir)
        self.sparse.save_index(dir)
        with open(os.path.join(dir, "parent_store.pkl"), "wb") as f:
            pickle.dump(self.parent_store, f)
        print(f"[hybrid_retriever] Saved hybrid index to {dir}")

    def load(self, dir: str) -> bool:
        """Load all indexes from disk if they exist."""
        parent_path = os.path.join(dir, "parent_store.pkl")
        if not os.path.exists(parent_path):
            return False

        if not self.dense.load_index(dir):
            return False
        if not self.sparse.load_index(dir):
            return False

        with open(parent_path, "rb") as f:
            self.parent_store = pickle.load(f)

        self.sparse.documents = self.dense.documents
        print(f"[hybrid_retriever] Loaded hybrid index from {dir}")
        return True
