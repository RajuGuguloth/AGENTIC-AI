"""Process-wide retriever and index lifecycle shared by the app and API."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from retrieval.dense_retriever import DenseRetriever
from retrieval.multimodal_retriever import MultimodalRetriever
from retrieval.sparse_retriever import SparseRetriever

_runtime_lock = threading.RLock()
_hybrid_retriever: MultimodalRetriever | None = None
_index_store_path = Path(
    os.getenv(
        "RAG_INDEX_PATH",
        str(Path(__file__).resolve().parents[1] / "index_store"),
    )
).expanduser().resolve()


def get_index_store_path() -> Path:
    return _index_store_path


def get_runtime_lock() -> threading.RLock:
    return _runtime_lock


def get_hybrid_retriever() -> MultimodalRetriever:
    """Load the persisted retrieval engine once in this process."""
    global _hybrid_retriever
    with _runtime_lock:
        if _hybrid_retriever is None:
            dense = DenseRetriever()
            sparse = SparseRetriever()
            retriever = MultimodalRetriever(dense, sparse)
            if retriever.load(str(_index_store_path)):
                print(f"[rag_runtime] Restored index from {_index_store_path}")
            else:
                print(f"[rag_runtime] No persisted index at {_index_store_path}")
            _hybrid_retriever = retriever
        return _hybrid_retriever


def rebuild_index(
    parent_docs: List[Document],
    child_docs: List[Document],
    *,
    replace: bool = True,
) -> dict:
    """Persist a replacement index or append documents to the current index."""
    if not parent_docs or not child_docs:
        raise ValueError("Cannot index an empty set of parent or child documents.")

    retriever = get_hybrid_retriever()
    with _runtime_lock:
        if replace or not retriever.dense.is_ready:
            indexed_parents = list(parent_docs)
            indexed_children = list(child_docs)
        else:
            indexed_parents = list(retriever.parent_store.values()) + list(parent_docs)
            indexed_children = list(retriever.dense.documents) + list(child_docs)

        retriever.dense.build_index(indexed_children)
        retriever.sparse.build_index(indexed_children)
        retriever.parent_store.clear()
        retriever.add_parents(indexed_parents)
        retriever.save(str(_index_store_path))

        from retrieval.query_cache import get_query_cache

        get_query_cache().clear()

        return {
            "parents_indexed": len(parent_docs),
            "children_indexed": len(child_docs),
            "total_parents": len(indexed_parents),
            "total_children": len(indexed_children),
            "index_path": str(_index_store_path),
            "replace": replace,
        }
