"""
Structured RAG tracing for Gemini-backed pipelines.
Writes JSONL records to ./logs/rag_traces.jsonl
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_PATH = Path("./logs/rag_traces.jsonl")


def _serialize_scores(chunks_scores: Optional[List]) -> List[Dict[str, Any]]:
    if not chunks_scores:
        return []
    serialized = []
    for item in chunks_scores:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            serialized.append({"index": item[0], "score": float(item[1])})
        elif isinstance(item, dict):
            serialized.append(item)
    return serialized


def serialize_retrieved_docs(docs: Optional[List]) -> List[Dict[str, Any]]:
    """
    Serialize retrieved LangChain documents for JSONL traces.

    Includes cosine ``score`` (for threshold_optimizer) plus explicit
    ``retrieval_score`` and ``reranker_score`` when present in metadata.
    """
    if not docs:
        return []
    serialized: List[Dict[str, Any]] = []
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        item: Dict[str, Any] = {
            "source": meta.get("source"),
            "page": meta.get("page"),
        }
        cosine = meta.get("retrieval_score")
        if cosine is not None:
            item["score"] = float(cosine)
            item["retrieval_score"] = float(cosine)
        reranker = meta.get("reranker_score")
        if reranker is not None:
            item["reranker_score"] = float(reranker)
        serialized.append(item)
    return serialized


def log_trace(
    trace_id: str,
    query: str,
    path: str,
    chunks_scores: Optional[List] = None,
    answer: str = "",
    latency: float = 0.0,
    verification_results: Optional[Dict[str, Any]] = None,
    gemini_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Append a structured trace record for a RAG request.

    Args:
        trace_id: Correlation id (generated if empty).
        query: User or sub-query text.
        path: Pipeline label, e.g. "chat", "agent", "dense_retriever".
        chunks_scores: Retrieval scores as (index, score) pairs or dicts.
        answer: Generated answer text (truncated in log).
        latency: End-to-end seconds.
        verification_results: Relevance/groundedness outcomes.
        gemini_metadata: safety_ratings, finish_reason, etc.

    Returns:
        trace_id used for the record.
    """
    trace_id = trace_id or str(uuid.uuid4())
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "trace_id": trace_id,
        "ts": time.time(),
        "path": path,
        "query": query,
        "latency_ms": round(latency * 1000, 2),
        "chunks_scores": _serialize_scores(chunks_scores),
        "answer_preview": (answer or "")[:500],
        "verification": verification_results or {},
        "gemini": gemini_metadata or {},
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return trace_id


def new_trace_id() -> str:
    return str(uuid.uuid4())
