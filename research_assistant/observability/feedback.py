"""
User feedback collection for RAG traces.
Appends structured records to ./logs/feedback.jsonl
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

FEEDBACK_LOG_PATH = Path("./logs/feedback.jsonl")

Rating = Literal["positive", "negative"]


def append_feedback(
    trace_id: str,
    rating: Rating,
    comment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Append a user feedback record linked to a RAG trace_id.

    Args:
        trace_id: Correlation id from log_trace / answer_document_question.
        rating: "positive" or "negative".
        comment: Optional free-text note from the user.
        metadata: Optional extra fields (path, query preview, etc.).

    Returns:
        The written feedback record.
    """
    if not trace_id or not trace_id.strip():
        raise ValueError("trace_id is required")

    normalized_rating = rating.lower().strip()
    if normalized_rating not in ("positive", "negative"):
        raise ValueError("rating must be 'positive' or 'negative'")

    record = {
        "trace_id": trace_id.strip(),
        "rating": normalized_rating,
        "comment": (comment or "").strip(),
        "ts": time.time(),
        "metadata": metadata or {},
    }

    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
