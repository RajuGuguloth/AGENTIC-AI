"""
FastAPI router for RAG user feedback.
POST /api/feedback → append to ./logs/feedback.jsonl
"""

from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from observability.feedback import append_feedback

try:
    from config import Config
    from memory.user_memory import resolve_user_id, update_from_feedback

    HAS_USER_MEMORY = True
except ImportError:
    HAS_USER_MEMORY = False

router = APIRouter(prefix="/api", tags=["feedback"])


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., min_length=1, description="RAG trace correlation id")
    rating: Literal["positive", "negative"] = Field(..., description="User satisfaction signal")
    comment: Optional[str] = Field(default="", max_length=2000)
    metadata: Optional[Dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    status: str
    trace_id: str
    rating: str


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback for a completed RAG turn."""
    try:
        record = append_feedback(
            trace_id=body.trace_id,
            rating=body.rating,
            comment=body.comment,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write feedback: {exc}") from exc

    if HAS_USER_MEMORY and getattr(Config, "LEVEL5_ENABLED", False):
        meta = body.metadata or {}
        user_id = meta.get("user_id") or resolve_user_id(session_id=meta.get("session_id", ""))
        update_from_feedback(user_id, body.rating, comment=body.comment or "")

    return FeedbackResponse(
        status="ok",
        trace_id=record["trace_id"],
        rating=record["rating"],
    )
