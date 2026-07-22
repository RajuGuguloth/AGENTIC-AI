"""
Gemini-specific relevance and groundedness verification.
Uses google.generativeai with low temperature, safety settings, and retry backoff.
"""

import asyncio
import re
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.generativeai import GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from config import Config


class GeminiQuotaExhaustedError(Exception):
    """Raised when Gemini daily quota is exceeded."""


RELEVANCE_PROMPT = """You are a relevance grader for a document Q&A system. Answer ONLY 'YES' or 'NO'.

Question: {question}
Context: {context}

Is the context useful for answering the question?
- YES if it defines, describes, or discusses the topic (including synonyms like self-attention vs self attention)
- YES if it contains related technical content that helps answer the question
- NO only if the context is completely unrelated

Answer (YES/NO):"""

GROUNDEDNESS_PROMPT = """You are a hallucination detector. Answer ONLY 'YES' or 'NO'.

Context: {context}
Answer: {answer}

Does the answer contain ANY information NOT in the context?
- If answer has info NOT in context → NO
- If answer is fully supported → YES

Answer (YES/NO):"""


def _default_safety_settings() -> Dict[HarmCategory, HarmBlockThreshold]:
    threshold_name = Config.GEMINI_SAFETY_THRESHOLD
    threshold = getattr(HarmBlockThreshold, threshold_name, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
    categories = [
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    ]
    return {cat: threshold for cat in categories}


def _parse_yes_no(text: str) -> bool:
    normalized = (text or "").strip().upper()
    if re.search(r"\bYES\b", normalized):
        return True
    if re.search(r"\bNO\b", normalized):
        return False
    return False


def _is_daily_quota_error(message: str) -> bool:
    lower = (message or "").lower()
    normalized = lower.replace("_", "").replace("-", "")
    return "429" in lower and "perday" in normalized


def _extract_response_text(response) -> str:
    """Extract text from Gemini response without raising on blocked/empty parts."""
    try:
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()
    except Exception:
        pass
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = []
        for part in parts:
            t = getattr(part, "text", None)
            if t:
                chunks.append(str(t))
        if chunks:
            return " ".join(chunks).strip()
    return ""


def _extract_gemini_metadata(response) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if response is None:
        return meta
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        meta["finish_reason"] = str(getattr(candidate, "finish_reason", ""))
        safety = getattr(candidate, "safety_ratings", None) or []
        meta["safety_ratings"] = [
            {
                "category": str(getattr(r, "category", "")),
                "probability": str(getattr(r, "probability", "")),
            }
            for r in safety
        ]
    return meta


class GeminiVerifier:
    """Unified Gemini verification for relevance and groundedness."""

    def __init__(self, model: GenerativeModel):
        self.model = model
        self._last_metadata: Dict[str, Any] = {}

    @classmethod
    def from_config(cls) -> "GeminiVerifier":
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for GeminiVerifier")
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = GenerativeModel(
            model_name=Config.GEMINI_MODEL,
            safety_settings=_default_safety_settings(),
            generation_config=genai.GenerationConfig(
                temperature=Config.GEMINI_TEMPERATURE,
                max_output_tokens=32,
            ),
        )
        return cls(model)

    @property
    def last_metadata(self) -> Dict[str, Any]:
        return self._last_metadata

    async def _generate_with_retry(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                self._last_metadata = _extract_gemini_metadata(response)
                return _extract_response_text(response)
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                is_rate_limit = "429" in msg or "resource_exhausted" in msg or "quota" in msg
                if attempt < Config.MAX_RETRIES - 1 and is_rate_limit:
                    if _is_daily_quota_error(msg):
                        raise GeminiQuotaExhaustedError(str(exc)) from exc
                    delay = max(Config.RETRY_BACKOFF * (2 ** attempt), 15.0)
                    import re
                    retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
                    if retry_match:
                        delay = float(retry_match.group(1)) + 2.0
                    print(f"[gemini_verifier] Rate limit — retry {attempt + 1}/{Config.MAX_RETRIES} in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                if _is_daily_quota_error(msg):
                    raise GeminiQuotaExhaustedError(str(exc)) from exc
                raise
        raise last_error  # type: ignore[misc]

    async def is_relevant(
        self,
        question: str,
        context: str,
        score: float = None,
        *,
        score_is_cosine: bool = True,
    ) -> bool:
        # Only apply cosine threshold to FAISS similarity scores (0–1), NOT reranker logits
        if (
            score is not None
            and score_is_cosine
            and 0.0 <= score <= 1.0
            and score < Config.get_effective_retrieval_threshold(question)
        ):
            return False
        if not (context or "").strip():
            return False

        prompt = RELEVANCE_PROMPT.format(
            question=question,
            context=(context or "")[:500],
        )
        try:
            text = await self._generate_with_retry(prompt)
            return _parse_yes_no(text)
        except GeminiQuotaExhaustedError:
            raise
        except Exception as exc:
            print(f"[gemini_verifier] Relevance check failed (fail-closed): {exc}")
            return False

    async def is_grounded(self, answer: str, context: str) -> bool:
        if not (answer or "").strip():
            return False
        if not (context or "").strip():
            return False

        prompt = GROUNDEDNESS_PROMPT.format(
            context=(context or "")[:800],
            answer=(answer or "")[:800],
        )
        try:
            text = await self._generate_with_retry(prompt)
            return _parse_yes_no(text)
        except GeminiQuotaExhaustedError:
            raise
        except Exception as exc:
            print(f"[gemini_verifier] Groundedness check failed (fail-closed): {exc}")
            return False

    async def verify_batch(
        self,
        question: str,
        chunks: List[Any],
        answers: List[str],
    ) -> List[bool]:
        """
        Verify relevance for each chunk and groundedness for each answer.
        Returns a flat list: [chunk_relevance..., answer_groundedness...]
        """
        results: List[bool] = []

        for chunk in chunks:
            content = getattr(chunk, "page_content", str(chunk))
            score = None
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                score = chunk.metadata.get("retrieval_score")
            results.append(
                await self.is_relevant(
                    question, content, score=score, score_is_cosine=score is not None
                )
            )

        for answer in answers:
            combined_context = "\n\n".join(
                getattr(c, "page_content", str(c))[:500] for c in chunks
            )
            results.append(await self.is_grounded(answer, combined_context))

        return results
