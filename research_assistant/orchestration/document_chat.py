"""
Document chat — NotebookLM-style Q&A over indexed sources.
Each turn retrieves fresh context from the hybrid index and uses chat history for follow-ups.
"""

import asyncio
import time
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from config import Config
from observability.gemini_tracer import log_trace, new_trace_id, serialize_retrieved_docs
from orchestration.simple_summarizer import format_llm_error
from retrieval.hybrid_retriever import HybridRetriever

try:
    from retrieval.query_utils import normalize_technical_query
    HAS_QUERY_UTILS = True
except ImportError:
    HAS_QUERY_UTILS = False

try:
    from memory.user_memory import boost_query, format_preference_prompt, update_from_interaction
    HAS_USER_MEMORY = True
except ImportError:
    HAS_USER_MEMORY = False

try:
    from optimization.prompt_optimizer import get_prompt, record_ab_outcome
    HAS_PROMPT_OPTIMIZER = True
except ImportError:
    HAS_PROMPT_OPTIMIZER = False

try:
    from verification.gemini_verifier import GeminiVerifier
    HAS_GEMINI_VERIFIER = True
except ImportError:
    HAS_GEMINI_VERIFIER = False

ChatHistory = List[dict]


def _message_text(msg) -> Tuple[str, str]:
    """Return (role, content) from Gradio message dict or legacy tuple."""
    if isinstance(msg, dict):
        return msg.get("role", "user"), msg.get("content", "")
    if isinstance(msg, (list, tuple)) and len(msg) == 2:
        return "user", str(msg[0])
    return "user", str(msg)


def _format_history(history: ChatHistory, max_turns: int = 6) -> str:
    if not history:
        return "_No prior messages._"

    lines = []
    # Gradio 6: flat list of {role, content} messages
    if history and isinstance(history[0], dict):
        for msg in history[-max_turns * 2 :]:
            role, content = _message_text(msg)
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        return "\n".join(lines)

    # Legacy tuple pairs
    for user_msg, bot_msg in history[-max_turns:]:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {bot_msg}")
    return "\n".join(lines)


def _append_turn(history: ChatHistory, question: str, answer: str) -> ChatHistory:
    history = history or []
    return history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _build_context_from_docs(docs: List) -> Tuple[str, str]:
    context_parts = []
    source_lines = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        header = f"[{source}" + (f", page {page}" if page else "") + "]"
        text = doc.page_content.strip()

        if doc.metadata.get("content_type") == "table":
            text = f"[TABLE]\n{text}"

        if text:
            context_parts.append(f"{header}\n{text}")
            source_lines.append(f"- {source}" + (f" (page {page})" if page else ""))

    context = "\n\n---\n\n".join(context_parts)
    sources = "\n".join(dict.fromkeys(source_lines))
    return context, sources


def _strictness_suffix(attempt: int) -> str:
    """Escalate grounding instructions on each regeneration attempt."""
    if attempt == 0:
        return ""
    if attempt == 1:
        return (
            "\n\nCRITICAL: Use ONLY the above text. "
            "Say 'I don't know' if the context is insufficient."
        )
    return (
        "\n\nCRITICAL (FINAL ATTEMPT): Answer ONLY with verbatim facts from the context. "
        "No inference, no external knowledge. If unsure, respond: "
        "'I cannot find that in the provided documents.'"
    )


async def _regenerate_until_grounded(
    verifier: "GeminiVerifier",
    chain,
    question: str,
    history: ChatHistory,
    context: str,
    initial_answer: str,
    verification_results: Dict[str, Any],
) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Retry generation up to Config.MAX_RETRIES when groundedness fails.
    Uses exponential backoff between attempts (Gemini rate-limit friendly).
    """
    answer = initial_answer
    grounded = await verifier.is_grounded(answer, context)
    gemini_metadata = verifier.last_metadata
    attempts_log: List[Dict[str, Any]] = [{"attempt": 1, "grounded": grounded}]

    for retry_idx in range(1, Config.MAX_RETRIES):
        if grounded:
            break

        delay = Config.RETRY_BACKOFF * (2 ** (retry_idx - 1))
        print(
            f"[document_chat] Groundedness failed (attempt {retry_idx}/{Config.MAX_RETRIES}) "
            f"— regenerating in {delay:.1f}s"
        )
        await asyncio.sleep(delay)

        strict_context = context + _strictness_suffix(retry_idx)
        response = await chain.ainvoke(
            {
                "history": _format_history(history or []),
                "context": strict_context,
                "question": question,
            }
        )
        answer = response.content if hasattr(response, "content") else str(response)
        grounded = await verifier.is_grounded(answer, strict_context)
        gemini_metadata = verifier.last_metadata
        attempts_log.append({"attempt": retry_idx + 1, "grounded": grounded})

    verification_results["grounded"] = grounded
    verification_results["grounded_attempts"] = attempts_log
    verification_results["regeneration_count"] = max(0, len(attempts_log) - 1)
    if grounded and len(attempts_log) > 1:
        verification_results["grounded_after_retry"] = True
    return answer, grounded, gemini_metadata


async def answer_document_question(
    question: str,
    history: ChatHistory,
    llm,
    retriever: HybridRetriever,
    user_id: str = "anonymous",
) -> Dict[str, str]:
    """Answer a user question using RAG + optional conversation history.

    Returns dict with keys: answer, sources, trace_id.
    """
    trace_id = new_trace_id()
    started = time.time()
    verification_results: Dict[str, object] = {}
    gemini_metadata: Dict[str, object] = {}

    prior_query = ""
    if history and len(history) >= 2:
        _, prior_query = _message_text(history[-2])

    if HAS_USER_MEMORY and Config.LEVEL5_ENABLED:
        update_from_interaction(
            user_id,
            question,
            is_follow_up=len(history or []) >= 2,
            prior_query=prior_query,
        )
        search_query = boost_query(user_id, question)
    else:
        search_query = question

    if HAS_QUERY_UTILS:
        search_query = normalize_technical_query(search_query)

    loop = asyncio.get_event_loop()
    docs: List = await loop.run_in_executor(None, retriever.search, search_query)
    retrieved_docs: List = list(docs)

    if not docs:
        log_trace(
            trace_id=trace_id,
            query=question,
            path="chat",
            chunks_scores=[],
            answer="No documents retrieved.",
            latency=time.time() - started,
            verification_results={"retrieval": "empty"},
        )
        return {
            "answer": (
                "I don't have enough confidence in the retrieved passages to answer that. "
                "Try rephrasing, or rebuild the index from **📚 Data Sources**."
            ),
            "sources": "",
            "trace_id": trace_id,
        }

    verifier = None
    if Config.LLM_BACKEND == "gemini" and HAS_GEMINI_VERIFIER:
        try:
            verifier = GeminiVerifier.from_config()
        except Exception as exc:
            print(f"[document_chat] GeminiVerifier unavailable: {exc}")

    if verifier:
        relevant_docs = []
        for doc in docs:
            # Use FAISS cosine only for score pre-filter — NOT reranker_score (different scale)
            cosine = doc.metadata.get("retrieval_score")
            if await verifier.is_relevant(
                question,
                doc.page_content,
                score=cosine,
                score_is_cosine=cosine is not None,
            ):
                relevant_docs.append(doc)

        # Fallback: retrieval was confident but Gemini rejected all (API error or over-strict)
        if not relevant_docs and docs:
            fallback = [
                d for d in docs
                if (d.metadata.get("reranker_score") or 0) >= 0.5
                or (d.metadata.get("retrieval_score") or 0) >= Config.MIN_RETRIEVAL_SCORE_LOW
            ]
            if fallback:
                print(
                    f"[document_chat] Relevance fallback: using top {len(fallback[:3])} "
                    f"high-confidence docs (reranker/cosine)"
                )
                relevant_docs = fallback[:3]
                verification_results["relevance_fallback"] = True

        verification_results["relevance_passed"] = len(relevant_docs)
        verification_results["relevance_total"] = len(docs)
        gemini_metadata = verifier.last_metadata
        docs = relevant_docs

    if not docs:
        log_trace(
            trace_id=trace_id,
            query=question,
            path="chat",
            chunks_scores=serialize_retrieved_docs(retrieved_docs),
            answer="No relevant documents after verification.",
            latency=time.time() - started,
            verification_results=verification_results,
            gemini_metadata=gemini_metadata,
        )
        return {
            "answer": (
                "I couldn't verify that any retrieved passage is relevant to your question. "
                "Please rephrase or upload a more specific document."
            ),
            "sources": "",
            "trace_id": trace_id,
        }

    context, sources = _build_context_from_docs(docs)
    if not context.strip():
        log_trace(
            trace_id=trace_id,
            query=question,
            path="chat",
            chunks_scores=[],
            answer="No readable text in retrieved sections.",
            latency=time.time() - started,
            verification_results=verification_results,
            gemini_metadata=gemini_metadata,
        )
        return {
            "answer": "The index exists but retrieved sections had no readable text.",
            "sources": "",
            "trace_id": trace_id,
        }

    gen_template_name = "default_v1"
    gen_system_prompt = (
        "You are a document assistant (like NotebookLM). "
        "Answer the user's question using ONLY the document context below. "
        "Use simple, clear language. Mention page/source when helpful. "
        "If the answer is not in the context, say you don't know — do not guess. "
        "For follow-up questions, use the conversation history too."
    )
    if HAS_PROMPT_OPTIMIZER and Config.LEVEL5_ENABLED:
        gen_template_name, gen_system_prompt = get_prompt("generation", trace_id=trace_id)

    preference_block = ""
    if HAS_USER_MEMORY and Config.LEVEL5_ENABLED:
        preference_block = format_preference_prompt(user_id)
    system_content = gen_system_prompt
    if preference_block:
        system_content = f"{gen_system_prompt}\n\n{preference_block}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_content),
            (
                "human",
                "Conversation so far:\n{history}\n\n"
                "Document context:\n{context}\n\n"
                "User question: {question}\n\n"
                "Answer:",
            ),
        ]
    )

    chain = prompt | llm
    response = await chain.ainvoke(
        {
            "history": _format_history(history or []),
            "context": context,
            "question": question,
        }
    )
    answer = response.content if hasattr(response, "content") else str(response)

    if verifier:
        answer, grounded, gemini_metadata = await _regenerate_until_grounded(
            verifier=verifier,
            chain=chain,
            question=question,
            history=history,
            context=context,
            initial_answer=answer,
            verification_results=verification_results,
        )
        if not grounded:
            print(
                f"[document_chat] Groundedness failed after {Config.MAX_RETRIES} attempts "
                f"(trace_id={trace_id})"
            )

    if HAS_PROMPT_OPTIMIZER and Config.LEVEL5_ENABLED:
        grounded_ok = verification_results.get("grounded", True)
        record_ab_outcome("generation", gen_template_name, bool(grounded_ok))

    log_trace(
        trace_id=trace_id,
        query=question,
        path="chat",
        chunks_scores=serialize_retrieved_docs(retrieved_docs),
        answer=answer,
        latency=time.time() - started,
        verification_results=verification_results,
        gemini_metadata=gemini_metadata,
    )

    return {
        "answer": answer,
        "sources": f"**Sources used:**\n{sources}" if sources else "",
        "trace_id": trace_id,
    }


async def chat_turn(
    question: str,
    history: ChatHistory,
    llm,
    retriever: HybridRetriever,
    user_id: str = "anonymous",
) -> Tuple[ChatHistory, str, str]:
    """Process one chat turn; returns updated history, source panel text, and trace_id."""
    history = history or []
    question = (question or "").strip()
    if not question:
        return history, "", ""

    try:
        result = await answer_document_question(
            question, history, llm, retriever, user_id=user_id
        )
        return (
            _append_turn(history, question, result["answer"]),
            result["sources"],
            result.get("trace_id", ""),
        )
    except Exception as exc:
        error_msg = format_llm_error(exc)
        return _append_turn(history, question, error_msg), "", ""
