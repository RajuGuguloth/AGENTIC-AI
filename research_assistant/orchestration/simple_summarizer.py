"""
Simple PDF summary — one retrieval pass + one LLM call.
Avoids planner/structured-output paths that often fail with Gemini quota limits.
"""

import asyncio
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate

from retrieval.hybrid_retriever import HybridRetriever


def format_llm_error(exc: Exception) -> str:
    """Turn API errors into user-readable messages."""
    msg = str(exc)
    lower = msg.lower()

    if "429" in msg or "resource_exhausted" in lower or "quota" in lower:
        return (
            "## Gemini API quota exceeded\n\n"
            "Your **PDF was indexed successfully**, but Google blocked the LLM request.\n\n"
            "**Fix:**\n"
            "1. Wait a few minutes and try again, or\n"
            "2. Create a new key at https://aistudio.google.com/apikey (starts with `AIza...`), or\n"
            "3. Enable billing / use a different model in `.env` (`GEMINI_MODEL=gemini-2.5-flash`)\n\n"
            f"_Technical detail:_ {msg[:400]}"
        )

    if "api key" in lower or "401" in msg or "403" in msg or "invalid" in lower and "key" in lower:
        return (
            "## Invalid or missing Gemini API key\n\n"
            "Set a valid key in `research_assistant/.env`:\n"
            "```\nLLM_BACKEND=gemini\nGEMINI_API_KEY=AIza...your-key\n```\n"
            "Get a key: https://aistudio.google.com/apikey\n\n"
            f"_Technical detail:_ {msg[:400]}"
        )

    return f"## Summary failed\n\n{msg}"


async def summarize_from_index(
    goal: str,
    llm,
    retriever: HybridRetriever,
) -> Dict[str, str]:
    """
    Retrieve top chunks and produce one clear summary (no planner / no structured graders).
    """
    loop = asyncio.get_event_loop()
    docs: List = await loop.run_in_executor(None, retriever.search, goal)

    if not docs:
        return {
            "report": (
                "## No content retrieved\n\n"
                "The index exists but no matching text was found. "
                "Try rebuilding the index from your PDF."
            ),
            "findings": "",
            "sub_queries": "_Simple mode: single retrieval + one LLM call_",
        }

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        header = f"[{source}" + (f", page {page}" if page else "") + "]"
        text = doc.page_content.strip()
        if text:
            context_parts.append(f"{header}\n{text}")

    context = "\n\n---\n\n".join(context_parts)
    if not context.strip():
        return {
            "report": "## No text in retrieved chunks\n\nRe-index your PDF and try again.",
            "findings": "",
            "sub_queries": "_Retrieved parent docs had empty content_",
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Summarize using ONLY the context below. "
                "Use simple language, short sections, and bullet points. "
                "If the context is insufficient, say so clearly.",
            ),
            (
                "human",
                "Task: {goal}\n\nContext from the document:\n{context}\n\nWrite the summary:",
            ),
        ]
    )

    chain = prompt | llm
    response = await chain.ainvoke({"goal": goal, "context": context})
    report = response.content if hasattr(response, "content") else str(response)

    sources = "\n".join(
        f"- {d.metadata.get('source', '?')} (page {d.metadata.get('page', '?')})"
        for d in docs
    )

    return {
        "report": report,
        "findings": f"### Sources used\n{sources}",
        "sub_queries": "### Mode\nSimple summarize (1 retrieval + 1 LLM call)",
    }
