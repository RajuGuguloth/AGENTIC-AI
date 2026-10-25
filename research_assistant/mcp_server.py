"""
MCP server exposing the Agentic RAG pipeline as native Claude tools.

Run with (stdio transport for Claude Code / Claude Desktop):
    ./.venv/bin/python mcp_server.py

Tools exposed:
    - list_indexed_papers()               → what's currently in the index
    - search_papers(query, top_k)         → raw hybrid-retrieval hits (fast)
    - ask_paper(question)                 → full agentic answer with grounding + sources
    - index_status()                      → readiness + counts
    - build_index_from_paths(paths)       → ingest PDFs, chunk, embed, persist
    - deep_read_paper(paper_url_or_path)  → structured report + PowerPoint

Resources exposed:
    - resource://eval/latest_report       → latest evaluation report JSON
    - resource://papers/{source_id}       → full text of a retrieved parent chunk

Everything else (verifier, self-heal loop, RAGAS eval, memory) lives in the
existing project modules — this file is only the MCP glue.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP
import httpx

# ── Configure structured logging BEFORE importing anything that logs at import
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
MCP_LOG_PATH = LOG_DIR / "mcp_calls.jsonl"

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[mcp %(asctime)s %(levelname)s] %(message)s",
)
log = logging.getLogger("mcp_rag")


def _record_call(tool: str, args: Dict[str, Any], result_summary: Dict[str, Any], latency_ms: float, error: Optional[str] = None) -> str:
    """Append one JSONL trace per MCP call. Returns the trace_id."""
    trace_id = uuid.uuid4().hex[:12]
    entry = {
        "trace_id": trace_id,
        "ts": time.time(),
        "tool": tool,
        "args_preview": {k: (str(v)[:120] if not isinstance(v, (int, float, bool)) else v) for k, v in args.items()},
        "latency_ms": round(latency_ms, 1),
        "result_summary": result_summary,
        "error": error,
    }
    try:
        with MCP_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("failed to persist mcp trace: %s", exc)
    return trace_id


RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:7860").rstrip("/")


async def _api_request(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, str]] = None,
    files=None,
) -> Dict[str, Any]:
    headers = {}
    api_key = os.getenv("RAG_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    timeout = httpx.Timeout(300.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method,
            f"{RAG_API_URL}{path}",
            headers=headers,
            json=json_body,
            data=data,
            files=files,
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:1000]
        raise RuntimeError(
            f"RAG API request failed ({response.status_code}): {detail}"
        ) from exc
    return response.json()


# ── Server definition
mcp = FastMCP(
    name="agentic-rag",
    instructions=(
        "Agentic RAG over indexed research papers. "
        "Prefer ask_paper for user-facing questions (returns a full grounded answer with sources); "
        "use search_papers to preview raw retrieved chunks when the caller needs the evidence directly. "
        "Use deep_read_paper to generate a structured report and slides from a paper URL or local PDF. "
        "Retrieved chunks expose resource://papers/{source_id}; the latest evaluation is available at "
        "resource://eval/latest_report. "
        "Call index_status first if you're unsure whether an index exists."
    ),
)


@mcp.resource("resource://eval/latest_report", mime_type="application/json")
def latest_evaluation_report() -> str:
    """Read the latest RAG evaluation report as JSON."""
    t0 = time.time()
    report_path = ROOT / "evaluation" / "orchestrator_report.json"
    try:
        report = report_path.read_text(encoding="utf-8")
        _record_call(
            "resource://eval/latest_report",
            {},
            {"characters": len(report)},
            (time.time() - t0) * 1000,
        )
        return report
    except Exception as exc:
        _record_call(
            "resource://eval/latest_report",
            {},
            {},
            (time.time() - t0) * 1000,
            error=str(exc),
        )
        raise


@mcp.resource("resource://papers/{source_id}")
async def parent_chunk_resource(source_id: str) -> str:
    """Read the full text of a parent chunk returned by search_papers."""
    t0 = time.time()
    try:
        response = await _api_request("GET", f"/v1/chunks/{quote(source_id, safe='')}")
        content = response.get("text", "")
        _record_call(
            "resource://papers/{source_id}",
            {"source_id": source_id},
            {"characters": len(content)},
            (time.time() - t0) * 1000,
        )
        return content
    except Exception as exc:
        _record_call(
            "resource://papers/{source_id}",
            {"source_id": source_id},
            {},
            (time.time() - t0) * 1000,
            error=str(exc),
        )
        raise

# ────────────────────────────────────────────────────────────────────────────
# Tool 1: index_status
# ────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def index_status() -> Dict[str, Any]:
    """
    Report whether the RAG index is ready to serve queries.
    Returns readiness flag and best-effort chunk counts.
    """
    t0 = time.time()
    try:
        health = await _api_request("GET", "/v1/health")
        result = {
            "ready": health["index_ready"],
            "index_path": health["index_path"],
            "children_indexed": health["children_indexed"],
            "parents_stored": health["parents_stored"],
        }
        _record_call("index_status", {}, result, (time.time() - t0) * 1000)
        return result
    except Exception as exc:
        _record_call("index_status", {}, {}, (time.time() - t0) * 1000, error=str(exc))
        raise


# ────────────────────────────────────────────────────────────────────────────
# Tool 2: list_indexed_papers
# ────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def list_indexed_papers() -> Dict[str, Any]:
    """
    List every distinct source document currently indexed
    (deduplicated by 'source' metadata field).
    """
    t0 = time.time()
    try:
        result = await _api_request("GET", "/v1/papers")
        _record_call("list_indexed_papers", {}, {"distinct_sources": result.get("distinct_sources", 0)}, (time.time() - t0) * 1000)
        return result
    except Exception as exc:
        _record_call("list_indexed_papers", {}, {}, (time.time() - t0) * 1000, error=str(exc))
        raise


# ────────────────────────────────────────────────────────────────────────────
# Tool 3: search_papers
# ────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def search_papers(query: str, top_k: int = 6) -> Dict[str, Any]:
    """
    Call the shared hybrid API and return top-K parent chunks with scores.

    Args:
        query: user question or search string
        top_k: number of parent chunks to return (default 6)
    """
    t0 = time.time()
    args = {"query": query, "top_k": top_k}
    try:
        result = await _api_request(
            "POST",
            "/v1/search",
            json_body={"query": query, "top_k": top_k},
        )
        result["chunks"] = [
            {**chunk, "text": (chunk.get("text") or "")[:1500]}
            for chunk in result.get("chunks", [])
        ]
        _record_call(
            "search_papers",
            args,
            {"chunks": len(result["chunks"])},
            (time.time() - t0) * 1000,
        )
        return result
    except Exception as exc:
        _record_call("search_papers", args, {}, (time.time() - t0) * 1000, error=str(exc))
        raise


# ────────────────────────────────────────────────────────────────────────────
# Tool 4: ask_paper
# ────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def ask_paper(question: str, user_id: str = "mcp-client") -> Dict[str, Any]:
    """
    Full agentic answer: retrieval → LLM-as-judge relevance → generation →
    groundedness check → bounded retry. Returns the answer, its sources,
    and the internal trace_id used by the observability layer.

    Args:
        question: natural-language question about the indexed papers
        user_id: caller identifier for the memory / feedback subsystem
    """
    t0 = time.time()
    args = {"question": question, "user_id": user_id}
    try:
        result = await _api_request(
            "POST",
            "/v1/query",
            json_body={"question": question, "user_id": user_id},
        )
        summary = {"answered": result.get("answered", False), "trace_id": result.get("trace_id")}
        _record_call("ask_paper", args, summary, (time.time() - t0) * 1000)
        return result
    except Exception as exc:
        _record_call("ask_paper", args, {}, (time.time() - t0) * 1000, error=str(exc))
        raise


# ────────────────────────────────────────────────────────────────────────────
# Tool 5: build_index_from_paths
# ────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def build_index_from_paths(
    pdf_paths: List[str],
    replace: bool = True,
    with_vision: bool = False,
) -> Dict[str, Any]:
    """
    Ingest one or more local PDF files, chunk with parent-child + semantic
    splitting, build the FAISS + BM25 indexes, and persist to disk. The files
    are uploaded to the shared RAG API, so they need only be readable here.

    Args:
        pdf_paths: list of absolute paths to PDF files on the local filesystem
    """
    t0 = time.time()
    args = {"pdf_paths": pdf_paths, "replace": replace, "with_vision": with_vision}
    try:
        if not pdf_paths:
            raise ValueError("At least one PDF path must be provided.")

        paths = [Path(path).expanduser() for path in pdf_paths]
        missing = [str(path) for path in paths if not path.is_file() or path.suffix.lower() != ".pdf"]
        if missing:
            raise FileNotFoundError(f"PDF paths not found: {missing[:3]}")
        with ExitStack() as stack:
            multipart_files = [
                ("files", (path.name, stack.enter_context(path.open("rb")), "application/pdf"))
                for path in paths
            ]
            result = await _api_request(
                "POST",
                "/v1/upload",
                data={"replace": str(replace).lower(), "with_vision": str(with_vision).lower()},
                files=multipart_files,
            )
        _record_call("build_index_from_paths", args, result, (time.time() - t0) * 1000)
        return result
    except Exception as exc:
        _record_call("build_index_from_paths", args, {}, (time.time() - t0) * 1000, error=str(exc))
        raise


# ────────────────────────────────────────────────────────────────────────────
# Tool 6: deep_read_paper
# ────────────────────────────────────────────────────────────────────────────
@mcp.tool()
async def deep_read_paper(paper_url_or_path: str) -> Dict[str, Any]:
    """
    Run the full academic deep-read pipeline and return the report and artifacts.

    Args:
        paper_url_or_path: paper URL or local PDF path
    """
    t0 = time.time()
    args = {"paper_url_or_path": paper_url_or_path}
    try:
        if not paper_url_or_path.strip():
            raise ValueError("paper_url_or_path must not be empty")

        from orchestration.deep_read.orchestrator import run_deep_read

        def log_progress(fraction: float, message: str) -> None:
            log.info("deep-read %.0f%% — %s", fraction * 100, message)

        result = await run_deep_read(paper_url_or_path.strip(), progress=log_progress)
        response = {
            "title": result.title,
            "report_markdown": result.report_markdown,
            "report_path": result.report_path,
            "ppt_path": result.ppt_path,
            "artifacts_dir": result.artifacts_dir,
            "figure_count": result.figure_count,
            "table_count": result.table_count,
            "page_count": result.page_count,
            "vision_page_count": result.vision_page_count,
            "latency_sec": result.latency_sec,
        }
        _record_call(
            "deep_read_paper",
            args,
            {
                "title": result.title,
                "report_chars": len(result.report_markdown),
                "figure_count": result.figure_count,
                "table_count": result.table_count,
                "page_count": result.page_count,
                "vision_page_count": result.vision_page_count,
            },
            (time.time() - t0) * 1000,
        )
        return response
    except Exception as exc:
        _record_call("deep_read_paper", args, {}, (time.time() - t0) * 1000, error=str(exc))
        raise


if __name__ == "__main__":
    # stdio transport — the standard for MCP clients (Claude Code, Claude Desktop, etc.)
    mcp.run(transport="stdio")
