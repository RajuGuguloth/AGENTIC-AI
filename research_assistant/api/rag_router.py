"""Authenticated HTTP API for the shared Agentic RAG runtime."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import Config
from retrieval.runtime import (
    get_hybrid_retriever,
    get_index_store_path,
    get_runtime_lock,
    rebuild_index,
)

router = APIRouter(prefix="/v1", tags=["rag"])
log = logging.getLogger("rag_api")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(default=6, ge=1, le=20)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    user_id: str = Field(default="api-client", min_length=1, max_length=128)


class IngestRequest(BaseModel):
    pdf_paths: List[str] = Field(..., min_length=1, max_length=50)
    replace: bool = True
    with_vision: bool = False


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("RAG_API_KEY", "")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="RAG API authentication is not configured. Set RAG_API_KEY.",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@router.get("/health")
async def health() -> Dict[str, Any]:
    retriever = await run_in_threadpool(get_hybrid_retriever)
    return {
        "status": "ready" if retriever.dense.is_ready else "index_empty",
        "index_ready": bool(retriever.dense.is_ready),
        "index_path": str(get_index_store_path()),
        "children_indexed": int(retriever.dense.index.ntotal)
        if retriever.dense.index is not None
        else 0,
        "parents_stored": len(retriever.parent_store),
    }


@router.get("/papers", dependencies=[Depends(require_api_key)])
async def papers() -> Dict[str, Any]:
    retriever = await run_in_threadpool(get_hybrid_retriever)
    with get_runtime_lock():
        sources: Dict[str, int] = {}
        for parent in retriever.parent_store.values():
            source = (parent.metadata or {}).get("source") or "unknown"
            sources[source] = sources.get(source, 0) + 1
    result = [
        {"source": source, "parent_chunks": count}
        for source, count in sorted(sources.items(), key=lambda item: -item[1])
    ]
    return {"papers": result, "distinct_sources": len(result)}


@router.get("/chunks/{source_id}", dependencies=[Depends(require_api_key)])
async def chunk(source_id: str) -> Dict[str, Any]:
    retriever = await run_in_threadpool(get_hybrid_retriever)
    with get_runtime_lock():
        parent = retriever.parent_store.get(source_id)
        if parent is None:
            raise HTTPException(status_code=404, detail="Parent chunk not found.")
        return {
            "source_id": source_id,
            "text": parent.page_content or "",
            "metadata": dict(parent.metadata or {}),
        }


@router.post("/search", dependencies=[Depends(require_api_key)])
async def search(body: SearchRequest) -> Dict[str, Any]:
    retriever = await run_in_threadpool(get_hybrid_retriever)
    if not retriever.dense.is_ready:
        raise HTTPException(status_code=503, detail="Index is empty. Ingest a PDF first.")
    try:
        def search_documents():
            with get_runtime_lock():
                return retriever.search(body.query.strip(), k=body.top_k)

        docs = await run_in_threadpool(search_documents)
    except Exception as exc:
        log.exception("RAG search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    chunks = []
    for doc in docs:
        metadata = doc.metadata or {}
        chunks.append(
            {
                "text": doc.page_content or "",
                "source": metadata.get("source"),
                "source_id": metadata.get("parent_id"),
                "page": metadata.get("page"),
                "retrieval_score": metadata.get("retrieval_score"),
                "reranker_score": metadata.get("reranker_score"),
                "child_rrf_score": metadata.get("child_rrf_score"),
            }
        )
    return {"query": body.query, "chunks": chunks, "count": len(chunks)}


def _answer_question(question: str, user_id: str) -> Dict[str, Any]:
    Config.validate()
    llm = Config.get_llm()
    retriever = get_hybrid_retriever()
    result = asyncio.run(
        _answer_document_question(question, llm, retriever, user_id)
    )
    return result


async def _answer_document_question(question: str, llm, retriever, user_id: str):
    from orchestration.document_chat import answer_document_question

    return await answer_document_question(question, [], llm, retriever, user_id=user_id)


def _index_paths(
    paths: List[Path],
    *,
    replace: bool,
    with_vision: bool,
    file_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from etl.chunker import chunk_documents_parent_child
    from etl.pdf_loader import load_pdfs_from_files

    raw_docs = load_pdfs_from_files(
        [str(path) for path in paths],
        with_vision=with_vision,
    )
    if not raw_docs:
        raise ValueError("No readable PDF pages were found.")
    parent_docs, child_docs = chunk_documents_parent_child(raw_docs)
    result = rebuild_index(parent_docs, child_docs, replace=replace)
    result["files"] = file_names or [path.name for path in paths]
    result["vision_enabled"] = with_vision
    return result


@router.post("/query", dependencies=[Depends(require_api_key)])
async def query(body: QueryRequest) -> Dict[str, Any]:
    retriever = await run_in_threadpool(get_hybrid_retriever)
    if not retriever.dense.is_ready:
        raise HTTPException(status_code=503, detail="Index is empty. Ingest a PDF first.")
    try:
        result = await run_in_threadpool(
            _answer_question,
            body.question.strip(),
            body.user_id,
        )
        return {
            "answered": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "trace_id": result["trace_id"],
        }
    except Exception as exc:
        log.exception("RAG query failed")
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}") from exc


@router.post("/ingest", dependencies=[Depends(require_api_key)])
async def ingest(body: IngestRequest) -> Dict[str, Any]:
    paths = [Path(path).expanduser().resolve() for path in body.pdf_paths]
    invalid = [str(path) for path in paths if not path.is_file() or path.suffix.lower() != ".pdf"]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={"message": "Every path must be an existing PDF.", "paths": invalid[:5]},
        )

    try:
        return await run_in_threadpool(
            lambda: _index_paths(
                paths,
                replace=body.replace,
                with_vision=body.with_vision,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("PDF ingestion failed")
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {exc}") from exc


@router.post("/upload", dependencies=[Depends(require_api_key)])
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    replace: bool = Form(default=True),
    with_vision: bool = Form(default=False),
) -> Dict[str, Any]:
    if not files or len(files) > 50:
        raise HTTPException(status_code=400, detail="Upload between 1 and 50 PDF files.")

    max_file_bytes = 100 * 1024 * 1024
    names: List[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="rag-upload-") as temp_dir:
            paths = []
            total_bytes = 0
            for index, upload_file in enumerate(files):
                name = Path(upload_file.filename or "").name
                if not name.lower().endswith(".pdf"):
                    raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
                path = Path(temp_dir) / f"{index}-{name}"
                first_bytes = b""
                file_bytes = 0
                with path.open("wb") as target:
                    while chunk_bytes := await upload_file.read(1024 * 1024):
                        if not first_bytes:
                            first_bytes = chunk_bytes[:5]
                        file_bytes += len(chunk_bytes)
                        total_bytes += len(chunk_bytes)
                        if file_bytes > max_file_bytes:
                            raise HTTPException(status_code=413, detail=f"{name} exceeds 100 MB.")
                        if total_bytes > max_file_bytes:
                            raise HTTPException(status_code=413, detail="Combined upload exceeds 100 MB.")
                        target.write(chunk_bytes)
                if first_bytes != b"%PDF-":
                    raise HTTPException(status_code=400, detail=f"{name} is not a valid PDF.")
                paths.append(path)
                names.append(name)
                await upload_file.close()

            return await run_in_threadpool(
                lambda: _index_paths(
                    paths,
                    replace=replace,
                    with_vision=with_vision,
                    file_names=names,
                )
            )
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("PDF upload and ingestion failed")
        raise HTTPException(status_code=500, detail=f"PDF upload failed: {exc}") from exc
    finally:
        for upload_file in files:
            await upload_file.close()
