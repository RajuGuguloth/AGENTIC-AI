"""Page-by-page vision analysis for handwritten notes and PDF visuals."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

from llm.gateway import LLMGateway, get_vision_gateway

try:
    import fitz
except ImportError:
    fitz = None


VISION_SYSTEM_PROMPT = """You analyze academic and technical document pages.
Transcribe visible handwritten and printed notes as accurately as possible.
Describe diagrams, charts, tables, equations, arrows, labels, and their relationships.
Preserve meaningful structure and mark uncertain handwriting as [unclear].
Do not guess missing text or infer unsupported conclusions. Keep the result concise
but include details useful for searching and understanding this page."""

VISION_USER_PROMPT = """Analyze page {page_num} of this document.
Use these headings:
Handwritten and printed text:
Visuals and layout:
Important labels or values:
If a heading has nothing relevant, write "None found."
Transcribe handwriting rather than describing its penmanship."""


async def analyze_page_images(
    page_images: Iterable[Tuple[int, bytes]],
    *,
    gateway: Optional[LLMGateway] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    max_concurrency: int = 3,
) -> Dict[int, str]:
    """Analyze page image bytes concurrently while limiting provider requests."""
    images = list(page_images)
    if not images:
        return {}
    gateway = gateway or get_vision_gateway()
    semaphore = asyncio.Semaphore(max_concurrency)
    completed = 0

    async def analyze(page_num: int, image: bytes) -> Tuple[int, str]:
        nonlocal completed
        async with semaphore:
            try:
                result = await gateway.complete_with_image(
                    VISION_SYSTEM_PROMPT,
                    VISION_USER_PROMPT.format(page_num=page_num),
                    image,
                    mime_type="image/png",
                    temperature=0.1,
                    max_tokens=1200,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Vision analysis failed for PDF page {page_num}: {exc}"
                ) from exc
            if not result.strip():
                raise RuntimeError(
                    f"Vision model returned an empty result for PDF page {page_num}"
                )
        completed += 1
        if progress:
            progress(completed / len(images), f"Analyzed page {page_num} of {len(images)}")
        return page_num, result.strip()

    results = await asyncio.gather(*(analyze(page, image) for page, image in images))
    return dict(results)


async def analyze_pdf_pages(
    pdf_path: str,
    *,
    gateway: Optional[LLMGateway] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict[int, str]:
    """Render and analyze every page of a PDF; errors are surfaced to the caller."""
    if fitz is None:
        raise ImportError("PyMuPDF is required for page vision: pip install pymupdf")
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found for vision analysis: {pdf_path}")

    gateway = gateway or get_vision_gateway()
    descriptions: Dict[int, str] = {}
    with fitz.open(str(path)) as pdf:
        total_pages = len(pdf)
        for start in range(0, total_pages, 3):
            page_images: List[Tuple[int, bytes]] = []
            for page_index in range(start, min(start + 3, total_pages)):
                pixmap = pdf[page_index].get_pixmap(
                    matrix=fitz.Matrix(2.0, 2.0),
                    alpha=False,
                )
                page_images.append((page_index + 1, pixmap.tobytes("png")))

            def batch_progress(fraction: float, message: str) -> None:
                if progress:
                    overall = (start + fraction * len(page_images)) / total_pages
                    progress(overall, message)

            descriptions.update(
                await analyze_page_images(
                    page_images,
                    gateway=gateway,
                    progress=batch_progress,
                )
            )
    return descriptions


def enrich_pdf_documents_with_vision(
    documents: List[Document],
    *,
    gateway: Optional[LLMGateway] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[Document]:
    """Append page-specific vision findings to PDF Documents for RAG indexing."""
    grouped: Dict[str, List[Document]] = defaultdict(list)
    for document in documents:
        file_path = document.metadata.get("file_path")
        if not file_path:
            raise ValueError("PDF page is missing file_path metadata for vision analysis")
        grouped[str(file_path)].append(document)

    gateway = gateway or get_vision_gateway()
    enriched: List[Document] = []
    for file_index, (file_path, file_documents) in enumerate(grouped.items(), start=1):
        def file_progress(fraction: float, message: str) -> None:
            if progress:
                overall = (file_index - 1 + fraction) / len(grouped)
                progress(overall, f"{Path(file_path).name}: {message}")

        descriptions = asyncio.run(
            analyze_pdf_pages(
                file_path,
                gateway=gateway,
                progress=file_progress,
            )
        )
        for document in file_documents:
            page_num = int(document.metadata.get("page", 0))
            description = descriptions.get(page_num)
            if description is None:
                raise RuntimeError(
                    f"Vision model did not return analysis for {file_path} page {page_num}"
                )
            text = document.page_content.strip()
            content = (
                f"{text}\n\n[Page {page_num} vision analysis]\n{description}".strip()
            )
            enriched.append(
                Document(
                    page_content=content,
                    metadata={**document.metadata, "vision_analyzed": True},
                )
            )
    return enriched
