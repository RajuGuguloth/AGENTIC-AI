"""
Deep-Read orchestrator: link/upload → extract → section analysis → report + PPT.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from etl.ingest import ingest_uploaded_file, resolve_to_pdf
from etl.pdf_rich_loader import RichPDFDocument, extract_rich_pdf
from llm.gateway import LLMGateway, get_llm_gateway
from orchestration.deep_read.merger import merge_report
from orchestration.deep_read.ppt_builder import build_ppt
from orchestration.deep_read.section_agents import analyze_all_sections

DEEP_READ_ROOT = Path("./artifacts/deep_read")


@dataclass
class DeepReadResult:
    job_id: str
    title: str
    report_markdown: str
    report_path: str
    ppt_path: str
    artifacts_dir: str
    figure_count: int
    table_count: int
    page_count: int
    latency_sec: float


ProgressCallback = Callable[[float, str], None]


def _noop_progress(p: float, msg: str) -> None:
    print(f"[deep_read] {p:.0%} — {msg}")


async def run_deep_read(
    source: Union[str, Path],
    *,
    gateway: Optional[LLMGateway] = None,
    progress: Optional[ProgressCallback] = None,
    is_upload: bool = False,
) -> DeepReadResult:
    """
    Run full Deep-Read pipeline on a URL, arXiv link, or PDF path.

    Args:
        source: URL, arXiv link, or local PDF path (or Gradio upload path)
        gateway: Optional LLM gateway (defaults to env config)
        progress: Optional callback(fraction, message)
        is_upload: True if source is a Gradio temp upload path
    """
    started = time.time()
    progress = progress or _noop_progress
    gateway = gateway or get_llm_gateway()
    job_id = str(uuid.uuid4())[:8]
    job_dir = DEEP_READ_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    progress(0.05, "Resolving PDF source…")
    if is_upload:
        pdf_path = ingest_uploaded_file(str(source), str(job_dir / "source"))
    else:
        pdf_path = resolve_to_pdf(source, str(job_dir / "source"))

    progress(0.15, "Extracting text, figures, and tables…")
    doc: RichPDFDocument = extract_rich_pdf(str(pdf_path), str(job_dir / "extracted"))

    progress(0.30, "Analyzing metadata & problem…")
    sections = await analyze_all_sections(doc, gateway)

    progress(0.75, "Merging report…")
    report_md = merge_report(sections, doc.title, str(pdf_path))
    report_path = job_dir / "deep_read_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    progress(0.85, "Building PowerPoint…")
    ppt_path = job_dir / "deep_read_slides.pptx"
    try:
        build_ppt(sections, doc, str(ppt_path))
    except ImportError as exc:
        ppt_path = job_dir / "deep_read_slides.txt"
        ppt_path.write_text(f"PPT skipped: {exc}. Install python-pptx.", encoding="utf-8")

    progress(1.0, "Deep-Read complete!")
    return DeepReadResult(
        job_id=job_id,
        title=doc.title,
        report_markdown=report_md,
        report_path=str(report_path),
        ppt_path=str(ppt_path),
        artifacts_dir=str(job_dir),
        figure_count=len(doc.figures),
        table_count=len(doc.tables),
        page_count=doc.total_pages,
        latency_sec=time.time() - started,
    )
