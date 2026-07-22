"""
Ingest research papers from URL, arXiv link, or local file path.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests

ARXIV_PDF_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)([\d.]+(?:v\d+)?)",
    re.I,
)


def _download_url(url: str, dest: Path) -> Path:
    resp = requests.get(url, timeout=120, headers={"User-Agent": "DeepRead/1.0"})
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def resolve_to_pdf(source: Union[str, Path], work_dir: str) -> Path:
    """
    Resolve link, arXiv ID/URL, or file path to a local PDF path.

    Args:
        source: URL, arXiv link, or filesystem path
        work_dir: Directory to store downloaded PDFs

    Returns:
        Path to PDF file
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    source_str = str(source).strip()

    # Local file
    local = Path(source_str)
    if local.exists() and local.suffix.lower() == ".pdf":
        dest = work / local.name
        if local.resolve() != dest.resolve():
            shutil.copy2(local, dest)
        return dest

    # arXiv URL or ID
    arxiv_match = ARXIV_PDF_RE.search(source_str)
    if arxiv_match or re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", source_str):
        arxiv_id = arxiv_match.group(1) if arxiv_match else source_str
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        dest = work / f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
        return _download_url(pdf_url, dest)

    # Direct PDF URL
    parsed = urlparse(source_str)
    if parsed.scheme in ("http", "https"):
        name = Path(parsed.path).name or "paper.pdf"
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        dest = work / name
        return _download_url(source_str, dest)

    raise ValueError(
        f"Could not resolve source to PDF: {source_str}. "
        "Use a PDF file path, PDF URL, or arXiv link."
    )


def ingest_uploaded_file(upload_path: str, work_dir: str) -> Path:
    """Copy Gradio uploaded file into work directory."""
    return resolve_to_pdf(upload_path, work_dir)
