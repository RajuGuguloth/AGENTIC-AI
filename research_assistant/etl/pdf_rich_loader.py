"""
Rich PDF extraction: text per page, embedded figures, optional tables.
Uses PyMuPDF (fitz) for images; pdfplumber for tables when available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


@dataclass
class ExtractedFigure:
    figure_id: str
    page: int
    path: str
    caption_hint: str = ""


@dataclass
class ExtractedTable:
    table_id: str
    page: int
    markdown: str


@dataclass
class ExtractedPage:
    page_num: int
    text: str
    screenshot_path: Optional[str] = None


@dataclass
class RichPDFDocument:
    source_path: str
    title: str
    total_pages: int
    pages: List[ExtractedPage] = field(default_factory=list)
    figures: List[ExtractedFigure] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    full_text: str = ""
    artifacts_dir: str = ""

    def text_for_section(self, keywords: List[str], max_chars: int = 12000) -> str:
        """Return pages whose text matches section keywords."""
        if not keywords:
            return self.full_text[:max_chars]
        pattern = re.compile("|".join(re.escape(k) for k in keywords), re.I)
        chunks = []
        for page in self.pages:
            if pattern.search(page.text):
                chunks.append(f"--- Page {page.page_num} ---\n{page.text}")
        combined = "\n\n".join(chunks) if chunks else self.full_text
        return combined[:max_chars]

    def figures_summary(self) -> str:
        lines = []
        for fig in self.figures:
            lines.append(
                f"- {fig.figure_id} (page {fig.page}): {fig.caption_hint or 'embedded figure'}"
            )
        return "\n".join(lines) if lines else "No figures extracted."

    def tables_summary(self) -> str:
        lines = []
        for tbl in self.tables:
            preview = tbl.markdown[:500].replace("\n", " ")
            lines.append(f"### {tbl.table_id} (page {tbl.page})\n{tbl.markdown[:2000]}")
        return "\n\n".join(lines) if lines else "No tables extracted."


def _guess_title(pages: List[ExtractedPage]) -> str:
    if not pages:
        return "Untitled"
    first = pages[0].text.strip().split("\n")
    for line in first[:5]:
        line = line.strip()
        if 10 < len(line) < 200:
            return line
    return first[0][:120] if first else "Untitled"


def _find_caption_near_page(doc: "fitz.Document", page_num: int) -> str:
    try:
        page = doc[page_num - 1]
        text = page.get_text()
        for line in text.split("\n"):
            if re.match(r"^(Figure|Fig\.|Table)\s+\d", line.strip(), re.I):
                return line.strip()[:200]
    except Exception:
        pass
    return ""


def extract_rich_pdf(pdf_path: str, artifacts_dir: str) -> RichPDFDocument:
    """
    Extract text, figures, and tables from a PDF into artifacts_dir.
    """
    if fitz is None:
        raise ImportError("PyMuPDF is required: pip install pymupdf")

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(pdf_path)

    out = Path(artifacts_dir)
    figures_dir = out / "figures"
    pages_dir = out / "pages"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(path))
    pages: List[ExtractedPage] = []
    figures: List[ExtractedFigure] = []
    fig_counter = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        text = page.get_text().strip()

        # Page screenshot (useful for vision / PPT fallback)
        screenshot_path = str(pages_dir / f"page_{page_num:03d}.png")
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            pix.save(screenshot_path)
        except Exception:
            screenshot_path = None

        pages.append(ExtractedPage(page_num=page_num, text=text, screenshot_path=screenshot_path))

        # Embedded images
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base = doc.extract_image(xref)
                ext = base.get("ext", "png")
                fig_counter += 1
                fig_id = f"fig_{fig_counter}"
                fig_path = figures_dir / f"{fig_id}_p{page_num}.{ext}"
                fig_path.write_bytes(base["image"])
                figures.append(
                    ExtractedFigure(
                        figure_id=fig_id,
                        page=page_num,
                        path=str(fig_path),
                        caption_hint=_find_caption_near_page(doc, page_num),
                    )
                )
            except Exception:
                continue

    doc.close()

    # Tables via pdfplumber
    tables: List[ExtractedTable] = []
    if pdfplumber:
        try:
            with pdfplumber.open(str(path)) as pdf:
                tbl_counter = 0
                for i, page in enumerate(pdf.pages):
                    for table in page.extract_tables() or []:
                        if not table or len(table) < 2:
                            continue
                        tbl_counter += 1
                        headers = [str(c or "") for c in table[0]]
                        rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                        for row in table[1:]:
                            cells = [str(c or "").replace("|", "\\|") for c in row]
                            rows.append("| " + " | ".join(cells) + " |")
                        tables.append(
                            ExtractedTable(
                                table_id=f"table_{tbl_counter}",
                                page=i + 1,
                                markdown="\n".join(rows),
                            )
                        )
        except Exception as exc:
            print(f"[pdf_rich_loader] Table extraction skipped: {exc}")

    full_text = "\n\n".join(f"[Page {p.page_num}]\n{p.text}" for p in pages if p.text)

    return RichPDFDocument(
        source_path=str(path),
        title=_guess_title(pages),
        total_pages=len(pages),
        pages=pages,
        figures=figures,
        tables=tables,
        full_text=full_text,
        artifacts_dir=str(out),
    )
