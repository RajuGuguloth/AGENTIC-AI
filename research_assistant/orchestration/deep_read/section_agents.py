"""
Section-wise analysis agents for Deep-Read pipeline.
Each section writes beginner-friendly, evidence-linked content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from etl.pdf_rich_loader import RichPDFDocument
from llm.gateway import LLMGateway

READER_SYSTEM = """You are an expert research mentor explaining papers to first-time readers.
Write clearly, define jargon inline, use bullet points where helpful.
Every claim must reference page numbers like (p.3) or figure IDs when available.
If information is missing from the context, say "Not found in the provided text."
Do NOT invent results or citations."""


@dataclass
class SectionResult:
    section_id: str
    title: str
    content: str


SECTION_SPECS = [
    {
        "id": "metadata",
        "title": "1. Paper Metadata",
        "keywords": ["abstract", "introduction", "author"],
        "prompt": """From the context, extract and explain:
- Title, authors (if visible), venue/year hints
- One-paragraph plain-language abstract summary
- 3-5 keywords / topics
Write for someone who has never read an ML paper before.""",
    },
    {
        "id": "problem",
        "title": "2. Problem & Motivation",
        "keywords": ["introduction", "motivation", "problem", "related work", "background"],
        "prompt": """Explain:
- What problem does this paper solve? (p. references)
- Why does it matter? Real-world analogy if helpful
- What limitations of prior work motivate this paper?
Use simple language; define acronyms on first use.""",
    },
    {
        "id": "method",
        "title": "3. Method & Architecture",
        "keywords": ["method", "model", "architecture", "approach", "framework", "attention"],
        "prompt": """Explain the core method step-by-step:
- High-level pipeline (input → output)
- Key components and how they connect
- Reference any architecture figures mentioned in context
Explain as if teaching an undergraduate — intuitive first, then technical detail.""",
    },
    {
        "id": "figures_tables",
        "title": "4. Figures & Tables",
        "keywords": ["figure", "table", "fig.", "experiment"],
        "prompt": """Summarize each figure and table listed below.
For each: what it shows, why it matters, one sentence takeaway.
If a figure path is listed, describe what the reader should notice.""",
    },
    {
        "id": "results",
        "title": "5. Results & Evaluation",
        "keywords": ["result", "experiment", "evaluation", "benchmark", "performance", "bleu"],
        "prompt": """Explain:
- Main experiments and datasets used
- Key metrics and how the proposed method compares
- Strongest claims backed by numbers (with page refs)
Avoid hype; note if comparisons are limited.""",
    },
    {
        "id": "limitations",
        "title": "6. Limitations & Related Work",
        "keywords": ["limitation", "conclusion", "future", "discussion", "related"],
        "prompt": """Cover:
- Stated limitations or gaps (or infer cautiously from discussion)
- How this work relates to prior approaches
- Open questions and future directions
End with 3 bullet "What a first-time reader should remember".""",
    },
]


async def analyze_section(
    spec: dict,
    doc: RichPDFDocument,
    gateway: LLMGateway,
) -> SectionResult:
    if spec["id"] == "figures_tables":
        context = (
            f"Figure list:\n{doc.figures_summary()}\n\n"
            f"Tables:\n{doc.tables_summary()}\n\n"
            f"Relevant text:\n{doc.text_for_section(['figure', 'table', 'fig'], max_chars=8000)}"
        )
    else:
        context = doc.text_for_section(spec["keywords"], max_chars=14000)

    user = f"""Paper title: {doc.title}
Total pages: {doc.total_pages}

CONTEXT:
{context}

TASK:
{spec['prompt']}"""

    content = await gateway.complete(READER_SYSTEM, user, temperature=0.25, max_tokens=2500)
    return SectionResult(section_id=spec["id"], title=spec["title"], content=content.strip())


async def analyze_all_sections(
    doc: RichPDFDocument,
    gateway: LLMGateway,
) -> List[SectionResult]:
    import asyncio

    tasks = [analyze_section(spec, doc, gateway) for spec in SECTION_SPECS]
    return list(await asyncio.gather(*tasks))
