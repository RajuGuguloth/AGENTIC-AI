"""
Query normalization for technical / academic retrieval.
"""

import re
from typing import List

# Common academic hyphenation variants (user query → expanded terms)
_TERM_ALIASES = {
    "self attention": "self-attention",
    "multi head": "multi-head",
    "cross attention": "cross-attention",
    "feed forward": "feed-forward",
    "layer norm": "layer normalization",
    "position encoding": "positional encoding",
}

_TECHNICAL_KEYWORDS = {
    "attention", "transformer", "encoder", "decoder", "embedding",
    "softmax", "gradient", "loss", "bleu", "token", "layer", "head",
    "self-attention", "multi-head", "positional",
}


def normalize_technical_query(query: str) -> str:
    """
    Expand informal queries to match paper terminology.
    E.g. 'what is self attention' → includes 'self-attention'.
    """
    q = (query or "").strip()
    if not q:
        return q

    lower = q.lower()
    extras: List[str] = []

    for informal, formal in _TERM_ALIASES.items():
        if informal in lower and formal not in lower:
            extras.append(formal)

    # Hyphen ↔ space variants for compound terms
    for token in re.findall(r"[a-z]{3,}(?:[- ][a-z]{3,})+", lower):
        spaced = token.replace("-", " ")
        hyphenated = token.replace(" ", "-")
        if spaced in lower and hyphenated not in lower:
            extras.append(hyphenated)
        if hyphenated in lower and spaced not in lower:
            extras.append(spaced)

    if extras:
        return f"{q} {' '.join(dict.fromkeys(extras))}"
    return q


def is_technical_query(query: str) -> bool:
    """Heuristic: short definitional or ML/NLP terminology queries."""
    lower = (query or "").lower()
    if any(kw in lower for kw in _TECHNICAL_KEYWORDS):
        return True
    if re.match(r"^(what is|what are|define|explain)\s+\w", lower):
        return True
    return False
