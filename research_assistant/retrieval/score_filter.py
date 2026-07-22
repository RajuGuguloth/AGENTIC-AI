"""
Score filtering for dense retrieval results.
Drops chunks below MIN_RETRIEVAL_SCORE to avoid weak-match hallucinations.
"""

from typing import List, Tuple

from config import Config

try:
    from retrieval.query_utils import is_technical_query
except ImportError:
    def is_technical_query(_q: str) -> bool:
        return False


def filter_by_score(
    results: List[Tuple[int, float]],
    min_score: float,
    query: str = "",
) -> Tuple[List[Tuple[int, float]], float]:
    """
    Filter (doc_index, score) pairs by minimum cosine similarity.

    Args:
        results: List of (index, score) from FAISS search.
        min_score: Minimum acceptable similarity score.

    Returns:
        (filtered_results, lowest_score_seen)
        lowest_score_seen is 0.0 when results is empty.
    """
    if not results:
        return [], 0.0

    lowest = min(score for _, score in results)
    filtered = [(idx, score) for idx, score in results if score >= min_score]

    if len(filtered) < len(results) and Config.LOG_FILTERED_RESULTS:
        dropped = len(results) - len(filtered)
        q_hint = f" query='{query[:40]}'" if query else ""
        tech = " [technical]" if query and is_technical_query(query) else ""
        print(
            f"[score_filter] Dropped {dropped}/{len(results)} results "
            f"below threshold {min_score:.3f} (lowest seen: {lowest:.3f}){tech}{q_hint}"
        )

    return filtered, lowest
