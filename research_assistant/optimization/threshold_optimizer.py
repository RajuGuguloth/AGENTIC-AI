"""
Dynamic MIN_RETRIEVAL_SCORE optimization via grid search on production logs.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import Config
from optimization.common import (
    OPTIMIZATION_HISTORY_PATH,
    append_jsonl,
    load_jsonl,
    load_runtime_config,
    save_runtime_config,
    send_alert,
    update_env_min_retrieval_score,
)

TRACES_PATH = Path("./logs/rag_traces.jsonl")
FEEDBACK_PATH = Path("./logs/feedback.jsonl")
THRESHOLD_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
WEEK_SECONDS = 7 * 24 * 3600
SIGNIFICANT_CHANGE = 0.1
BATCH_SIZE = 32


def _chunk_scores(trace: Dict[str, Any]) -> List[float]:
    scores: List[float] = []
    for item in trace.get("chunks_scores") or []:
        if isinstance(item, dict) and "score" in item:
            scores.append(float(item["score"]))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            scores.append(float(item[1]))
    return sorted(scores, reverse=True)


def _simulate_at_threshold(trace: Dict[str, Any], threshold: float, k: int = 5) -> Tuple[bool, int]:
    """Return (has_results, top_k_count) after applying threshold."""
    scores = _chunk_scores(trace)
    if not scores:
        verification = trace.get("verification") or {}
        if verification.get("retrieval") == "empty":
            return False, 0
        # No scores logged — assume trace would pass if relevance was found
        rel_passed = verification.get("relevance_passed", 0)
        return rel_passed > 0, min(int(rel_passed), k)

    kept = [s for s in scores if s >= threshold][:k]
    return len(kept) > 0, len(kept)


def _build_feedback_index(feedback: List[Dict[str, Any]]) -> Dict[str, str]:
    return {f["trace_id"]: f.get("rating", "") for f in feedback if f.get("trace_id")}


def _compute_metrics_for_threshold(
    traces: List[Dict[str, Any]],
    feedback_index: Dict[str, str],
    threshold: float,
    k: int = 5,
) -> Dict[str, float]:
    """
    Precision@5: positive-feedback traces that still retrieve >=1 chunk at threshold.
    Recall@5: traces with relevance_passed > 0 that still retrieve at threshold.
    """
    tp_precision = fp_precision = 0
    tp_recall = fn_recall = 0

    for trace in traces:
        trace_id = trace.get("trace_id", "")
        has_results, top_k = _simulate_at_threshold(trace, threshold, k=k)
        verification = trace.get("verification") or {}
        relevance_found = int(verification.get("relevance_passed", 0) or 0) > 0
        rating = feedback_index.get(trace_id)

        if rating == "positive":
            if has_results and top_k >= 1:
                tp_precision += 1
            else:
                fp_precision += 1

        if relevance_found:
            if has_results:
                tp_recall += 1
            else:
                fn_recall += 1

    precision = tp_precision / (tp_precision + fp_precision) if (tp_precision + fp_precision) else 0.0
    recall = tp_recall / (tp_recall + fn_recall) if (tp_recall + fn_recall) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "precision_at_5": round(precision, 4),
        "recall_at_5": round(recall, 4),
        "f1": round(f1, 4),
        "positive_evaluated": tp_precision + fp_precision,
        "relevance_evaluated": tp_recall + fn_recall,
    }


def run_grid_search_batched(
    traces: List[Dict[str, Any]],
    feedback_index: Dict[str, str],
    grid: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """Evaluate thresholds in batches (Gemini-friendly batch scoring pattern)."""
    grid = grid or THRESHOLD_GRID
    results: List[Dict[str, float]] = []
    for i in range(0, len(grid), BATCH_SIZE):
        batch = grid[i : i + BATCH_SIZE]
        for threshold in batch:
            results.append(_compute_metrics_for_threshold(traces, feedback_index, threshold))
    return results


def select_best_threshold(results: List[Dict[str, float]]) -> Dict[str, float]:
    return max(results, key=lambda r: (r["f1"], r["recall_at_5"], -r["threshold"]))


def optimize_threshold(
    weekly: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Weekly grid search over MIN_RETRIEVAL_SCORE and apply best F1 threshold.
    """
    since_ts = time.time() - WEEK_SECONDS if weekly else None
    traces = load_jsonl(TRACES_PATH, since_ts=since_ts)
    feedback = load_jsonl(FEEDBACK_PATH, since_ts=since_ts)

    if not traces:
        print("[threshold_optimizer] No traces in window — skipping.")
        return {"status": "skipped", "reason": "no_traces"}

    feedback_index = _build_feedback_index(feedback)
    grid_results = run_grid_search_batched(traces, feedback_index)
    best = select_best_threshold(grid_results)

    runtime = load_runtime_config()
    old_threshold = float(runtime.get("min_retrieval_score", Config.MIN_RETRIEVAL_SCORE))
    new_threshold = float(best["threshold"])

    record = {
        "ts": time.time(),
        "event": "threshold_optimization",
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "best_metrics": best,
        "grid_results": grid_results,
        "traces_evaluated": len(traces),
        "feedback_count": len(feedback),
        "applied": not dry_run,
    }
    append_jsonl(OPTIMIZATION_HISTORY_PATH, record)

    if dry_run:
        print(f"[threshold_optimizer] Dry run — best threshold {new_threshold} (F1={best['f1']})")
        return record

    save_runtime_config({"min_retrieval_score": new_threshold})
    Config.MIN_RETRIEVAL_SCORE = new_threshold
    update_env_min_retrieval_score(new_threshold)

    delta = abs(new_threshold - old_threshold)
    if delta >= SIGNIFICANT_CHANGE:
        send_alert(
            f"MIN_RETRIEVAL_SCORE changed significantly: {old_threshold:.2f} → {new_threshold:.2f} "
            f"(Δ={delta:.2f}, F1={best['f1']:.3f}, precision@5={best['precision_at_5']:.3f})"
        )
    else:
        print(
            f"[threshold_optimizer] Updated threshold {old_threshold:.2f} → {new_threshold:.2f} "
            f"(F1={best['f1']:.3f})"
        )

    return record


if __name__ == "__main__":
    optimize_threshold()
