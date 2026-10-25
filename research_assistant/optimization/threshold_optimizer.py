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
    Feedback precision: retrieved traces that were positively rated
        = TP / (TP + FP) where
          TP = positive rating AND has_results,
          FP = negative rating AND has_results.
    Feedback recall: positively rated traces that still retrieve at threshold
        = TP / (TP + FN) where
          FN = positive rating AND NOT has_results.
    """
    tp = fp = fn = 0

    for trace in traces:
        trace_id = trace.get("trace_id", "")
        has_results, _top_k = _simulate_at_threshold(trace, threshold, k=k)
        rating = feedback_index.get(trace_id)
        if rating not in ("positive", "negative"):
            continue
        is_positive = rating == "positive"
        if has_results and is_positive:
            tp += 1
        elif has_results and not is_positive:
            fp += 1
        elif not has_results and is_positive:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "feedback_precision": round(precision, 4),
        "feedback_recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "rated_evaluated": tp + fp + fn,
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
    return max(results, key=lambda r: (r["f1"], r["feedback_precision"], r["threshold"]))


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
    rated_traces = [
        trace for trace in traces
        if feedback_index.get(trace.get("trace_id")) in ("positive", "negative")
    ]
    if not rated_traces:
        print("[threshold_optimizer] No rated traces in window — skipping.")
        return {"status": "skipped", "reason": "no_rated_feedback"}

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
            f"(Δ={delta:.2f}, F1={best['f1']:.3f}, feedback precision={best['feedback_precision']:.3f})"
        )
    else:
        print(
            f"[threshold_optimizer] Updated threshold {old_threshold:.2f} → {new_threshold:.2f} "
            f"(F1={best['f1']:.3f})"
        )

    return record


if __name__ == "__main__":
    optimize_threshold()
