#!/usr/bin/env python3
"""
Daily RAG evaluation from structured logs.

Reads:
  - ./logs/rag_traces.jsonl
  - ./logs/feedback.jsonl

Computes:
  - verification_pass_rate (RAGAS faithfulness proxy)
  - context_relevancy_rate (chunk relevance judge pass rate)
  - empty_retrieval_rate
  - regeneration_rate
  - verification_coverage
  - first_pass_faithfulness_rate
  - relevance_fallback_rate
  - avg_latency (ms)
  - user_satisfaction

Alerts when verification_pass_rate < 0.7
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TRACES_PATH = Path("./logs/rag_traces.jsonl")
FEEDBACK_PATH = Path("./logs/feedback.jsonl")
VERIFICATION_ALERT_THRESHOLD = 0.7


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[daily_eval] Skipping malformed line {line_no} in {path}: {exc}", file=sys.stderr)
    return records


def _verification_block(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("verification") or {}


def _is_empty_retrieval(record: Dict[str, Any]) -> bool:
    return _verification_block(record).get("retrieval") == "empty"


def _verifier_ran(record: Dict[str, Any]) -> bool:
    """True when relevance or groundedness verification was attempted."""
    verification = _verification_block(record)
    if verification.get("retrieval") == "empty":
        return False
    if verification.get("relevance_total", 0) > 0:
        return True
    if "grounded" in verification or "grounded_after_retry" in verification:
        return True
    attempts = verification.get("grounded_attempts")
    return isinstance(attempts, list) and len(attempts) > 0


def _is_verification_pass(record: Dict[str, Any]) -> Optional[bool]:
    """
    Return True/False if verification ran, None if not applicable.
    Pass = final grounded check succeeded (including after regeneration).
    """
    verification = _verification_block(record)
    if not verification:
        return None

    if verification.get("retrieval") in ("empty",):
        return None

    if verification.get("grounded") is True:
        return True
    if verification.get("grounded_after_retry") is True:
        return True

    attempts = verification.get("grounded_attempts")
    if isinstance(attempts, list) and attempts:
        return bool(attempts[-1].get("grounded"))

    if "grounded" in verification or "grounded_after_retry" in verification:
        return False

    if verification.get("relevance_total", 0) > 0 and verification.get("relevance_passed", 0) == 0:
        return False

    return None


def compute_metrics(
    traces: List[Dict[str, Any]],
    feedback: List[Dict[str, Any]],
) -> Dict[str, Any]:
    latencies = [
        float(r["latency_ms"])
        for r in traces
        if isinstance(r.get("latency_ms"), (int, float))
    ]

    verification_outcomes: List[bool] = []
    relevancy_outcomes: List[bool] = []
    regeneration_outcomes: List[bool] = []
    first_pass_outcomes: List[bool] = []
    fallback_outcomes: List[bool] = []
    verifier_ran_count = 0

    empty_retrieval_count = 0
    for record in traces:
        verification = _verification_block(record)

        if _is_empty_retrieval(record):
            empty_retrieval_count += 1

        if _verifier_ran(record):
            verifier_ran_count += 1

        outcome = _is_verification_pass(record)
        if outcome is not None:
            verification_outcomes.append(outcome)

        rel_total = int(verification.get("relevance_total", 0) or 0)
        rel_passed = int(verification.get("relevance_passed", 0) or 0)
        if rel_total > 0:
            relevancy_outcomes.append(rel_passed > 0)
            fallback_outcomes.append(bool(verification.get("relevance_fallback")))

        regen = int(verification.get("regeneration_count", 0) or 0)
        attempts = verification.get("grounded_attempts")
        if regen > 0 or (isinstance(attempts, list) and len(attempts) > 1):
            regeneration_outcomes.append(regen > 0)

        if isinstance(attempts, list) and attempts:
            first_pass_outcomes.append(bool(attempts[0].get("grounded")))

    trace_count = len(traces)
    verification_pass_rate = (
        sum(1 for ok in verification_outcomes if ok) / len(verification_outcomes)
        if verification_outcomes
        else 0.0
    )
    context_relevancy_rate = (
        sum(1 for ok in relevancy_outcomes if ok) / len(relevancy_outcomes)
        if relevancy_outcomes
        else 0.0
    )
    empty_retrieval_rate = empty_retrieval_count / trace_count if trace_count else 0.0
    regeneration_rate = (
        sum(1 for ok in regeneration_outcomes if ok) / len(regeneration_outcomes)
        if regeneration_outcomes
        else 0.0
    )
    verification_coverage = verifier_ran_count / trace_count if trace_count else 0.0
    first_pass_faithfulness_rate = (
        sum(1 for ok in first_pass_outcomes if ok) / len(first_pass_outcomes)
        if first_pass_outcomes
        else 0.0
    )
    relevance_fallback_rate = (
        sum(1 for ok in fallback_outcomes if ok) / len(fallback_outcomes)
        if fallback_outcomes
        else 0.0
    )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    positive = sum(1 for f in feedback if f.get("rating") == "positive")
    negative = sum(1 for f in feedback if f.get("rating") == "negative")
    rated = positive + negative
    user_satisfaction = positive / rated if rated else 0.0

    return {
        "trace_count": trace_count,
        "feedback_count": len(feedback),
        "verification_evaluated": len(verification_outcomes),
        "verification_pass_rate": round(verification_pass_rate, 4),
        "context_relevancy_rate": round(context_relevancy_rate, 4),
        "empty_retrieval_rate": round(empty_retrieval_rate, 4),
        "regeneration_rate": round(regeneration_rate, 4),
        "verification_coverage": round(verification_coverage, 4),
        "first_pass_faithfulness_rate": round(first_pass_faithfulness_rate, 4),
        "relevance_fallback_rate": round(relevance_fallback_rate, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "user_satisfaction": round(user_satisfaction, 4),
        "feedback_positive": positive,
        "feedback_negative": negative,
    }


def run_daily_eval(
    traces_path: Path = TRACES_PATH,
    feedback_path: Path = FEEDBACK_PATH,
    alert_threshold: float = VERIFICATION_ALERT_THRESHOLD,
) -> Dict[str, Any]:
    traces = _load_jsonl(traces_path)
    feedback = _load_jsonl(feedback_path)
    metrics = compute_metrics(traces, feedback)

    print("=== Daily RAG Evaluation (RAGAS-aligned) ===")
    print(f"Traces loaded:              {metrics['trace_count']}")
    print(f"Feedback loaded:            {metrics['feedback_count']}")
    print(f"Verification evaluated:   {metrics['verification_evaluated']}")
    print(f"verification_coverage:      {metrics['verification_coverage']:.2%}")
    print(f"verification_pass_rate:     {metrics['verification_pass_rate']:.2%}  (faithfulness proxy)")
    print(f"first_pass_faithfulness:    {metrics['first_pass_faithfulness_rate']:.2%}")
    print(f"context_relevancy_rate:   {metrics['context_relevancy_rate']:.2%}")
    print(f"relevance_fallback_rate:    {metrics['relevance_fallback_rate']:.2%}")
    print(f"empty_retrieval_rate:       {metrics['empty_retrieval_rate']:.2%}")
    print(f"regeneration_rate:          {metrics['regeneration_rate']:.2%}")
    print(f"avg_latency:                {metrics['avg_latency_ms']:.1f} ms")
    print(f"user_satisfaction:          {metrics['user_satisfaction']:.2%} "
          f"({metrics['feedback_positive']}+ / {metrics['feedback_negative']}-)")

    if metrics["verification_evaluated"] > 0 and metrics["verification_pass_rate"] < alert_threshold:
        print(
            f"\n⚠️  ALERT: verification_pass_rate {metrics['verification_pass_rate']:.2%} "
            f"is below threshold {alert_threshold:.0%}. "
            "Review groundedness prompts, retrieval scores, or regeneration settings."
        )

    if metrics["trace_count"] > 0 and metrics["trace_count"] < 30:
        print(
            f"\nℹ️  Note: only {metrics['trace_count']} traces — metrics are directional, "
            "not statistically stable yet (recommend ≥30)."
        )

    return metrics


if __name__ == "__main__":
    run_daily_eval()
