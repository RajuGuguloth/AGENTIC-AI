"""
Self-healing pipeline — hourly anomaly detection and automatic recovery.
Includes Gemini token-bucket rate limiting for recovery API calls.
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from config import Config
from evaluation.daily_eval import _is_verification_pass, compute_metrics
from optimization.common import (
    SELF_HEALING_LOG_PATH,
    append_jsonl,
    load_jsonl,
    load_runtime_config,
    save_runtime_config,
    send_alert,
)
from optimization.prompt_optimizer import rollback_prompt

TRACES_PATH = Path("./logs/rag_traces.jsonl")
FEEDBACK_PATH = Path("./logs/feedback.jsonl")
HOUR_SECONDS = 3600
VERIFICATION_DROP_THRESHOLD = 0.20
LATENCY_SPIKE_MULTIPLIER = 2.0
EMPTY_RETRIEVAL_THRESHOLD = 0.50
THRESHOLD_RECOVERY_STEP = 0.1


class GeminiTokenBucket:
    """Token bucket for Gemini API calls during recovery actions."""

    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()

    def acquire(self, tokens: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


_token_bucket = GeminiTokenBucket(
    rate=float(getattr(Config, "GEMINI_TOKEN_BUCKET_RATE", 10.0)),
    capacity=float(getattr(Config, "GEMINI_TOKEN_BUCKET_CAPACITY", 20.0)),
)


def _traces_in_window(traces: List[Dict[str, Any]], hours: float = 1.0) -> List[Dict[str, Any]]:
    cutoff = time.time() - hours * HOUR_SECONDS
    return [t for t in traces if float(t.get("ts", 0)) >= cutoff]


def _baseline_metrics(traces: List[Dict[str, Any]], hours: float = 24.0) -> Dict[str, float]:
    """Baseline from older window excluding last hour."""
    now = time.time()
    recent_cutoff = now - HOUR_SECONDS
    baseline_cutoff = now - hours * HOUR_SECONDS
    baseline_traces = [
        t for t in traces
        if baseline_cutoff <= float(t.get("ts", 0)) < recent_cutoff
    ]
    if not baseline_traces:
        return {"verification_pass_rate": 0.8, "avg_latency_ms": 5000.0, "empty_retrieval_rate": 0.1}

    outcomes = []
    latencies = []
    empty = 0
    for t in baseline_traces:
        outcome = _is_verification_pass(t)
        if outcome is not None:
            outcomes.append(outcome)
        if isinstance(t.get("latency_ms"), (int, float)):
            latencies.append(float(t["latency_ms"]))
        verification = t.get("verification") or {}
        if verification.get("retrieval") == "empty":
            empty += 1

    return {
        "verification_pass_rate": sum(outcomes) / len(outcomes) if outcomes else 0.8,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 5000.0,
        "empty_retrieval_rate": empty / len(baseline_traces) if baseline_traces else 0.1,
    }


def _current_metrics(recent: List[Dict[str, Any]]) -> Dict[str, float]:
    feedback = load_jsonl(FEEDBACK_PATH)
    return compute_metrics(recent, feedback)


def detect_anomalies(
    recent: List[Dict[str, Any]],
    baseline: Dict[str, float],
) -> List[Dict[str, Any]]:
    if not recent:
        return []

    current = _current_metrics(recent)
    anomalies: List[Dict[str, Any]] = []

    if baseline["verification_pass_rate"] - current["verification_pass_rate"] > VERIFICATION_DROP_THRESHOLD:
        anomalies.append({
            "type": "verification_drop",
            "before": baseline["verification_pass_rate"],
            "after": current["verification_pass_rate"],
            "delta": baseline["verification_pass_rate"] - current["verification_pass_rate"],
        })

    if current["avg_latency_ms"] > baseline["avg_latency_ms"] * LATENCY_SPIKE_MULTIPLIER:
        anomalies.append({
            "type": "latency_spike",
            "before": baseline["avg_latency_ms"],
            "after": current["avg_latency_ms"],
            "multiplier": current["avg_latency_ms"] / max(baseline["avg_latency_ms"], 1),
        })

    empty_count = sum(
        1 for t in recent
        if (t.get("verification") or {}).get("retrieval") == "empty"
    )
    empty_rate = empty_count / len(recent)
    if empty_rate > EMPTY_RETRIEVAL_THRESHOLD:
        anomalies.append({
            "type": "empty_retrieval_spike",
            "before": baseline.get("empty_retrieval_rate", 0),
            "after": empty_rate,
        })

    return anomalies


def apply_recovery(anomaly: Dict[str, Any]) -> Dict[str, Any]:
    """Execute automatic recovery for a detected anomaly."""
    if not _token_bucket.acquire():
        return {"action": "rate_limited", "anomaly": anomaly["type"]}

    runtime = load_runtime_config()
    action_record: Dict[str, Any] = {
        "ts": time.time(),
        "anomaly": anomaly,
        "actions": [],
    }

    if anomaly["type"] == "empty_retrieval_spike":
        old = float(runtime.get("min_retrieval_score", Config.MIN_RETRIEVAL_SCORE))
        new = max(0.15, old - THRESHOLD_RECOVERY_STEP)
        save_runtime_config({"min_retrieval_score": new})
        Config.MIN_RETRIEVAL_SCORE = new
        action_record["actions"].append({
            "type": "lower_threshold",
            "old": old,
            "new": new,
        })

    elif anomaly["type"] == "latency_spike":
        save_runtime_config({"query_cache_enabled": True})
        action_record["actions"].append({"type": "enable_query_cache"})

    elif anomaly["type"] == "verification_drop":
        restored = rollback_prompt("generation")
        action_record["actions"].append({
            "type": "rollback_prompt",
            "task": "generation",
            "restored": restored,
        })

    append_jsonl(SELF_HEALING_LOG_PATH, action_record)
    return action_record


def run_self_healing(dry_run: bool = False) -> Dict[str, Any]:
    """
    Scan last hour of traces, detect anomalies, apply recovery.
    """
    all_traces = load_jsonl(TRACES_PATH)
    recent = _traces_in_window(all_traces, hours=1.0)
    baseline = _baseline_metrics(all_traces, hours=24.0)
    anomalies = detect_anomalies(recent, baseline)

    report = {
        "ts": time.time(),
        "recent_traces": len(recent),
        "baseline": baseline,
        "anomalies": anomalies,
        "recoveries": [],
        "dry_run": dry_run,
    }

    if not anomalies:
        print("[self_healer] No anomalies detected in last hour.")
        return report

    for anomaly in anomalies:
        msg = (
            f"Anomaly detected: {anomaly['type']} "
            f"(before={anomaly.get('before')}, after={anomaly.get('after')})"
        )
        print(f"[self_healer] {msg}")

        if dry_run:
            report["recoveries"].append({"anomaly": anomaly["type"], "action": "dry_run"})
            continue

        recovery = apply_recovery(anomaly)
        report["recoveries"].append(recovery)
        send_alert(
            f"Self-healing applied for {anomaly['type']}. "
            f"Actions: {recovery.get('actions')}. "
            f"Before/after: {anomaly.get('before')} → {anomaly.get('after')}"
        )

    return report


if __name__ == "__main__":
    run_self_healing()
