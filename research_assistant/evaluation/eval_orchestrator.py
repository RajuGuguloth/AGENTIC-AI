#!/usr/bin/env python3
"""
Multi-agent RAG evaluation orchestrator.

Agents (run in parallel, then sequential decision + execute):
  1. Validator  — index, config, API key probe, unit tests
  2. Monitor    — daily_eval metrics, trace/feedback counts
  3. Decision   — pick run mode from validator + monitor signals
  4. Executor   — run ragas_batch with chosen parameters

Usage:
  python evaluation/eval_orchestrator.py
  python evaluation/eval_orchestrator.py --skip-tests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INDEX_PATH = ROOT / "index_store"
RESULTS_FULL = Path(__file__).resolve().parent / "ragas_results_full.jsonl"
RESULTS_REPORT = Path(__file__).resolve().parent / "orchestrator_report.json"


async def agent_validator(skip_tests: bool = False) -> Dict[str, Any]:
    """Validate environment, index, and Gemini API availability."""
    from config import Config

    out: Dict[str, Any] = {"agent": "validator", "ok": True, "checks": []}

    def check(name: str, passed: bool, detail: str = ""):
        out["checks"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            out["ok"] = False

    check("llm_backend_gemini", Config.LLM_BACKEND == "gemini", Config.LLM_BACKEND)
    check("gemini_key_present", bool(Config.GEMINI_API_KEY), "set" if Config.GEMINI_API_KEY else "missing")

    index_ok = (INDEX_PATH / "parent_store.pkl").exists()
    vectors = 0
    if index_ok:
        try:
            from retrieval.dense_retriever import DenseRetriever
            d = DenseRetriever()
            if d.load_index(str(INDEX_PATH)):
                vectors = d.index.ntotal if d.index else 0
        except Exception as exc:
            check("index_load", False, str(exc)[:200])
    check("index_exists", index_ok, f"{vectors} vectors")

    api_status = "unknown"
    api_detail = ""
    if Config.GEMINI_API_KEY:
        try:
            from verification.gemini_verifier import GeminiVerifier

            v = GeminiVerifier.from_config()
            text = await v._generate_with_retry("Reply with only YES")
            api_status = "ok" if (text or "").strip() else "empty_response"
            api_detail = (text or "")[:30]
        except Exception as exc:
            err = str(exc)
            if "429" in err and "perday" in err.replace("_", "").replace("-", "").lower():
                api_status = "daily_quota_exhausted"
            elif "429" in err:
                api_status = "rate_limited"
            elif "401" in err or "403" in err or "API key" in err.lower():
                api_status = "invalid_key"
            elif "limit: 0" in err:
                api_status = "no_free_tier"
            else:
                api_status = "error"
            api_detail = err[:200]
    out["api_status"] = api_status
    out["api_detail"] = api_detail
    check("gemini_api_call", api_status == "ok", api_status)

    if not skip_tests:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "test_level5.py")],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=120,
        )
        passed = proc.returncode == 0
        tail = (proc.stdout or proc.stderr or "")[-200:]
        check("unit_tests", passed, tail.replace("\n", " ")[:200])

    return out


async def agent_monitor() -> Dict[str, Any]:
    """Collect production monitoring metrics from logs."""
    from evaluation.daily_eval import compute_metrics, _load_jsonl

    traces_path = ROOT / "logs" / "rag_traces.jsonl"
    feedback_path = ROOT / "logs" / "feedback.jsonl"

    traces = _load_jsonl(traces_path)
    feedback = _load_jsonl(feedback_path)
    metrics = compute_metrics(traces, feedback)

    last_full: Optional[Dict[str, Any]] = None
    if RESULTS_FULL.exists():
        for line in RESULTS_FULL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("type") == "summary":
                    last_full = row

    return {
        "agent": "monitor",
        "trace_count": metrics.get("trace_count", 0),
        "verification_pass_rate": metrics.get("verification_pass_rate"),
        "context_relevancy_rate": metrics.get("context_relevancy_rate"),
        "empty_retrieval_rate": metrics.get("empty_retrieval_rate"),
        "avg_latency_ms": metrics.get("avg_latency_ms"),
        "last_full_eval_mode": (last_full or {}).get("mode"),
        "last_full_eval_f1": ((last_full or {}).get("retrieval") or {}).get("f1"),
        "last_context_relevancy": ((last_full or {}).get("ragas_proxies") or {}).get(
            "context_relevancy_rate"
        ),
    }


def agent_decision(validator: Dict[str, Any], monitor: Dict[str, Any]) -> Dict[str, Any]:
    """Choose evaluation mode based on validator and monitor signals."""
    api = validator.get("api_status", "unknown")
    index_ok = any(c["name"] == "index_exists" and c["passed"] for c in validator.get("checks", []))

    if not index_ok:
        return {
            "agent": "decision",
            "action": "abort",
            "reason": "Index missing — build index from Data Sources tab first.",
            "mode": None,
        }

    if api == "ok":
        return {
            "agent": "decision",
            "action": "run_ragas",
            "mode": "full_llm",
            "reason": "Gemini API healthy — run LLM-as-judge on 6 in-domain queries (free-tier safe).",
            "args": {
                "full": True,
                "proxy_judge": False,
                "in_domain_only": True,
                "limit": 6,
                "judge_k": 1,
                "api_delay": 15.0,
            },
        }

    if api in ("daily_quota_exhausted", "rate_limited", "no_free_tier"):
        return {
            "agent": "decision",
            "action": "run_ragas",
            "mode": "full_proxy",
            "reason": f"Gemini API blocked ({api}) — run 30-query proxy-judge eval (reranker relevancy, no API).",
            "args": {
                "full": True,
                "proxy_judge": True,
                "in_domain_only": False,
                "limit": None,
                "judge_k": 2,
                "api_delay": 0.0,
            },
        }

    if api == "invalid_key":
        return {
            "agent": "decision",
            "action": "abort",
            "reason": "Invalid Gemini API key — create AIza key at https://aistudio.google.com/apikey",
            "mode": None,
        }

    return {
        "agent": "decision",
        "action": "run_ragas",
        "mode": "full_proxy",
        "reason": f"Gemini uncertain ({api}) — defaulting to proxy-judge full eval.",
        "args": {
            "full": True,
            "proxy_judge": True,
            "in_domain_only": False,
            "limit": None,
            "judge_k": 2,
            "api_delay": 0.0,
        },
    }


async def agent_executor(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Execute ragas_batch based on decision."""
    if decision.get("action") != "run_ragas":
        return {"agent": "executor", "skipped": True, "reason": decision.get("reason")}

    from evaluation.ragas_batch import load_golden, run_batch, write_results, print_report

    golden_path = Path(__file__).resolve().parent / "golden_set.json"
    golden = load_golden(golden_path)
    args = decision["args"]

    rows, summary = await run_batch(
        golden,
        INDEX_PATH,
        full=args["full"],
        limit=args.get("limit"),
        judge_k=args.get("judge_k", 2),
        api_delay=args.get("api_delay", 15.0),
        in_domain_only=args.get("in_domain_only", False),
        proxy_judge=args.get("proxy_judge", False),
    )
    write_results(rows, summary, RESULTS_FULL)
    print_report(summary)

    return {
        "agent": "executor",
        "skipped": False,
        "mode": decision.get("mode"),
        "summary": summary,
        "output": str(RESULTS_FULL),
    }


async def run_orchestrator(skip_tests: bool = False) -> Dict[str, Any]:
    started = time.time()
    print("=== Eval Orchestrator: Validator + Monitor (parallel) ===")
    validator, monitor = await asyncio.gather(
        agent_validator(skip_tests=skip_tests),
        agent_monitor(),
    )

    print("\n--- Validator ---")
    for c in validator.get("checks", []):
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}: {c.get('detail', '')}")
    print(f"  API status: {validator.get('api_status')}")

    print("\n--- Monitor ---")
    for k, v in monitor.items():
        if k != "agent":
            print(f"  {k}: {v}")

    decision = agent_decision(validator, monitor)
    print("\n--- Decision ---")
    print(f"  action: {decision.get('action')}")
    print(f"  mode:   {decision.get('mode')}")
    print(f"  reason: {decision.get('reason')}")

    print("\n=== Executor ===")
    executor = await agent_executor(decision)

    report = {
        "elapsed_sec": round(time.time() - started, 2),
        "validator": validator,
        "monitor": monitor,
        "decision": decision,
        "executor": executor,
    }
    RESULTS_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nOrchestrator report: {RESULTS_REPORT}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent RAG eval orchestrator")
    parser.add_argument("--skip-tests", action="store_true", help="Skip unit test check")
    args = parser.parse_args()
    asyncio.run(run_orchestrator(skip_tests=args.skip_tests))


if __name__ == "__main__":
    main()
