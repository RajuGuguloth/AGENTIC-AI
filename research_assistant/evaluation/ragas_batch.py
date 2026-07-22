#!/usr/bin/env python3
"""
Offline RAGAS-aligned batch evaluation on a labeled golden query set.

Scores retrieval (Hit@k, precision/recall/abstention), context relevancy
(LLM-as-judge per chunk), and faithfulness (generate + groundedness check).

Usage:
  python evaluation/ragas_batch.py                    # retrieval-only (no API)
  python evaluation/ragas_batch.py --full             # + Gemini judges + answers
  python evaluation/ragas_batch.py --limit 5          # quick smoke test
  python evaluation/ragas_batch.py --golden path.json # custom golden set

Requires ./index_store/ built from the Transformer paper (or matching corpus).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_set.json"
DEFAULT_INDEX = ROOT / "index_store"
RESULTS_PATH = Path(__file__).resolve().parent / "ragas_results.jsonl"

try:
    from retrieval.query_utils import normalize_technical_query
except ImportError:
    def normalize_technical_query(q: str) -> str:
        return q


def load_golden(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Golden set must be a JSON array: {path}")
    return data


def _retrieval_passed(docs: List, threshold: float) -> bool:
    if not docs:
        return False
    top = docs[0].metadata.get("retrieval_score")
    if top is None:
        return True
    return float(top) >= threshold


def aggregate_retrieval_metrics(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Query-level retrieval precision/recall for abstention task."""
    tp = fp = tn = fn = 0
    in_domain_hits = 0
    in_domain_total = 0
    off_topic_blocked = 0
    off_topic_total = 0

    for row in rows:
        expect = bool(row["expect_retrieval"])
        passed = bool(row["retrieval_passed"])
        if expect:
            in_domain_total += 1
            if passed:
                in_domain_hits += 1
                tp += 1
            else:
                fn += 1
        else:
            off_topic_total += 1
            if passed:
                fp += 1
            else:
                tn += 1
                off_topic_blocked += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hit_at_k_in_domain": round(in_domain_hits / in_domain_total, 4) if in_domain_total else 0.0,
        "off_topic_block_rate": round(off_topic_blocked / off_topic_total, 4) if off_topic_total else 0.0,
    }


def aggregate_judge_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """RAGAS-aligned LLM judge aggregates."""
    rel_rates: List[float] = []
    faith_pass = 0
    faith_total = 0
    first_pass = 0
    first_total = 0

    for row in rows:
        total = int(row.get("relevance_total") or 0)
        passed = int(row.get("relevance_passed") or 0)
        if total > 0:
            rel_rates.append(passed / total)

        if row.get("faithfulness_evaluated"):
            faith_total += 1
            if row.get("faithfulness_passed"):
                faith_pass += 1
            if row.get("first_pass_grounded") is not None:
                first_total += 1
                if row["first_pass_grounded"]:
                    first_pass += 1

    return {
        "context_relevancy_rate": round(sum(rel_rates) / len(rel_rates), 4) if rel_rates else None,
        "faithfulness_rate": round(faith_pass / faith_total, 4) if faith_total else None,
        "first_pass_faithfulness_rate": round(first_pass / first_total, 4) if first_total else None,
        "faithfulness_evaluated": faith_total,
        "relevance_judged_queries": len(rel_rates),
    }


def _is_daily_quota_error(message: str) -> bool:
    """True only for daily free-tier exhaustion, not per-minute RPM limits."""
    lower = (message or "").lower()
    normalized = lower.replace("_", "").replace("-", "")
    return "429" in lower and "perday" in normalized


def _rpm_retry_seconds(message: str, default: float = 45.0) -> float:
    """Parse retry_delay from Gemini 429 when available."""
    import re
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", lower := (message or "").lower())
    if match:
        return float(match.group(1)) + 2.0
    match = re.search(r"seconds:\s*(\d+)", message or "")
    if match:
        return float(match.group(1)) + 2.0
    return default


async def _grade_relevance(
    verifier,
    question: str,
    docs: List,
    *,
    judge_k: int = 2,
    api_delay: float = 0.0,
) -> Tuple[int, int]:
    passed = 0
    to_judge = docs[:judge_k]
    for doc in to_judge:
        cosine = doc.metadata.get("retrieval_score")
        try:
            ok = await verifier.is_relevant(
                question,
                doc.page_content,
                score=cosine,
                score_is_cosine=cosine is not None,
            )
        except Exception as exc:
            print(f"[ragas_batch] relevance judge error: {exc}")
            from verification.gemini_verifier import GeminiQuotaExhaustedError
            if isinstance(exc, GeminiQuotaExhaustedError) or _is_daily_quota_error(str(exc)):
                raise
            ok = False
        if ok:
            passed += 1
        if api_delay > 0:
            await asyncio.sleep(api_delay)
    return passed, len(to_judge)


async def _generate_and_check_faithfulness(
    llm,
    verifier,
    question: str,
    docs: List,
) -> Dict[str, Any]:
    from langchain_core.prompts import ChatPromptTemplate

    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        header = f"[{source}" + (f", page {page}" if page else "") + "]"
        context_parts.append(f"{header}\n{doc.page_content.strip()}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using ONLY the document context. If unknown, say you don't know.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ]
    )
    chain = prompt | llm
    try:
        response = await chain.ainvoke({"context": context, "question": question})
    except Exception as exc:
        print(f"[ragas_batch] generation error: {exc}")
        return {
            "answer_preview": "",
            "first_pass_grounded": False,
            "faithfulness_passed": False,
            "faithfulness_evaluated": False,
            "faithfulness_error": str(exc)[:200],
        }
    answer = response.content if hasattr(response, "content") else str(response)

    try:
        first_grounded = await verifier.is_grounded(answer, context)
    except Exception as exc:
        print(f"[ragas_batch] groundedness judge error: {exc}")
        first_grounded = False
    return {
        "answer_preview": (answer or "")[:300],
        "first_pass_grounded": first_grounded,
        "faithfulness_passed": first_grounded,
        "faithfulness_evaluated": True,
    }


async def evaluate_one(
    item: Dict[str, Any],
    retriever,
    threshold: float,
    *,
    full: bool,
    verifier=None,
    llm=None,
    judge_k: int = 2,
    api_delay: float = 0.0,
    proxy_judge: bool = False,
    reranker_threshold: float = 0.5,
) -> Dict[str, Any]:
    query = item["query"]
    search_query = normalize_technical_query(query)
    docs = await asyncio.get_event_loop().run_in_executor(
        None, retriever.search, search_query
    )

    top_score = None
    if docs:
        raw = docs[0].metadata.get("retrieval_score")
        top_score = float(raw) if raw is not None else None

    retrieval_passed = _retrieval_passed(docs, threshold)
    row: Dict[str, Any] = {
        "id": item.get("id"),
        "query": query,
        "category": item.get("category"),
        "expect_retrieval": bool(item.get("expect_retrieval")),
        "retrieval_passed": retrieval_passed,
        "chunks_returned": len(docs),
        "top_retrieval_score": top_score,
        "top_reranker_score": (
            float(docs[0].metadata["reranker_score"])
            if docs and docs[0].metadata.get("reranker_score") is not None
            else None
        ),
    }

    if full and proxy_judge and docs and item.get("expect_retrieval"):
        to_judge = docs[:judge_k]
        rel_passed = sum(
            1 for d in to_judge
            if (d.metadata.get("reranker_score") or 0) >= reranker_threshold
            or (d.metadata.get("retrieval_score") or 0) >= threshold
        )
        row["relevance_passed"] = rel_passed
        row["relevance_total"] = len(to_judge)
        row["proxy_judge"] = True
        row["faithfulness_evaluated"] = False
        row["faithfulness_note"] = "Skipped — use --full without --proxy-judge when API quota available"

    elif full and verifier and docs and item.get("expect_retrieval"):
        try:
            rel_passed, rel_total = await _grade_relevance(
                verifier, query, docs, judge_k=judge_k, api_delay=api_delay
            )
            row["relevance_passed"] = rel_passed
            row["relevance_total"] = rel_total

            if rel_passed > 0 and llm is not None:
                if api_delay > 0:
                    await asyncio.sleep(api_delay)
                faith = await _generate_and_check_faithfulness(
                    llm, verifier, query, docs[: min(rel_passed, 3)]
                )
                row.update(faith)
                if api_delay > 0 and faith.get("faithfulness_evaluated"):
                    await asyncio.sleep(api_delay)
            else:
                row["faithfulness_evaluated"] = False
                row["faithfulness_passed"] = False
        except Exception as exc:
            from verification.gemini_verifier import GeminiQuotaExhaustedError
            if isinstance(exc, GeminiQuotaExhaustedError) or _is_daily_quota_error(str(exc)):
                row["judge_error"] = "daily_quota_exhausted"
                raise
            row["judge_error"] = str(exc)[:200]
            row["faithfulness_evaluated"] = False
            row["faithfulness_passed"] = False

    return row


async def run_batch(
    golden: List[Dict[str, Any]],
    index_path: Path,
    *,
    full: bool = False,
    limit: Optional[int] = None,
    judge_k: int = 2,
    api_delay: float = 15.0,
    in_domain_only: bool = False,
    proxy_judge: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from config import Config
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.sparse_retriever import SparseRetriever
    from retrieval.hybrid_retriever import HybridRetriever

    if limit:
        golden = golden[:limit]
    if full and in_domain_only:
        golden = [g for g in golden if g.get("expect_retrieval")]

    dense = DenseRetriever()
    sparse = SparseRetriever()
    retriever = HybridRetriever(dense, sparse)
    if not retriever.load(str(index_path)):
        raise RuntimeError(
            f"No index at {index_path}. Build index from Data Sources tab first."
        )

    threshold = Config.get_effective_retrieval_threshold("")
    verifier = None
    llm = None

    if full and proxy_judge:
        print(
            f"[ragas_batch] Proxy-judge mode: reranker>=0.5 or cosine>={threshold} "
            f"on top-{judge_k} chunks (no Gemini API)"
        )
    elif full:
        if not Config.GEMINI_API_KEY:
            raise RuntimeError(
                "--full requires GEMINI_API_KEY for LLM-as-judge evaluation."
            )
        from verification.gemini_verifier import GeminiVerifier

        verifier = GeminiVerifier.from_config()
        try:
            Config.validate()
            llm = Config.get_llm()
        except Exception as exc:
            print(f"[ragas_batch] LLM generation skipped: {exc}")
            llm = None
        print(
            f"[ragas_batch] Full mode: model={Config.GEMINI_MODEL}, "
            f"judge top-{judge_k} chunks, {api_delay}s delay between API calls"
        )
    else:
        api_delay = 0.0

    if not full:
        api_delay = 0.0

    rows: List[Dict[str, Any]] = []
    started = time.time()
    quota_exhausted = False
    for i, item in enumerate(golden, 1):
        print(f"[ragas_batch] ({i}/{len(golden)}) {item.get('id')}: {item['query'][:50]}")
        try:
            row = await evaluate_one(
                item,
                retriever,
                threshold,
                full=full,
                verifier=verifier,
                llm=llm,
                judge_k=judge_k,
                api_delay=api_delay if full and not proxy_judge else 0.0,
                proxy_judge=proxy_judge,
            )
        except Exception as exc:
            if full and _is_daily_quota_error(str(exc)):
                print(f"[ragas_batch] Daily Gemini quota hit — saving partial results ({len(rows)} done).")
                quota_exhausted = True
                break
            print(f"[ragas_batch] Query failed ({item.get('id')}): {exc}")
            row = {
                "id": item.get("id"),
                "query": item["query"],
                "category": item.get("category"),
                "expect_retrieval": bool(item.get("expect_retrieval")),
                "error": str(exc)[:300],
            }
        else:
            pass
        rows.append(row)

    summary = {
        "mode": "full_proxy" if (full and proxy_judge) else ("full" if full else "retrieval_only"),
        "golden_count": len(golden),
        "evaluated_count": len(rows),
        "quota_exhausted": quota_exhausted,
        "threshold": threshold,
        "judge_k": judge_k if full else None,
        "api_delay_sec": api_delay if full else None,
        "gemini_model": Config.GEMINI_MODEL if full else None,
        "elapsed_sec": round(time.time() - started, 2),
        "retrieval": aggregate_retrieval_metrics(
            [r for r in rows if "retrieval_passed" in r]
        ),
    }
    if full:
        summary["ragas_proxies"] = aggregate_judge_metrics(rows)
        if proxy_judge:
            summary["ragas_proxies"]["note"] = "context_relevancy uses reranker/cosine proxy, not LLM judge"

    return rows, summary


def write_results(rows: List[Dict[str, Any]], summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps({"type": "row", **row}, ensure_ascii=False) + "\n")


def print_report(summary: Dict[str, Any]) -> None:
    print("\n=== RAGAS Batch Evaluation ===")
    print(f"Mode:           {summary['mode']}")
    print(f"Queries:        {summary['golden_count']}")
    print(f"Threshold:      {summary['threshold']}")
    print(f"Elapsed:        {summary['elapsed_sec']}s")

    r = summary["retrieval"]
    print("\n--- Retrieval (labeled golden set) ---")
    print(f"Precision:      {r['precision']:.2%}  (TP={r['tp']} FP={r['fp']})")
    print(f"Recall:         {r['recall']:.2%}  (TP={r['tp']} FN={r['fn']})")
    print(f"F1:             {r['f1']:.2%}")
    print(f"Hit@k in-domain:{r['hit_at_k_in_domain']:.2%}")
    print(f"Off-topic block:{r['off_topic_block_rate']:.2%}")

    proxies = summary.get("ragas_proxies")
    if proxies:
        print("\n--- RAGAS proxies (LLM-as-judge) ---")
        if summary.get("quota_exhausted"):
            print("⚠️  Stopped early: Gemini daily quota exhausted (partial eval)")
        if summary.get("gemini_model"):
            print(f"Model:             {summary['gemini_model']}")
        if proxies["context_relevancy_rate"] is not None:
            print(f"Context relevancy: {proxies['context_relevancy_rate']:.2%} "
                  f"(n={proxies['relevance_judged_queries']})")
        if proxies["faithfulness_rate"] is not None:
            print(f"Faithfulness:      {proxies['faithfulness_rate']:.2%} "
                  f"(n={proxies['faithfulness_evaluated']})")
        if proxies["first_pass_faithfulness_rate"] is not None:
            print(f"First-pass faith.: {proxies['first_pass_faithfulness_rate']:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RAGAS-aligned batch eval")
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help="Path to golden_set.json",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Path to index_store directory",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run Gemini relevance + faithfulness judges (requires API key)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N queries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help="Write JSONL results path",
    )
    parser.add_argument(
        "--judge-k",
        type=int,
        default=2,
        help="Max chunks to LLM-judge per query in --full mode (default 2)",
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=15.0,
        help="Seconds between Gemini API calls in --full mode (default 15)",
    )
    parser.add_argument(
        "--in-domain-only",
        action="store_true",
        help="Skip off-topic queries in --full mode (saves API quota)",
    )
    parser.add_argument(
        "--proxy-judge",
        action="store_true",
        help="Use reranker/cosine proxy for context relevancy (no Gemini API)",
    )
    args = parser.parse_args()

    golden = load_golden(args.golden)
    rows, summary = asyncio.run(
        run_batch(
            golden,
            args.index,
            full=args.full,
            limit=args.limit,
            judge_k=args.judge_k,
            api_delay=args.api_delay,
            in_domain_only=args.in_domain_only,
            proxy_judge=args.proxy_judge,
        )
    )
    write_results(rows, summary, args.output)
    print_report(summary)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
