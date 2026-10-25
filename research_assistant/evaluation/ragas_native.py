#!/usr/bin/env python3
"""
True RAGAS-native evaluation using the `ragas` library (not proxies).

Metrics computed here (from the ragas package):
  - faithfulness          — claim-level groundedness of the answer in retrieved context
  - answer_relevancy      — semantic match of the answer to the query
  - context_precision     — rank-aware relevance of retrieved chunks (uses ground_truth if available)
  - context_recall        — ground_truth claims recoverable from retrieved chunks (requires ground_truth)
  - answer_correctness    — F1 of claims vs ground_truth (requires ground_truth)

Usage:
  python evaluation/ragas_native.py --golden evaluation/golden_set_ragas.json
  python evaluation/ragas_native.py --limit 3        # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_set_ragas.json"
DEFAULT_INDEX = ROOT / "index_store"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "ragas_native_results.jsonl"
DEFAULT_RETRIEVAL_TRACE = Path(__file__).resolve().parent / "ragas_retrieval_trace.jsonl"


def _load_golden(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Golden set must be a JSON array: {path}")
    if not data:
        raise ValueError(f"Golden set must contain at least one query: {path}")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Golden item at position {index} must be a JSON object.")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Golden item at position {index} needs a non-empty string query.")
        reference = item.get("ground_truth")
        if reference is not None and not isinstance(reference, str):
            raise ValueError(
                f"Golden item at position {index} has a non-string ground_truth."
            )
    return data


def _extract_response_text(resp: Any) -> str:
    content = getattr(resp, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        if text_parts:
            return "\n".join(text_parts)
        raise TypeError("LLM response did not contain text content.")
    raise TypeError(
        f"LLM response content must be text, got {type(content).__name__}."
    )


async def _generate_answer(llm, question: str, contexts: List[str]) -> str:
    """Ask the configured LLM to answer using ONLY the retrieved contexts."""
    from langchain_core.prompts import ChatPromptTemplate

    joined = "\n\n---\n\n".join(contexts)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer strictly using the provided context. If the context does not "
                "contain the answer, reply exactly: I don't know.",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ]
    )
    chain = prompt | llm
    resp = await chain.ainvoke({"context": joined, "question": question})
    return _extract_response_text(resp)


async def _collect_samples(
    golden: List[Dict[str, Any]],
    retriever,
    llm,
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    """For each golden item, retrieve top-k contexts and generate an answer."""
    from retrieval.query_utils import normalize_technical_query

    samples: List[Dict[str, Any]] = []
    for i, item in enumerate(golden, 1):
        query = item["query"]
        search_query = normalize_technical_query(query)
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(None, retriever.search, search_query, top_k)
        docs = docs[:top_k]
        contexts = [d.page_content for d in docs]
        if not contexts:
            answer = "I don't know."
        else:
            try:
                answer = await _generate_answer(llm, query, contexts)
            except Exception as exc:
                raise RuntimeError(
                    "Stopping native RAGAS evaluation because answer generation "
                    f"failed for {item.get('id')}: {exc}"
                ) from exc
        print(f"[ragas_native] ({i}/{len(golden)}) {item.get('id', '')} — retrieved {len(contexts)} chunks")
        samples.append(
            {
                "id": item.get("id"),
                "user_input": query,
                "retrieved_contexts": contexts,
                "retrieved_sources": [
                    {
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "retrieval_score": doc.metadata.get("retrieval_score"),
                        "reranker_score": doc.metadata.get("reranker_score"),
                    }
                    for doc in docs
                ],
                "response": answer,
                "reference": item.get("ground_truth"),
                "category": item.get("category"),
            }
        )
    return samples


def _select_metrics(has_ground_truth: bool):
    """Return the list of ragas metrics to run. Metrics that need a reference
    are only included when every sample has one."""
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        LLMContextPrecisionWithoutReference,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        AnswerCorrectness,
    )

    metrics = [Faithfulness(), ResponseRelevancy()]
    if has_ground_truth:
        metrics.append(LLMContextPrecisionWithReference())
        metrics.append(LLMContextRecall())
        metrics.append(AnswerCorrectness())
    else:
        metrics.append(LLMContextPrecisionWithoutReference())
    return metrics


def _wrap_llm_and_embeddings(embedding_model: Any):
    """Wrap the configured LangChain LLM + embeddings for ragas.

    ragas needs a chat model (for judgement) and an embeddings model (for
    similarity-based metrics like answer_relevancy). Reuse the retrieval
    embedder when available to avoid loading a second model copy.
    """
    from config import Config
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings

    Config.validate()
    llm = Config.get_llm()
    if Config.LLM_BACKEND == "gemini":
        llm.max_retries = 0

    class ExistingModelEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            vectors = embedding_model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vectors.tolist()

        def embed_query(self, text: str) -> List[float]:
            vector = embedding_model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vector.tolist()

    embeddings = ExistingModelEmbeddings()

    ragas_llm = (
        LangchainLLMWrapper(llm, bypass_n=True)
        if Config.LLM_BACKEND == "gemini"
        else LangchainLLMWrapper(llm)
    )
    return llm, ragas_llm, LangchainEmbeddingsWrapper(embeddings)


def _load_retriever(index_path: Path):
    from retrieval.dense_retriever import DenseRetriever
    from retrieval.sparse_retriever import SparseRetriever
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(DenseRetriever(), SparseRetriever())
    if not retriever.load(str(index_path)):
        raise RuntimeError(
            f"No index at {index_path}. Build one from the Data Sources tab first."
        )
    return retriever


def run_retrieval_diagnostics(
    golden_path: Path,
    index_path: Path,
    *,
    limit: Optional[int] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Trace local retrieval stages without calling an LLM provider."""
    from retrieval.query_utils import normalize_technical_query

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    golden = _load_golden(golden_path)
    if limit:
        golden = golden[:limit]

    retriever = _load_retriever(index_path)
    rows: List[Dict[str, Any]] = []
    retrieval_scores: List[float] = []
    reranker_scores: List[float] = []

    for i, item in enumerate(golden, 1):
        query = item["query"]
        docs = retriever.search(normalize_technical_query(query), k=top_k)
        hits = []
        for doc in docs:
            metadata = doc.metadata or {}
            retrieval_score = metadata.get("retrieval_score")
            reranker_score = metadata.get("reranker_score")
            if retrieval_score is not None:
                retrieval_scores.append(float(retrieval_score))
            if reranker_score is not None:
                reranker_scores.append(float(reranker_score))
            hits.append(
                {
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "retrieval_score": retrieval_score,
                    "rrf_score": metadata.get("child_rrf_score"),
                    "reranker_score": reranker_score,
                    "context_characters": len(doc.page_content),
                }
            )
        rows.append(
            {
                "id": item.get("id"),
                "query": query,
                "category": item.get("category"),
                "retrieval_query": normalize_technical_query(query),
                "retrieved_count": len(hits),
                "retrieved_sources": hits,
            }
        )
        print(f"[ragas_native] retrieval {i}/{len(golden)} {item.get('id')}: {len(hits)} hits")

    total = len(rows)
    summary = {
        "mode": "retrieval_diagnostics_only",
        "llm_called": False,
        "golden_count": total,
        "top_k": top_k,
        "retrieval_empty": sum(row["retrieved_count"] == 0 for row in rows),
        "queries_with_results": sum(row["retrieved_count"] > 0 for row in rows),
        "average_results_per_query": round(
            sum(row["retrieved_count"] for row in rows) / total, 3
        ) if total else 0.0,
        "mean_retrieval_score": round(
            sum(retrieval_scores) / len(retrieval_scores), 4
        ) if retrieval_scores else None,
        "mean_reranker_score": round(
            sum(reranker_scores) / len(reranker_scores), 4
        ) if reranker_scores else None,
    }
    return {"summary": summary, "rows": rows}


async def run_ragas_native(
    golden_path: Path,
    index_path: Path,
    *,
    limit: Optional[int] = None,
    top_k: int = 5,
    max_workers: int = 1,
) -> Dict[str, Any]:
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    golden = _load_golden(golden_path)
    if limit:
        golden = golden[:limit]

    retriever = _load_retriever(index_path)

    llm, ragas_llm, ragas_emb = _wrap_llm_and_embeddings(retriever.dense.model)

    print(f"[ragas_native] Collecting {len(golden)} samples (top_k={top_k})...")
    started = time.time()
    samples = await _collect_samples(golden, retriever, llm, top_k=top_k)

    has_ground_truth = all((s.get("reference") or "").strip() for s in samples)
    print(f"[ragas_native] has_ground_truth={has_ground_truth}")

    metrics = _select_metrics(has_ground_truth)

    ragas_rows = [
        {
            "user_input": s["user_input"],
            "retrieved_contexts": s["retrieved_contexts"],
            "response": s["response"],
            **({"reference": s["reference"]} if has_ground_truth else {}),
        }
        for s in samples
    ]
    eval_dataset = EvaluationDataset.from_list(ragas_rows)

    print(f"[ragas_native] Running metrics: {[m.name for m in metrics]}")
    result = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RunConfig(
            timeout=120,
            max_retries=1,
            max_wait=2,
            max_workers=max_workers,
        ),
        raise_exceptions=True,
    )

    scores_df = result.to_pandas()
    per_row: List[Dict[str, Any]] = []
    for sample, (_, row) in zip(samples, scores_df.iterrows()):
        row_out = {
            "id": sample["id"],
            "query": sample["user_input"],
            "category": sample.get("category"),
            "answer_preview": (sample["response"] or "")[:400],
            "contexts_used": len(sample["retrieved_contexts"]),
            "retrieved_sources": sample["retrieved_sources"],
        }
        for metric in metrics:
            key = metric.name
            if key in row:
                val = row[key]
                score = None if val is None else float(val)
                row_out[key] = score if score is not None and math.isfinite(score) else None
        per_row.append(row_out)

    aggregates: Dict[str, Optional[float]] = {}
    metric_coverage: Dict[str, Dict[str, int]] = {}
    for metric in metrics:
        key = metric.name
        if key not in scores_df.columns:
            metric_coverage[key] = {"scored": 0, "total": len(per_row)}
            aggregates[key] = None
            continue
        col = scores_df[key].dropna()
        metric_coverage[key] = {"scored": int(len(col)), "total": len(per_row)}
        aggregates[key] = round(float(col.mean()), 4) if len(col) else None

    pipeline_diagnostics = {
        "queries": len(samples),
        "retrieval_empty": sum(not s["retrieved_contexts"] for s in samples),
        "answers_abstained": sum(
            "i don't know" in s["response"].lower()
            or "cannot find sufficient evidence" in s["response"].lower()
            for s in samples
        ),
    }

    summary = {
        "golden_count": len(golden),
        "evaluated_count": len(per_row),
        "top_k": top_k,
        "elapsed_sec": round(time.time() - started, 2),
        "metrics_run": [m.name for m in metrics],
        "aggregates": aggregates,
        "metric_coverage": metric_coverage,
        "pipeline_diagnostics": pipeline_diagnostics,
        "has_ground_truth": has_ground_truth,
    }
    return {"summary": summary, "rows": per_row}


def _write_results(bundle: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "summary", **bundle["summary"]}, ensure_ascii=False) + "\n")
        for row in bundle["rows"]:
            f.write(json.dumps({"type": "row", **row}, ensure_ascii=False) + "\n")


def _print_report(summary: Dict[str, Any]) -> None:
    print("\n=== RAGAS Native Evaluation ===")
    print(f"Queries:      {summary['golden_count']}")
    print(f"Top-k:        {summary['top_k']}")
    print(f"Ground truth: {'yes' if summary['has_ground_truth'] else 'no'}")
    print(f"Elapsed:      {summary['elapsed_sec']}s")
    print("\n--- Metric averages ---")
    for name, value in summary["aggregates"].items():
        if value is None:
            print(f"{name:32s} n/a (no samples scored)")
        else:
            print(f"{name:32s} {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="True RAGAS-native evaluation")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent RAGAS metric tasks (default 1 to limit memory and API rate)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Trace local retrieval without calling Gemini or running answer-quality metrics",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.retrieval_only:
        bundle = run_retrieval_diagnostics(
            args.golden,
            args.index,
            limit=args.limit,
            top_k=args.top_k,
        )
        output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_RETRIEVAL_TRACE
        _write_results(bundle, output)
        print("\n=== Retrieval Pipeline Diagnostics (no LLM/RAGAS answer metrics) ===")
        print(json.dumps(bundle["summary"], indent=2))
        print(f"Trace written to {output}")
        return

    bundle = asyncio.run(
        run_ragas_native(
            args.golden,
            args.index,
            limit=args.limit,
            top_k=args.top_k,
            max_workers=args.max_workers,
        )
    )
    _write_results(bundle, args.output)
    _print_report(bundle["summary"])
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
