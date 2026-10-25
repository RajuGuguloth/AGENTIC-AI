import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.ragas_native import (
    _collect_samples,
    _extract_response_text,
    _load_golden,
    run_retrieval_diagnostics,
)


class TestRagasRetrievalDiagnostics(unittest.TestCase):
    def test_extracts_text_from_structured_llm_content(self):
        response = SimpleNamespace(
            content=[
                {"type": "text", "text": "Answer from"},
                {"type": "text", "text": "retrieved context"},
            ]
        )
        self.assertEqual(
            _extract_response_text(response),
            "Answer from\nretrieved context",
        )

    def test_rejects_non_text_llm_response(self):
        with self.assertRaisesRegex(TypeError, "must be text"):
            _extract_response_text(SimpleNamespace(content=None))

    def test_rejects_empty_or_non_string_golden_queries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            golden_path = Path(temp_dir) / "golden.json"
            for invalid_query in ("", 123, None):
                golden_path.write_text(
                    json.dumps([{"query": invalid_query}]),
                    encoding="utf-8",
                )
                with self.subTest(query=invalid_query), self.assertRaisesRegex(
                    ValueError, "non-empty string query"
                ):
                    _load_golden(golden_path)

            golden_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one query"):
                _load_golden(golden_path)

    def test_traces_retrieval_without_llm_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            golden_path = Path(temp_dir) / "golden.json"
            golden_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "q1",
                            "query": "self attention",
                            "category": "in_domain",
                        },
                        {
                            "id": "q2",
                            "query": "positional encoding",
                            "category": "in_domain",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            fake_retriever = SimpleNamespace(
                search=lambda query, k: [
                    SimpleNamespace(
                        page_content="Retrieved evidence",
                        metadata={
                            "source": "paper.pdf",
                            "page": 3,
                            "retrieval_score": 0.8,
                            "child_rrf_score": 0.03,
                            "reranker_score": 0.9,
                        },
                    )
                ]
            )

            with patch(
                "evaluation.ragas_native._load_retriever",
                return_value=fake_retriever,
            ):
                bundle = run_retrieval_diagnostics(
                    golden_path,
                    Path(temp_dir) / "index",
                    top_k=1,
                )

        self.assertFalse(bundle["summary"]["llm_called"])
        self.assertEqual(bundle["summary"]["golden_count"], 2)
        self.assertEqual(bundle["summary"]["retrieval_empty"], 0)
        self.assertEqual(bundle["rows"][0]["retrieved_sources"][0]["page"], 3)
        self.assertEqual(bundle["rows"][0]["retrieved_sources"][0]["reranker_score"], 0.9)

    def test_stops_on_answer_generation_error(self):
        fake_retriever = SimpleNamespace(
            search=lambda query, k: [
                SimpleNamespace(page_content="retrieved context", metadata={})
            ]
        )

        with patch(
            "evaluation.ragas_native._generate_answer",
            new=AsyncMock(side_effect=RuntimeError("503 UNAVAILABLE")),
        ):
            with self.assertRaisesRegex(RuntimeError, "answer generation failed"):
                asyncio.run(
                    _collect_samples(
                        [{"id": "q1", "query": "question", "ground_truth": "answer"}],
                        fake_retriever,
                        llm=object(),
                        top_k=1,
                    )
                )

    def test_collect_samples_honors_requested_top_k(self):
        seen = []
        fake_retriever = SimpleNamespace(
            search=lambda query, k: (
                seen.append(k)
                or [
                    SimpleNamespace(
                        page_content=f"context {index}",
                        metadata={},
                    )
                    for index in range(k)
                ]
            )
        )

        with patch(
            "evaluation.ragas_native._generate_answer",
            new=AsyncMock(return_value="answer"),
        ):
            samples = asyncio.run(
                _collect_samples(
                    [{"id": "q1", "query": "question"}],
                    fake_retriever,
                    llm=object(),
                    top_k=2,
                )
            )

        self.assertEqual(seen, [2])
        self.assertEqual(len(samples[0]["retrieved_contexts"]), 2)

    def test_does_not_score_unknown_answer_after_generation_error(self):
        fake_retriever = SimpleNamespace(
            search=lambda query, k: [
                SimpleNamespace(page_content="retrieved context", metadata={})
            ]
        )

        with patch(
            "evaluation.ragas_native._generate_answer",
            new=AsyncMock(side_effect=RuntimeError("invalid provider configuration")),
        ):
            with self.assertRaisesRegex(RuntimeError, "answer generation failed"):
                asyncio.run(
                    _collect_samples(
                        [{"id": "q1", "query": "question", "ground_truth": "answer"}],
                        fake_retriever,
                        llm=object(),
                        top_k=1,
                    )
                )


if __name__ == "__main__":
    unittest.main()
