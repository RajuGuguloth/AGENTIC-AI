"""
Unit tests for Level 5 Autonomous RAG capabilities.
Run: python -m pytest tests/test_level5.py -v
  or: python tests/test_level5.py
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestThresholdOptimizer(unittest.TestCase):
    def test_grid_search_selects_best_f1(self):
        from optimization.threshold_optimizer import (
            _compute_metrics_for_threshold,
            select_best_threshold,
            run_grid_search_batched,
        )

        traces = [
            {
                "trace_id": "t1",
                "chunks_scores": [{"score": 0.5}, {"score": 0.4}],
                "verification": {"relevance_passed": 2},
            },
            {
                "trace_id": "t2",
                "chunks_scores": [{"score": 0.28}],
                "verification": {"relevance_passed": 1},
            },
        ]
        feedback = {"t1": "positive"}
        results = run_grid_search_batched(traces, feedback)
        best = select_best_threshold(results)
        self.assertIn("threshold", best)
        self.assertIn("f1", best)

        low = _compute_metrics_for_threshold(traces, feedback, 0.50)
        high = _compute_metrics_for_threshold(traces, feedback, 0.25)
        self.assertGreaterEqual(high["recall_at_5"], low["recall_at_5"])


class TestUserMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        import memory.user_memory as um

        self.um = um
        um.USER_PROFILES_PATH = Path(self.tmp.name) / "user_profiles.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_preference_learning(self):
        uid = "test_user"
        self.um.update_from_feedback(uid, "positive", answer="Short.", metadata={})
        self.um.update_from_feedback(uid, "positive", answer="Brief.", metadata={})
        prompt = self.um.format_preference_prompt(uid)
        self.assertIn("concise", prompt.lower())

    def test_boost_query(self):
        uid = "term_user"
        profile = self.um.get_or_create_profile(uid)
        profile["boost_terms"] = {"transformer": 3, "attention": 2}
        self.um._save_all_profiles({uid: profile})
        boosted = self.um.boost_query(uid, "explain architecture")
        self.assertIn("transformer", boosted)


class TestPromptOptimizer(unittest.TestCase):
    def test_get_prompt_default(self):
        from optimization.prompt_optimizer import get_prompt, PROMPT_TEMPLATES

        name, text = get_prompt("generation", trace_id="abc")
        self.assertIn(name, PROMPT_TEMPLATES["generation"])
        self.assertTrue(len(text) > 10)

    def test_ab_outcome_tracking(self):
        import optimization.common as common
        from optimization.prompt_optimizer import record_ab_outcome, AB_MIN_SAMPLES

        with tempfile.TemporaryDirectory() as tmp:
            common.RUNTIME_CONFIG_PATH = Path(tmp) / "runtime.json"
            common.save_runtime_config({
                "ab_test": {
                    "generation": {
                        "variants": {"v_a": "prompt a", "v_b": "prompt b"},
                        "stats": {"v_a": {"samples": 0, "successes": 0}},
                    }
                }
            })
            record_ab_outcome("generation", "v_a", True)
            runtime = common.load_runtime_config()
            self.assertEqual(runtime["ab_test"]["generation"]["stats"]["v_a"]["samples"], 1)


class TestQueryCache(unittest.TestCase):
    def test_in_memory_cache(self):
        from retrieval.query_cache import QueryCache

        cache = QueryCache(ttl_seconds=60)
        cache.set("test query", [{"doc": 1}])
        hit = cache.get("test query")
        self.assertEqual(hit, [{"doc": 1}])
        cache.invalidate("test query")
        self.assertIsNone(cache.get("test query"))


class TestSelfHealer(unittest.TestCase):
    def test_anomaly_detection(self):
        from observability.self_healer import detect_anomalies

        recent = [
            {"ts": time.time(), "latency_ms": 20000, "verification": {"grounded": False}},
            {"ts": time.time(), "latency_ms": 25000, "verification": {"retrieval": "empty"}},
        ]
        baseline = {
            "verification_pass_rate": 0.9,
            "avg_latency_ms": 3000.0,
            "empty_retrieval_rate": 0.05,
        }
        anomalies = detect_anomalies(recent, baseline)
        types = {a["type"] for a in anomalies}
        self.assertTrue("latency_spike" in types or "empty_retrieval_spike" in types)

    def test_token_bucket(self):
        from observability.self_healer import GeminiTokenBucket

        bucket = GeminiTokenBucket(rate=100, capacity=2)
        self.assertTrue(bucket.acquire())
        self.assertTrue(bucket.acquire())
        self.assertFalse(bucket.acquire())


class TestDailyEval(unittest.TestCase):
    def test_compute_metrics(self):
        from evaluation.daily_eval import compute_metrics

        traces = [
            {
                "latency_ms": 1000,
                "verification": {
                    "grounded": True,
                    "relevance_passed": 2,
                    "relevance_total": 3,
                    "grounded_attempts": [{"attempt": 1, "grounded": True}],
                },
            },
            {
                "latency_ms": 3000,
                "verification": {
                    "grounded": False,
                    "relevance_passed": 0,
                    "relevance_total": 6,
                    "regeneration_count": 2,
                    "grounded_attempts": [
                        {"attempt": 1, "grounded": False},
                        {"attempt": 2, "grounded": False},
                    ],
                },
            },
            {
                "latency_ms": 500,
                "verification": {"retrieval": "empty"},
            },
        ]
        feedback = [{"rating": "positive"}, {"rating": "negative"}]
        metrics = compute_metrics(traces, feedback)
        self.assertEqual(metrics["verification_pass_rate"], 0.5)
        self.assertEqual(metrics["user_satisfaction"], 0.5)
        self.assertAlmostEqual(metrics["empty_retrieval_rate"], 1 / 3, places=4)
        self.assertAlmostEqual(metrics["context_relevancy_rate"], 0.5, places=4)
        self.assertAlmostEqual(metrics["regeneration_rate"], 1.0, places=4)
        self.assertAlmostEqual(metrics["verification_coverage"], 2 / 3, places=4)
        self.assertAlmostEqual(metrics["first_pass_faithfulness_rate"], 0.5, places=4)

    def test_serialize_retrieved_docs(self):
        from langchain_core.documents import Document
        from observability.gemini_tracer import serialize_retrieved_docs

        docs = [
            Document(
                page_content="text",
                metadata={
                    "source": "paper.pdf",
                    "page": 3,
                    "retrieval_score": 0.53,
                    "reranker_score": 0.81,
                },
            )
        ]
        out = serialize_retrieved_docs(docs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["score"], 0.53)
        self.assertEqual(out[0]["retrieval_score"], 0.53)
        self.assertEqual(out[0]["reranker_score"], 0.81)
        self.assertEqual(out[0]["source"], "paper.pdf")


class TestRagasBatch(unittest.TestCase):
    def test_aggregate_retrieval_metrics(self):
        from evaluation.ragas_batch import aggregate_retrieval_metrics

        rows = [
            {"expect_retrieval": True, "retrieval_passed": True},
            {"expect_retrieval": True, "retrieval_passed": False},
            {"expect_retrieval": False, "retrieval_passed": False},
            {"expect_retrieval": False, "retrieval_passed": True},
        ]
        m = aggregate_retrieval_metrics(rows)
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fn"], 1)
        self.assertEqual(m["tn"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)

    def test_aggregate_judge_metrics(self):
        from evaluation.ragas_batch import aggregate_judge_metrics

        rows = [
            {
                "relevance_passed": 3,
                "relevance_total": 6,
                "faithfulness_evaluated": True,
                "faithfulness_passed": True,
                "first_pass_grounded": True,
            },
            {
                "relevance_passed": 0,
                "relevance_total": 6,
                "faithfulness_evaluated": False,
            },
        ]
        m = aggregate_judge_metrics(rows)
        self.assertAlmostEqual(m["context_relevancy_rate"], 0.25, places=4)
        self.assertEqual(m["faithfulness_rate"], 1.0)
        self.assertEqual(m["faithfulness_evaluated"], 1)


class TestMultimodalFusion(unittest.TestCase):
    def test_fuse_embeddings(self):
        import numpy as np
        from retrieval.multimodal_retriever import TEXT_WEIGHT, IMAGE_WEIGHT

        def fuse(a, b, tw=TEXT_WEIGHT, iw=IMAGE_WEIGHT):
            fused = tw * a + iw * b
            norm = np.linalg.norm(fused)
            return fused / norm if norm > 0 else fused

        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        fused = fuse(a, b)
        self.assertAlmostEqual(float(np.linalg.norm(fused)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
