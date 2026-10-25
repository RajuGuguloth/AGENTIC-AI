import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestRagApiAuth(unittest.TestCase):
    def setUp(self):
        from api import rag_router

        self.rag_router = rag_router
        self.api = FastAPI()
        self.api.include_router(rag_router.router)
        self.client = TestClient(self.api)

    def test_protected_route_rejects_missing_and_invalid_keys(self):
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}):
            self.assertEqual(self.client.get("/v1/papers").status_code, 401)
            self.assertEqual(
                self.client.get("/v1/papers", headers={"X-API-Key": "wrong"}).status_code,
                401,
            )

    def test_protected_route_accepts_configured_key(self):
        fake = type(
            "FakeRetriever",
            (),
            {
                "parent_store": {
                    "p1": Document(
                        page_content="text",
                        metadata={"source": "paper.pdf"},
                    )
                }
            },
        )()
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router, "get_hybrid_retriever", return_value=fake
        ):
            response = self.client.get(
                "/v1/papers",
                headers={"X-API-Key": "test-key"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["distinct_sources"], 1)
        self.assertEqual(response.json()["papers"][0]["source"], "paper.pdf")

    def test_protected_route_fails_closed_without_configured_key(self):
        with patch.dict(os.environ, {"RAG_API_KEY": ""}):
            response = self.client.get(
                "/v1/papers",
                headers={"X-API-Key": "anything"},
            )
        self.assertEqual(response.status_code, 503)

    def test_upload_indexes_pdf_bytes_through_shared_service(self):
        uploaded = []

        def capture_upload(paths, **kwargs):
            uploaded.append(paths[0].read_bytes())
            return {"total_children": 1, "files": ["paper.pdf"]}

        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router,
            "_index_paths",
            side_effect=capture_upload,
        ):
            response = self.client.post(
                "/v1/upload",
                headers={"X-API-Key": "test-key"},
                files=[("files", ("paper.pdf", b"%PDF-test content", "application/pdf"))],
                data={"replace": "true", "with_vision": "false"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total_children"], 1)
        self.assertEqual(uploaded, [b"%PDF-test content"])

    def test_search_returns_ranked_parent_chunks(self):
        from types import SimpleNamespace

        fake = type(
            "FakeRetriever",
            (),
            {
                "dense": SimpleNamespace(is_ready=True),
                "search": lambda self, query, k: [
                    Document(
                        page_content="evidence",
                        metadata={
                            "source": "paper.pdf",
                            "parent_id": "parent-1",
                            "page": 2,
                            "retrieval_score": 0.8,
                        },
                    )
                ],
            },
        )()
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router, "get_hybrid_retriever", return_value=fake
        ):
            response = self.client.post(
                "/v1/search",
                headers={"X-API-Key": "test-key"},
                json={"query": "question", "top_k": 1},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["chunks"][0]["source_id"], "parent-1")
        self.assertEqual(response.json()["chunks"][0]["text"], "evidence")

    def test_query_returns_answer_source_and_trace(self):
        from types import SimpleNamespace

        fake = type("FakeRetriever", (), {"dense": SimpleNamespace(is_ready=True)})()
        expected = {
            "answer": "grounded answer",
            "sources": "- paper.pdf (page 2)",
            "trace_id": "trace-1",
        }
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router, "get_hybrid_retriever", return_value=fake
        ), patch.object(self.rag_router, "_answer_question", return_value=expected):
            response = self.client.post(
                "/v1/query",
                headers={"X-API-Key": "test-key"},
                json={"question": "question"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"answered": True, **expected})


class TestSharedIndexRebuild(unittest.TestCase):
    def test_replace_and_append_keep_parent_child_index_consistent(self):
        from retrieval import runtime

        class FakeDense:
            documents = []
            is_ready = False

            def build_index(self, documents):
                self.documents = list(documents)
                self.is_ready = True

        class FakeSparse:
            def build_index(self, documents):
                self.documents = list(documents)

        class FakeRetriever:
            def __init__(self):
                self.dense = FakeDense()
                self.sparse = FakeSparse()
                self.parent_store = {"old": Document(page_content="old")}
                self.saved_path = None

            def add_parents(self, documents):
                for document in documents:
                    self.parent_store[document.metadata["parent_id"]] = document

            def save(self, path):
                self.saved_path = path

        retriever = FakeRetriever()
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            runtime, "_hybrid_retriever", retriever
        ), patch.object(runtime, "_index_store_path", Path(temp_dir)):
            parent_a = Document(page_content="new parent", metadata={"parent_id": "a"})
            child_a = Document(page_content="new child", metadata={"parent_id": "a"})
            replaced = runtime.rebuild_index([parent_a], [child_a], replace=True)
            self.assertEqual(replaced["total_parents"], 1)
            self.assertEqual(set(retriever.parent_store), {"a"})

            parent_b = Document(page_content="another parent", metadata={"parent_id": "b"})
            child_b = Document(page_content="another child", metadata={"parent_id": "b"})
            appended = runtime.rebuild_index([parent_b], [child_b], replace=False)
            self.assertEqual(appended["total_parents"], 2)
            self.assertEqual(appended["total_children"], 2)
            self.assertEqual(set(retriever.parent_store), {"a", "b"})
            self.assertEqual(len(retriever.dense.documents), len(retriever.sparse.documents))
