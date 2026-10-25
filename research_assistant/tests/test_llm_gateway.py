import unittest
from unittest.mock import patch

from llm.gateway import LLMGateway


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


class FakeAsyncClient:
    requests = []

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, *, headers, json):
        self.requests.append({"url": url, "headers": headers, "payload": json})
        return FakeResponse()


class TestOpenAICompatibleGateway(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeAsyncClient.requests = []
        self.client_patch = patch("llm.gateway.httpx.AsyncClient", FakeAsyncClient)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    async def test_text_completion_sends_bearer_key(self):
        gateway = LLMGateway("openai_compat", "test-model", "test-secret")

        response = await gateway.complete("system", "user")

        self.assertEqual(response, "ok")
        self.assertEqual(
            FakeAsyncClient.requests[0]["headers"]["Authorization"],
            "Bearer test-secret",
        )

    async def test_image_completion_sends_bearer_key(self):
        gateway = LLMGateway("openai", "test-model", "test-secret")

        response = await gateway.complete_with_image("system", "user", b"image-bytes")

        self.assertEqual(response, "ok")
        self.assertEqual(
            FakeAsyncClient.requests[0]["headers"]["Authorization"],
            "Bearer test-secret",
        )


if __name__ == "__main__":
    unittest.main()
