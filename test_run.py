import sys
sys.path.insert(0, './research_assistant')
import asyncio
from agent import research_agent
from vector_store import VectorStore
from embeddings import Embedder

class MockEmbedder:
    def embed_query(self, text):
        return [0.1] * 384
    def embed_documents(self, docs):
        return [[0.1]*384 for _ in docs]

class MockVectorStore:
    def search(self, embedding, k=5):
        return [(0.9, "mock chunk text", {"source": "mock_source.pdf"})]

async def main():
    vs = MockVectorStore()
    emb = MockEmbedder()
    res = await research_agent("What is transformer?", vs, emb)
    print(res["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())
