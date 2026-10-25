# Agentic RAG Research Assistant

An agentic research assistant that indexes PDFs and arXiv papers, retrieves
evidence with hybrid dense and sparse search, and answers questions with
sources. It includes document chat, deep-read reports, evaluation and
observability workflows, a Gradio UI served by FastAPI, and an optional MCP
server.

## Quick start

Requirements: Python 3.10+ and credentials for the configured LLM provider.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure the selected LLM provider in `.env`. The default `LLM_BACKEND` is
`perplexity`; `gemini` and `ollama` are also supported. PDF page-vision analysis
needs an image-capable provider: OpenAI, Gemini, or a vision-capable Ollama
model. Page images are sent to the configured cloud provider when OpenAI or
Gemini vision is selected.

Start the app:

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860), upload PDFs under **Data
Sources**, build the index, and then use Research or Document Chat. The first
run downloads local embedding and reranker models. Indexes and other runtime
data are stored locally and are not included in Git.

## API, MCP, and Docker

The FastAPI service exposes authenticated RAG endpoints including `/v1/search`,
`/v1/query`, `/v1/ingest`, and `/v1/upload`. Set a unique `RAG_API_KEY` in
`.env`; all endpoints except `/v1/health` require the `X-API-Key` header.

MCP is included in the main `requirements.txt`. See
[MCP_README.md](./MCP_README.md) for server and client configuration. The
example client configuration is in
[mcp_client_config.example.json](./mcp_client_config.example.json).

For container deployment, run `docker compose up --build` from this directory.
The compose setup persists indexes, artifacts, logs, and downloaded models in
Docker volumes.

## Evaluation

Run the native RAGAS evaluation after building the matching paper index and
configuring an LLM provider:

```bash
python evaluation/ragas_native.py \
  --golden evaluation/golden_set_ragas.json \
  --index index_store
```

The runner evaluates the 12 reference-answer questions in the golden set and
writes per-question results and metric averages to
`evaluation/ragas_native_results.jsonl`. It makes provider API calls for answer
generation and judging; embeddings run locally. Results apply only to the
corpus represented by that golden set. This baseline uses a context-only
generation prompt rather than the app's full document-chat or Self-RAG
orchestration, so it is not an end-to-end score for the production pipeline.

To inspect retrieval without calling an LLM provider:

```bash
python evaluation/ragas_native.py --retrieval-only \
  --golden evaluation/golden_set_ragas.json \
  --index index_store
```

This writes ranked sources and retrieval/reranker scores to
`evaluation/ragas_retrieval_trace.jsonl`; it does not calculate answer-quality
RAGAS metrics. The separate `ragas_batch.py` report includes custom retrieval
metrics and proxy scores; do not present those as native RAGAS metrics.

## Further documentation

- [Quick start](./QUICK_START.md)
- [Project guide](./PROJECT_GUIDE.md)
- [MCP setup](./MCP_README.md)
- [Level 5 features](./LEVEL5_README.md)
