# Agentic RAG Research Assistant with MCP Integration

An agentic research assistant that ingests PDFs and arXiv papers, retrieves
evidence with hybrid dense and sparse search, and uses an LLM to answer
questions, summarize documents, and generate deeper research reports. An
optional Model Context Protocol (MCP) server exposes research tools to
MCP-compatible clients.

## Features

- PDF and arXiv document ingestion
- Hybrid retrieval using dense embeddings and BM25 with reciprocal-rank fusion
- Agentic research, document chat, verification, and deep-read workflows
- Gradio interface served by FastAPI
- MCP tools for compatible clients
- Evaluation, feedback, observability, and configurable LLM providers

## Quick start

Requirements: Python 3.10 or newer and an API key for the selected LLM provider.

```bash
cd research_assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set the credentials for your chosen provider. The default
`LLM_BACKEND` is `perplexity`; alternatively configure `gemini` or `ollama`.
PDF page-vision analysis also needs an image-capable provider (`OPENAI_API_KEY`,
`GEMINI_API_KEY`, or a configured vision-capable Ollama model). Page images
are sent to a cloud provider when OpenAI or Gemini vision is selected.
Then start the application:

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860). Upload PDFs in the
interface to build an index before asking questions. Indexes and runtime data
are local and are not included in the repository.

## MCP integration

MCP is included in `requirements.txt`. Follow the client configuration and
server instructions in
[`research_assistant/MCP_README.md`](research_assistant/MCP_README.md). The
example configuration is in
[`research_assistant/mcp_client_config.example.json`](research_assistant/mcp_client_config.example.json).

## Project documentation

- [Application quick start](research_assistant/QUICK_START.md)
- [Project guide](research_assistant/PROJECT_GUIDE.md)
- [MCP setup](research_assistant/MCP_README.md)
- [Level 5 features](research_assistant/LEVEL5_README.md)
- [Complete project report](COMPLETE_PROJECT_REPORT.md)

## Configuration and privacy

Keep API keys in your local `.env` file; never commit credentials. Local
indexes, user profiles, logs, and evaluation outputs are runtime data and are
ignored by Git. Use the example environment files as templates.
