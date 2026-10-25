# Quick Start Guide

## 1. Install dependencies

From this directory, create and activate a virtual environment, then install
the project requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure an LLM provider

Copy the example configuration and set the API key for the provider you use:

```bash
cp .env.example .env
```

The default `LLM_BACKEND` is `perplexity`. You can also configure `gemini` or
`ollama` in `.env`. Do not commit real API keys.

PDF indexing and Deep-Read also analyze every page image to transcribe
handwriting and describe charts/diagrams. Configure `VISION_PROVIDER=auto`
with `OPENAI_API_KEY` or `GEMINI_API_KEY`; set `VISION_MODEL` to override the
default model. For local-only processing, set `VISION_PROVIDER=ollama` and
choose a vision-capable model. Cloud provider settings transmit page images to
that service. If no vision provider is configured, vision-enabled indexing and
Deep-Read report a configuration error instead of silently skipping analysis.

## 3. Start the application

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860). Upload one or more PDFs
in the interface and build an index before asking research questions.

## Example research questions

- Summarize the main contributions of these papers.
- What limitations do the documents mention?
- Compare the methods proposed in these papers.
- What datasets were used?
- What future research directions are suggested?

For MCP client setup, see [MCP_README.md](MCP_README.md).
