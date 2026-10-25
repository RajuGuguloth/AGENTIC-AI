# MCP Server for Agentic RAG

An MCP (Model Context Protocol) adapter for the shared Agentic RAG FastAPI
service. MCP clients use one warmed retrieval runtime through HTTP rather than
loading a second FAISS index and embedding/reranker models in the MCP process.

## What it exposes

| Tool | Purpose |
|---|---|
| `index_status()` | Is the index ready? How many chunks? |
| `list_indexed_papers()` | Which source documents are currently indexed |
| `search_papers(query, top_k)` | Raw hybrid-retriever hits (fast, no LLM) |
| `ask_paper(question)` | Full agentic answer: retrieve → judge → generate → verify → retry |
| `build_index_from_paths(pdf_paths)` | Ingest new PDFs, chunk, embed, persist |
| `deep_read_paper(paper_url_or_path)` | Analyze page text, handwriting, and visuals; generate a report and PowerPoint |

The server also exposes these read-only resources:

| Resource URI | Purpose |
|---|---|
| `resource://eval/latest_report` | Latest `evaluation/orchestrator_report.json` |
| `resource://papers/{source_id}` | Full parent-chunk text for a `source_id` returned by `search_papers` |

Every tool and resource read is trace-logged to `logs/mcp_calls.jsonl` — name,
argument preview, latency, and either a result summary or an error.

## Architecture

```
┌───────────────────┐      MCP over stdio      ┌───────────────────┐
│   Claude Code /   │ ───── JSON-RPC ─────→    │   mcp_server.py   │
│   Claude Desktop  │                          │   (this file)     │
└───────────────────┘                          └────────┬──────────┘
                                                        │
                                                        │ imports
                                                        ▼
                          ┌───────────────────────────────────────────┐
                          │   Existing AGENTIC-AI project modules      │
                          │                                             │
                          │   retrieval/  → FAISS + BM25 + RRF + rerank│
                          │   orchestration/document_chat.py            │
                          │   orchestration/deep_read/ → report + slides│
                          │   verification/gemini_verifier.py           │
                          │   observability/gemini_tracer.py            │
                          │   evaluation/ → latest report resource      │
                          └───────────────────────────────────────────┘
```

The web UI and API share the same process-wide retriever. The MCP adapter calls
the authenticated API for status, indexing, retrieval, and Q&A; only
`deep_read_paper` runs the deep-read pipeline inside the MCP process.

```
┌─────────────────┐   MCP stdio   ┌────────────────────┐  HTTP + API key
│ MCP-compatible  │ ────────────→ │ mcp_server.py      │ ────────────────┐
│ client           │               │ lightweight adapter│                 ▼
└─────────────────┘               └────────────────────┘       ┌──────────────────┐
                                                                │ FastAPI + Gradio │
┌─────────────────┐                 shared process runtime      │ FAISS + BM25 RAG │
│ Browser / UI    │ ─────────────────────────────────────────→ │ persistent index │
└─────────────────┘                                            └──────────────────┘
```

## Setup

### 1. Configure and start the API

```bash
cd research_assistant
.venv-macos/bin/python app.py
```

The Mac environment is created with `python3.13 -m venv .venv-macos` and
`.venv-macos/bin/python -m pip install -r requirements.txt`. Keep `.env`
local and ignored by Git; it must contain both `RAG_API_KEY` and the selected
LLM provider credentials.

The API is available at `http://127.0.0.1:7860/v1`. The UI and API use the
same retriever and persisted index. `RAG_API_KEY` is separate from model
provider credentials and is required for all RAG operations except health.

### 2. Wire the server to your MCP client

Copy the `mcpServers` block from `mcp_client_config.example.json` into your
client's config file. Replace the Python and script paths with paths on your
machine and make the same `RAG_API_KEY` available to the MCP process. The MCP
server also accepts `RAG_API_URL` if the API is not on localhost.

**VS Code** — open `research_assistant` as the workspace folder. The
workspace-local `.vscode/mcp.json` uses `.venv-macos/bin/python` and loads
credentials from the ignored `.env` file. Use **⌘⇧P → MCP: List Servers** to
start or inspect `agentic-rag`, then restart the server after configuration
changes.

**Claude Code (WSL)** — edit `~/.claude/claude_desktop_config.json`
(create it if missing).

**Claude Desktop (Windows)** — edit
`%APPDATA%\Claude\claude_desktop_config.json`.

**Cursor** — Cursor Settings → MCP → Add server, then paste the block.

### 3. Restart the client

Restart the MCP server or fully reopen the client. On startup the client
spawns `mcp_server.py` as a subprocess and issues `tools/list`. If it worked,
your tools appear in the tool picker.

### 4. Verify with a smoke test

```bash
python - <<'PY'
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    import os
    params = StdioServerParameters(
        command=os.environ.get("PYTHON", "python"),
        args=["mcp_server.py"],
        env={
            "RAG_API_URL": os.environ.get("RAG_API_URL", "http://127.0.0.1:7860"),
            "RAG_API_KEY": os.environ["RAG_API_KEY"],
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            for t in tools.tools:
                print(t.name)

asyncio.run(main())
PY
```

Should print all six tool names. Resource templates appear in MCP clients that
support resource discovery.

## Usage examples

Once wired, you can ask Claude (in Claude Code or Desktop):

- "**Is my RAG index ready?**" → Claude calls `index_status()`.
- "**What papers do I have indexed?**" → `list_indexed_papers()`.
- "**Show me the top 3 chunks about self-attention.**" → `search_papers("self-attention", 3)`.
- "**What does the paper say about positional encoding?**" → `ask_paper("what does the paper say about positional encoding?")`.
- "**Index this file: /home/user/paper.pdf**" → `build_index_from_paths(["/home/user/paper.pdf"])`; the MCP adapter uploads it to the API.
- "**Deep-read this paper: /home/user/paper.pdf**" → `deep_read_paper("/home/user/paper.pdf")`.
- "**Show me the full text of that retrieved chunk**" → open
  `resource://papers/{source_id}` using the `source_id` from a `search_papers` result.
- "**Show the latest RAG evaluation results**" → read
  `resource://eval/latest_report`.

Claude picks which tool to call based on the wording. You don't have to name it.

## Docker deployment

The Docker image contains application code and dependencies, not indexes,
uploads, API keys, or generated artifacts. Set a strong `RAG_API_KEY` in
`research_assistant/.env`, then run:

```bash
docker compose up --build
```

Compose stores the FAISS/BM25 index, artifacts, logs, and Hugging Face model
cache in persistent named volumes. The API health check is available at
`http://127.0.0.1:7860/v1/health`. For a host-launched MCP client, keep
`RAG_API_URL=http://127.0.0.1:7860` and provide the same API key. Build/index
PDF paths are uploaded from the MCP host, so the API container does not need
access to the host's original PDF paths.
Compose binds the service to loopback by default. Put it behind a TLS reverse
proxy and use a deliberate network/access policy before exposing it remotely.
The container installs CPU-only PyTorch, so it does not require a CUDA host.

## Known caveats

- **Startup latency**: the API loads embedding and reranker models when the
  app starts; the first boot downloads those models if they are not cached.
- **No index yet?** The tools return a helpful error rather than crashing;
  call `build_index_from_paths` first.
- **Vision**: `with_vision=true` sends page images to the configured vision
  provider. Keep it disabled for local-only text extraction unless cloud or
  local vision settings are intentionally configured.
- **Security**: do not commit `.env` or paste `RAG_API_KEY` into public MCP
  client configurations.

## What's next (roadmap)

- Add an MCP prompt for "explain this paper to a beginner" so it's a one-click
  in Claude clients that surface prompts.
