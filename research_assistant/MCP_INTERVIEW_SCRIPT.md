# MCP Interview One-Pager

The 15-minute version of what you built and how to talk about it.

---

## The 30-second pitch (memorise this)

> "I extended my Agentic RAG project by wrapping it as an **MCP server** — Model Context Protocol, Anthropic's open standard for how LLMs talk to external tools. It exposes six tools, including academic deep-read report and slide generation, plus resources for evaluation results and full retrieved chunks. Any MCP-compatible client — Claude Code, Claude Desktop, Cursor — can now use my RAG pipeline as native tools, without me writing a custom integration per client. Every call is trace-logged for observability."

**Word count: 90. Speaking time: ~30 seconds.**

**Structure**: Define MCP → what tools you expose → the payoff (any client can use it) → observability.

---

## What MCP is (in case they ask)

> "MCP stands for Model Context Protocol. It's an open standard released by Anthropic in November 2024. Think of it as **USB for AI** — before MCP, every LLM framework had its own way of defining tools; OpenAI, LangChain, Anthropic, Cursor, each different. MCP standardises the protocol between LLM clients and tool servers, so I write my tools once and they work across every MCP-compatible client."

---

## The six tools — and why I picked them

- **`index_status()`** — cheap readiness check. Every good tool suite has one; Claude uses it before deciding whether to call the expensive tools.
- **`list_indexed_papers()`** — lets Claude discover what documents exist before searching them.
- **`search_papers(query, top_k)`** — raw retrieval hits. Fast, no LLM. For when the caller wants evidence directly.
- **`ask_paper(question)`** — full agentic answer with retrieve → judge → generate → verify → retry.
- **`build_index_from_paths(pdf_paths)`** — ingestion. Chunk, embed, persist to disk.
- **`deep_read_paper(paper_url_or_path)`** — extract a paper, analyze its sections, and return a structured report plus slides.

The `resource://eval/latest_report` and `resource://papers/{source_id}` resources let a client inspect the latest evaluation JSON or load full text for a retrieved parent chunk.

**Design principle**: cheap tools first, expensive tools last. Claude decides which to call based on the question — this is exactly the tool-use pattern Anthropic recommends.

---

## What I chose NOT to duplicate

The MCP server is **transport glue, not a second RAG engine**. It calls the authenticated FastAPI service for index status, indexing, retrieval, and Q&A, so those operations share the app's process-wide retriever. `deep_read_paper` directly calls the existing deep-read pipeline.

Interviewers respect this. It shows you understand **separation of concerns**: MCP is the transport, not the brain.

---

## Architecture question — draw this on a whiteboard

```
Claude Code / Desktop  ──────► MCP Server (mcp_server.py)  ──────► Existing RAG Pipeline
     (MCP client)      stdio        (this file)             imports    - retrieval/
                       JSON-RPC                                        - orchestration/
                                                                        - verification/
                                                                        - observability/
```

Three layers. Client speaks JSON-RPC over stdio. The MCP adapter exposes six tools and read-only resources, then proxies the RAG operations to the shared FastAPI process. Deep-Read runs through the existing pipeline in the MCP process.

---

## Key technical decisions to defend

1. **One shared retrieval runtime.** The FastAPI process lazily loads FAISS, embeddings, and the reranker once; MCP sends retrieval operations to that process instead of loading a second copy. `tools/list` therefore does not load the index.

2. **Trace logs per MCP call.** Every call writes a JSONL entry to `logs/mcp_calls.jsonl` — tool name, arg preview, latency, result summary or error. This is observability at the transport layer, on top of the existing observability at the RAG layer.

3. **`stdio` transport.** MCP supports both stdio and HTTP/SSE. I chose stdio because Claude Code and Claude Desktop use stdio natively, and it's simpler — no port management, no auth.

4. **Explicit errors.** The API reports an empty-index status or request error, and the adapter surfaces failed API calls as MCP tool errors. The client can then ask to index a PDF with `build_index_from_paths`.

---

## Anticipated interview questions (with 1-line answers)

**Q: Why MCP instead of just a REST API?**
A: An MCP client already knows how to discover tools, format arguments, and stream results. A custom REST API means every client — Claude, Cursor, VS Code — needs its own adapter. MCP is one integration, N clients.

**Q: How does MCP differ from function calling in the OpenAI API?**
A: Function calling is a *model capability* — the model returns "call this function with these args" and it's your job to execute and pass results back. MCP is a *protocol* — the transport, discovery, error handling are all standardised. MCP servers work with any MCP-aware model, not just OpenAI's.

**Q: What if the tool call fails?**
A: The server returns a structured error inside `isError=true`. The client (Claude) sees it and can retry, ask for clarification, or call a different tool.

**Q: Can you do this over HTTP instead of stdio?**
A: Yes — MCP defines both. I chose stdio because Claude Desktop and Claude Code use it natively for local servers. HTTP+SSE is for hosted/remote MCP servers, where you'd need auth and TLS.

**Q: How does Claude know which tool to call?**
A: Each tool has a name, a docstring (from the Python function), and a JSON Schema for its arguments (auto-generated by FastMCP from the type hints). Claude reads the docstring/description at connect time and picks the right tool for the user's question.

**Q: What did you find hard?**
A: Sharing the warmed retrieval runtime across the UI, API, and MCP clients. I kept the MCP server lightweight by routing status, search, indexing, and Q&A through the authenticated FastAPI API instead of loading a second index. Deep-Read is the exception and runs in the MCP process.

---

## The demo (if you get to show it)

1. Show `mcp_server.py` in your editor. Point to the `@mcp.tool()` decorators.
2. Run the smoke-test script from `MCP_README.md`. Watch tools appear.
3. Show `logs/mcp_calls.jsonl` — one line per call with latency and result summary.
4. If time permits, describe how Claude Code would call `ask_paper("what is self-attention?")` and get a full grounded answer back.

**Do not overclaim**. Say "I wired it to Claude Code via stdio" — that's what you actually did.

---

## Resume line

Add this to the Agentic RAG project bullet list:

> "Wrapped the RAG pipeline as an MCP (Model Context Protocol) server exposing
> six native Claude tools — `search_papers`, `ask_paper`, `index_status`,
> `list_indexed_papers`, `build_index_from_paths`, `deep_read_paper` — plus
> evaluation and retrieved-chunk resources, with per-call trace logging
> and a shared API-backed retrieval runtime, making the same pipeline usable from any
> MCP-compatible client (Claude Code, Claude Desktop, Cursor)."

---

## What NOT to say

- ❌ "I built MCP." — MCP is Anthropic's protocol. You built a *server* that speaks MCP.
- ❌ "I implemented the protocol from scratch." — You used the `mcp` Python SDK. Say so.
- ❌ "It's production-ready." — It's a working demo. Say "prototype" or "working end-to-end demo".
- ❌ Reading code line-by-line. Point to the *decision*, not the *syntax*.

---

## If you have 2 more hours before the interview

- **Best use**: record a 90-second screen capture of you starting the server + running the smoke test + watching a trace log line appear. Attach to your GitHub or portfolio.
- **Second best**: run the server against a real indexed PDF and screenshot the actual retrieval result. Concrete evidence beats abstract description.
