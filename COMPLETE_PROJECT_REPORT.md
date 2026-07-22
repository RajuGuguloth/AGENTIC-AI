# COMPLETE PROJECT REPORT — Agentic RAG Research Assistant

**Purpose of this document:** Ground-truth reference for Claude (or any LLM) to understand **exactly** what was built, what was measured, what works, what does not, and what was **not** used.  
**Rule for LLMs reading this:** Do not invent features, metrics, or frameworks not listed here. When uncertain, say "not implemented" or "not measured."

**Project root:** `/Users/raju/Downloads/AGENTIC-AI-main/research_assistant/`  
**App entry:** `python app.py` → `http://127.0.0.1:7860`  
**Primary test corpus:** Transformer paper PDF `1706.03762v7.pdf` (~80 indexed child chunks)

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Goals](#2-problem-statement--goals)
3. [Evolution: What Was Built in Phases](#3-evolution-what-was-built-in-phases)
4. [End-to-End System Architecture](#4-end-to-end-system-architecture)
5. [User-Facing Application (5 Tabs)](#5-user-facing-application-5-tabs)
6. [Pipeline A — Indexing (Offline)](#6-pipeline-a--indexing-offline)
7. [Pipeline B — Chat RAG with Verification](#7-pipeline-b--chat-rag-with-verification)
8. [Pipeline C — Research (Simple + Agentic)](#8-pipeline-c--research-simple--agentic)
9. [Pipeline D — Deep-Read (Report + PPT)](#9-pipeline-d--deep-read-report--ppt)
10. [Pipeline E — Evaluation & Monitoring](#10-pipeline-e--evaluation--monitoring)
11. [Level 5: Autonomous / Self-Improving Layer](#11-level-5-autonomous--self-improving-layer)
12. [Complete File Map (Every Module)](#12-complete-file-map-every-module)
13. [Technology Stack (Exact Tools & Models)](#13-technology-stack-exact-tools--models)
14. [Configuration & Hyperparameters](#14-configuration--hyperparameters)
15. [Measured Results (Only Defensible Numbers)](#15-measured-results-only-defensible-numbers)
16. [Production Logs State (As of Last Run)](#16-production-logs-state-as-of-last-run)
17. [Bugs Fixed During Development](#17-bugs-fixed-during-development)
18. [Known Limitations & Honest Gaps](#18-known-limitations--honest-gaps)
19. [What This Project Is NOT](#19-what-this-project-is-not)
20. [Commands to Run Everything](#20-commands-to-run-everything)
21. [Resume / Interview Claims — Truth Table](#21-resume--interview-claims--truth-table)
22. [Cost Per Request (Estimated)](#22-cost-per-request-estimated)
23. [Suggested Learning Path for the Author](#23-suggested-learning-path-for-the-author)

---

## 1. Executive Summary

**Agentic RAG Research Assistant** is a Python web application that lets a user upload an academic PDF (primarily tested on the Transformer paper), build a searchable index, and then:

- **Chat** with verified Q&A (retrieve → judge relevance → generate → judge groundedness → log)
- **Research** via simple summarize or multi-agent sub-query workflow
- **Deep-Read** arXiv/PDF into a structured markdown report + PowerPoint
- **Evaluate** retrieval quality on a 30-query golden set with automated orchestration

The system is **not** a deployed cloud product. It is a **production-oriented prototype**: structured logging, evaluation harness, optimization hooks, unit tests — but small index (~80 chunks, one paper), few live traces, and Gemini API quota limits blocked full faithfulness evaluation.

**Core innovation pattern:** Hybrid retrieval (FAISS + BM25 + RRF + CrossEncoder rerank) + parent-child chunking + **Self-RAG / Corrective RAG** verification (LLM-as-judge before and after generation).

**Orchestration style:** Custom Python `asyncio` agents + LangChain prompts. **LangGraph is NOT used** in project code (only present as transitive dependency in venv via LangChain).

---

## 2. Problem Statement & Goals

### Problem
General LLMs answer from training memory and can **hallucinate** on technical PDF content. Users need answers **grounded in their uploaded document** with traceability (sources, scores, logs).

### Goals Achieved
| Goal | Status |
|------|--------|
| Upload PDF and build searchable index | ✅ Done |
| Hybrid retrieval better than dense-only | ✅ Measured F1 93.88% on golden set |
| Verify chunks before generation | ✅ `GeminiVerifier.is_relevant()` |
| Verify answer after generation + retry | ✅ `is_grounded()` + up to 3 retries |
| Multi-step research for complex goals | ✅ Planner + parallel `SubQueryAgent` |
| Structured first-time paper reading | ✅ Deep-Read pipeline |
| Offline evaluation on labeled queries | ✅ 30-query golden set + `ragas_batch.py` |
| Production-style logging | ✅ `logs/rag_traces.jsonl` |
| Automated eval orchestration | ✅ `eval_orchestrator.py` |
| Self-tuning (threshold, prompts, healing) | ✅ Code exists (Level 5); not fully proven at scale |

### Goals Partially Achieved
| Goal | Status |
|------|--------|
| Full RAGAS faithfulness % on 30 queries | ❌ Blocked by Gemini daily quota |
| Stable production verification pass rate | ❌ Only 2 chat traces logged |
| Multi-document / web-scale index | ❌ Single paper prototype |
| LangGraph orchestration | ❌ Not implemented |

---

## 3. Evolution: What Was Built in Phases

This is the logical build order inferred from architecture (Levels 1→5):

### Phase 1 — Core RAG Foundation
- PDF loading (`etl/pdf_loader.py` — pypdf)
- Parent-child semantic chunking (`etl/chunker.py`)
- Dense FAISS index (`retrieval/dense_retriever.py` — SentenceTransformer)
- Sparse BM25 index (`retrieval/sparse_retriever.py`)
- Hybrid RRF fusion + CrossEncoder rerank (`retrieval/hybrid_retriever.py`)
- Gradio UI + index persistence (`app.py`, `index_store/`)

### Phase 2 — Verified Chat (Self-RAG)
- Document chat pipeline (`orchestration/document_chat.py`)
- Gemini relevance + groundedness judges (`verification/gemini_verifier.py`)
- Regeneration loop with stricter prompts (`MAX_RETRIES=3`)
- JSONL tracing (`observability/gemini_tracer.py`)
- User feedback API (`api/feedback_router.py`, `observability/feedback.py`)

### Phase 3 — Agentic Research
- Research planner (`orchestration/planner.py`) — decomposes goal into sub-queries
- Parallel sub-query agents (`orchestration/agent.py`) — Self-RAG per sub-query
- Report synthesizer (`orchestration/synthesizer.py`)
- Simple summarize fast path (`orchestration/simple_summarizer.py`)
- Chart generator (`visualization/chart_generator.py`)

### Phase 4 — Deep-Read Pipeline
- URL/arXiv ingest (`etl/ingest.py`)
- Rich PDF extraction — text, figures, tables, page images (`etl/pdf_rich_loader.py`)
- Six parallel section agents (`orchestration/deep_read/section_agents.py`)
- Report merger + PPT builder (`merger.py`, `ppt_builder.py`)
- Provider-agnostic LLM gateway (`llm/gateway.py`)
- Artifacts output (`artifacts/deep_read/{job_id}/`)

### Phase 5 — Evaluation & Level 5 Autonomy
- Golden set 30 queries (`evaluation/golden_set.json`)
- Batch eval runner (`evaluation/ragas_batch.py`)
- Daily production metrics (`evaluation/daily_eval.py`)
- Multi-agent eval orchestrator (`evaluation/eval_orchestrator.py`)
- Threshold grid search (`optimization/threshold_optimizer.py`)
- Prompt A/B optimizer (`optimization/prompt_optimizer.py`)
- User memory / preferences (`memory/user_memory.py`)
- Multimodal CLIP retrieval (`retrieval/multimodal_retriever.py`)
- Query cache stub (`retrieval/query_cache.py`)
- Self-healing (`observability/self_healer.py`)
- Unit tests 13/13 pass (`tests/test_level5.py`)
- Docs: `PROJECT_GUIDE.md`, `INTERVIEW_PREP.md`, `LEVEL5_README.md`

---

## 4. End-to-End System Architecture

```
USER (Browser)
    │
    ▼
app.py — Gradio UI (5 tabs) + FastAPI (port 7860)
    │
    ├─── INDEXING ─── PDF → chunker → FAISS + BM25 → index_store/
    │
    ├─── CHAT ─── document_chat.py
    │       query_utils → multimodal_retriever.search()
    │       → gemini_verifier (relevance ×6)
    │       → LangChain LLM generate
    │       → gemini_verifier (groundedness, retry ≤3)
    │       → gemini_tracer → logs/rag_traces.jsonl
    │
    ├─── RESEARCH (simple) ─── simple_summarizer.py (1 retrieve + 1 LLM)
    │
    ├─── RESEARCH (agentic) ─── planner → agent.py (parallel) → synthesizer → chart_generator
    │
    ├─── DEEP-READ ─── ingest → pdf_rich_loader → section_agents (×6) → merger + ppt_builder
    │
    └─── EVALUATION (CLI) ─── eval_orchestrator → ragas_batch → daily_eval
```

**Shared retrieval brain:** `MultimodalRetriever` extends `HybridRetriever` — used by Chat and Research. Deep-Read does **not** use FAISS index.

---

## 5. User-Facing Application (5 Tabs)

| Tab | User Action | Backend Function | Output |
|-----|-------------|------------------|--------|
| **📚 Data Sources** | Upload PDF, Build Index | `build_index_from_pdfs()` | `index_store/` persisted |
| **💬 Chat** | Ask question | `chat_turn()` → `answer_document_question()` | Answer + sources + trace_id |
| **🔍 Research** | Summarize or Run Agentic RAG | `summarize_indexed_pdf()` or `run_research_workflow()` | Report, chart, sub-queries, findings |
| **📖 Deep-Read** | arXiv URL or PDF upload | `run_deep_read()` | report.md + slides.pptx |
| **⚙️ Configuration** | View settings | Reads `config.py` / `.env` | Display only |

**FastAPI endpoint:** `POST /api/feedback` — thumbs up/down linked to `trace_id`.

---

## 6. Pipeline A — Indexing (Offline)

### Step-by-step
1. **Load PDF** — `etl/pdf_loader.py` uses `pypdf.PdfReader`, one LangChain `Document` per page with metadata: `source`, `page`, `file_path`.
2. **Parent-child chunk** — `etl/chunker.py`:
   - **Parents:** `RecursiveCharacterTextSplitter`, size=**1500**, overlap=**200**
   - **Children:** `SemanticChunker` with `HuggingFaceEmbeddings(all-MiniLM-L6-v2)`, `breakpoint_threshold_type="percentile"`
   - Each child has `parent_id` UUID linking to parent
3. **Dense index** — `dense_retriever.build_index(child_docs)`:
   - `SentenceTransformer(all-MiniLM-L6-v2)` → projected to **512 dims**
   - `faiss.IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
   - Optional CLIP for image chunks (`openai/clip-vit-base-patch32`)
4. **Sparse index** — `sparse_retriever.build_index(child_docs)`:
   - `BM25Okapi` from `rank_bm25`, k1=1.5, b=0.75
5. **Parent store** — `hybrid_retriever.add_parents(parent_docs)` — in-memory dict
6. **Persist** — `hybrid_retriever.save("./index_store/")`:
   - FAISS index, documents pickle, BM25 pickle, `parent_store.pkl`

### Measured index size (example run)
- **80 vectors** (child chunks)
- Source paper: **1706.03762v7.pdf** (Attention Is All You Need)

---

## 7. Pipeline B — Chat RAG with Verification

**File:** `orchestration/document_chat.py`  
**Pattern:** Self-RAG / Corrective RAG (NOT LangGraph)

### Execution order (every chat turn)

| Step | What happens | Module | Key params |
|------|--------------|--------|------------|
| 1 | Session + user memory boost | `memory/user_memory.py` | `LEVEL5_ENABLED=true` |
| 2 | Query normalization | `retrieval/query_utils.py` | e.g. "self attention" → "self-attention" |
| 3 | Hybrid search | `multimodal_retriever.search()` | TOP_K_DENSE=20, TOP_K_SPARSE=20, RRF_K=60 |
| 4 | Score filter on dense | `score_filter.py` | cosine ≥ 0.35 (0.25 technical) |
| 5 | RRF fuse child rankings | `hybrid_retriever.py` | |
| 6 | Map children → parents | `hybrid_retriever.py` | dedupe, fetch up to 12 parents |
| 7 | CrossEncoder rerank | `hybrid_retriever.py` | `BAAI/bge-reranker-large`, max_length=512 |
| 8 | Return top parents | | TOP_K_FINAL=**6** |
| 9 | Relevance judge per chunk | `gemini_verifier.is_relevant()` | skip if cosine < threshold; context truncated 500 chars |
| 10 | Relevance fallback | `document_chat.py` | if all rejected: keep docs with reranker≥0.5 OR cosine≥0.25, max 3 |
| 11 | Build context string | `_build_context_from_docs()` | `[source, page]` headers |
| 12 | Optional A/B prompt | `prompt_optimizer.get_prompt()` | task="generation" |
| 13 | Generate answer | LangChain `ChatPromptTemplate \| llm` | Gemini temp=0.7, max_tokens=1200 |
| 14 | Groundedness judge | `is_grounded()` | context≤800, answer≤800 chars |
| 15 | Regenerate if fail | `_regenerate_until_grounded()` | MAX_RETRIES=**3**, exponential backoff |
| 16 | Log trace | `gemini_tracer.log_trace()` | `logs/rag_traces.jsonl` |

### Gemini verifier details (`verification/gemini_verifier.py`)
- Model: `Config.GEMINI_MODEL` (currently `gemini-2.0-flash` in `.env`)
- Judge temperature: **0.1**, max_output_tokens: **32**
- Prompts: `RELEVANCE_PROMPT`, `GROUNDEDNESS_PROMPT` — YES/NO only
- Handles `GeminiQuotaExhaustedError` on daily quota 429
- Fail-closed on API errors (returns False for relevance/groundedness)

### Chat does NOT log to a separate path for Research agents
- Only `path: "chat"` traces are written from `document_chat.py`
- Research agent path does **not** write to `rag_traces.jsonl` today

---

## 8. Pipeline C — Research (Simple + Agentic)

### C1 — Simple Summarize (`orchestration/simple_summarizer.py`)
- **1×** hybrid retrieval + **1×** LLM call
- No planner, no judges
- Designed for Gemini quota constraints and fast PDF summary
- Used by "Summarize PDF Now" button

### C2 — Full Agentic Research (`app.py` → `run_research_workflow`)

```
Research Goal
    │
    ▼
ResearchPlanner.plan()          [orchestration/planner.py]
    LLM structured output → up to MAX_SUB_QUERIES (default 4) sub-queries
    │
    ▼
run_parallel_research()         [orchestration/agent.py]
    asyncio.gather(SubQueryAgent × N)
    │
    Per SubQueryAgent (Self-RAG loop):
    1. retriever.search(sub_query)
    2. grade_relevance() — GeminiVerifier or structured RelevanceScore
    3. If no relevant docs → rewrite_query() → search again
    4. agent.answer(sub_query, context)
    5. check_hallucination() — groundedness
    6. If fail → regenerate once with strict context append
    │
    ▼
ReportSynthesizer.synthesize()  [orchestration/synthesizer.py]
    Merge sub-answers → final markdown report
    │
    ▼
ChartGenerator                  [visualization/chart_generator.py]
    Optional chart from report text
```

**Agentic definition in this project:** Planner decomposes goal; parallel workers each run retrieve→grade→answer→verify. **Not** open-ended tool-calling loops. **Not** LangGraph.

---

## 9. Pipeline D — Deep-Read (Report + PPT)

**Separate pipeline — does NOT use FAISS index.**

| Step | Module | Output |
|------|--------|--------|
| Resolve PDF | `etl/ingest.py` | Local PDF from arXiv URL, PDF URL, or upload |
| Rich extract | `etl/pdf_rich_loader.py` | `RichPDFDocument`: text, figures, tables, page PNGs |
| 6 section agents (parallel) | `section_agents.py` | Introduction, Problem, Method, Figures/Tables, Results, Limitations |
| LLM calls | `llm/gateway.py` | temp=0.25, max_tokens=2500 per section |
| Merge report | `merger.py` | `deep_read_report.md` |
| Build slides | `ppt_builder.py` | `slides.pptx` with paper images |
| Artifacts | `artifacts/deep_read/{job_id}/` | All outputs |

### Example successful run
- Job ID: `9f0ef9ce`
- Paper: arXiv 1706.03762 (Transformer)
- Output: `artifacts/deep_read/9f0ef9ce/deep_read_report.md` (6-section structured report)

---

## 10. Pipeline E — Evaluation & Monitoring

### 10.1 Golden Set (`evaluation/golden_set.json`)
- **30 queries total**
- **24 in-domain** (Transformer paper topics)
- **6 off-topic** (quantum chemistry, stock trading, pasta recipe, etc.)
- Each row: `id`, `query`, `expect_retrieval` (true/false), `category`

### 10.2 Batch Eval (`evaluation/ragas_batch.py`)

**Modes:**
| Command | What it does |
|---------|--------------|
| `python evaluation/ragas_batch.py` | Retrieval-only, no API |
| `python evaluation/ragas_batch.py --full` | + Gemini judges + answer generation |
| `python evaluation/ragas_batch.py --full --proxy-judge` | Judges replaced by reranker≥0.5 proxy |

**Retrieval metrics computed:**
- TP, FP, TN, FN at query level
- Precision, Recall, F1
- Hit@k (in-domain)
- Off-topic block rate

**RAGAS-aligned proxies (not official RAGAS library):**
- Context relevancy rate (proxy: reranker score)
- Faithfulness rate (requires full LLM judge — often blocked)

### 10.3 Eval Orchestrator (`evaluation/eval_orchestrator.py`)

Four logical agents:
1. **Validator** — index exists, Gemini key, API probe, unit tests (13/13)
2. **Monitor** — reads `logs/rag_traces.jsonl`, last batch eval results
3. **Decision** — if API blocked → `full_proxy` mode; else full LLM judge
4. **Executor** — runs `ragas_batch.py` with chosen args

**Output:** `evaluation/orchestrator_report.json`, `evaluation/ragas_results_full.jsonl`

### 10.4 Daily Eval (`evaluation/daily_eval.py`)

Reads `logs/rag_traces.jsonl` + `logs/feedback.jsonl`, computes:
- `verification_pass_rate`
- `context_relevancy_rate`
- `empty_retrieval_rate`
- `regeneration_rate`
- `verification_coverage`
- `first_pass_faithfulness_rate`
- `relevance_fallback_rate`
- `avg_latency_ms`
- `user_satisfaction` (from feedback)

Alerts if `verification_pass_rate < 0.7`

---

## 11. Level 5: Autonomous / Self-Improving Layer

Documented in `LEVEL5_README.md`. Code exists; **not all loops proven at production scale.**

| Capability | Module | Schedule (intended) |
|------------|--------|---------------------|
| Threshold grid search | `optimization/threshold_optimizer.py` | Weekly |
| User memory | `memory/user_memory.py` | Every chat turn |
| Prompt A/B optimization | `optimization/prompt_optimizer.py` | Daily when pass rate < 0.7 for 3 days |
| Multimodal CLIP search | `retrieval/multimodal_retriever.py` | On demand |
| Query cache | `retrieval/query_cache.py` | Redis TTL=3600 (optional) |
| Self-healing | `observability/self_healer.py` | Hourly cron (via `migration_level5.sh`) |

### Self-healer recovery actions
| Anomaly detected | Action |
|------------------|--------|
| Empty retrieval > 50% | Lower MIN_RETRIEVAL_SCORE by 0.1 |
| Latency > 2× baseline | Enable query cache |
| Verification drop > 20% | Rollback generation prompt |

### User memory signals
| Signal | Learned preference |
|--------|-------------------|
| Thumbs up on short answers | `prefers_concise` |
| Follow-ups about pages | `wants_page_citations` |
| Repeated query terms | `boost_terms` for retrieval |

---

## 12. Complete File Map (Every Module)

```
research_assistant/
├── app.py                          # Main UI, all tab wiring, uvicorn :7860
├── config.py                       # All env settings, get_llm(), thresholds
├── requirements.txt                # Core dependencies
├── requirements_level5.txt         # Redis, pytesseract, scikit-learn extras
├── migration_level5.sh             # Level 5 setup + cron suggestions
│
├── etl/
│   ├── pdf_loader.py               # pypdf page extraction
│   ├── chunker.py                  # Parent-child semantic chunking
│   ├── ingest.py                   # arXiv/URL → PDF download
│   ├── pdf_rich_loader.py          # Deep-Read: figures, tables, pages
│   └── universal_loader.py         # Multi-format loader (images etc.)
│
├── retrieval/
│   ├── dense_retriever.py          # FAISS + SentenceTransformer + CLIP
│   ├── sparse_retriever.py         # BM25Okapi
│   ├── hybrid_retriever.py         # RRF + CrossEncoder rerank + parents
│   ├── multimodal_retriever.py     # CLIP fusion (0.6 text + 0.4 image)
│   ├── score_filter.py             # Cosine threshold filter
│   └── query_utils.py              # Technical query normalization
│   └── query_cache.py              # Redis/in-memory cache
│
├── orchestration/
│   ├── document_chat.py            # Verified chat RAG pipeline
│   ├── simple_summarizer.py        # 1 retrieve + 1 LLM
│   ├── planner.py                  # ResearchPlanner → sub-queries
│   ├── agent.py                    # SubQueryAgent + parallel research
│   ├── synthesizer.py              # Final report merge
│   └── deep_read/
│       ├── orchestrator.py         # Deep-Read main flow
│       ├── section_agents.py       # 6 parallel section LLM calls
│       ├── merger.py               # Markdown report assembly
│       └── ppt_builder.py          # PowerPoint generation
│
├── verification/
│   └── gemini_verifier.py          # Relevance + groundedness judge
│
├── llm/
│   └── gateway.py                  # Provider-agnostic async LLM (Deep-Read)
│
├── observability/
│   ├── gemini_tracer.py            # rag_traces.jsonl logging
│   ├── feedback.py                 # feedback.jsonl append
│   └── self_healer.py              # Anomaly detection + recovery
│
├── optimization/
│   ├── common.py                   # runtime_config.json, alerts
│   ├── threshold_optimizer.py      # Grid search cosine threshold
│   └── prompt_optimizer.py         # A/B prompt variants
│
├── memory/
│   └── user_memory.py              # Cross-session preferences
│
├── evaluation/
│   ├── golden_set.json             # 30 labeled test queries
│   ├── ragas_batch.py              # Offline batch evaluation
│   ├── daily_eval.py               # Production metrics from logs
│   ├── eval_orchestrator.py        # Validator+Monitor+Decision+Executor
│   ├── orchestrator_report.json    # Last orchestrator run output
│   └── ragas_results_full.jsonl    # Per-query eval results
│
├── api/
│   └── feedback_router.py          # FastAPI POST /api/feedback
│
├── visualization/
│   └── chart_generator.py          # Auto chart from research report
│
├── tests/
│   └── test_level5.py              # 13 unit tests (all pass)
│
├── logs/
│   ├── rag_traces.jsonl            # Production chat traces (2 entries)
│   └── feedback.jsonl              # User thumbs feedback
│
├── index_store/                    # Persisted FAISS + BM25 + parents
├── artifacts/deep_read/            # Generated reports + PPTs
│
├── PROJECT_GUIDE.md                # Beginner architecture guide
├── INTERVIEW_PREP.md               # Interview Q&A + cost estimates
└── LEVEL5_README.md                # Level 5 autonomy documentation
```

---

## 13. Technology Stack (Exact Tools & Models)

| Layer | Technology | Used for |
|-------|------------|----------|
| Language | Python 3 | Everything |
| UI | Gradio 4+ | Web tabs, chat, file upload |
| API | FastAPI + Uvicorn | Feedback endpoint, mount Gradio |
| LLM (primary) | Google Gemini (`gemini-2.0-flash` in .env) | Generation + verification |
| LLM alt | Perplexity, Ollama | Configurable via `LLM_BACKEND` |
| LLM framework | LangChain | Prompts, chains, structured output |
| **NOT used** | **LangGraph** | Not in project code |
| **NOT used** | **Official RAGAS Python package** | Custom RAGAS-aligned metrics only |
| Embeddings | SentenceTransformer `all-MiniLM-L6-v2` | Dense vectors (local, free) |
| Vector DB | FAISS `IndexFlatIP` dim=512 | Similarity search (local) |
| Keyword search | BM25Okapi (`rank_bm25`) | Sparse retrieval (local) |
| Reranker | CrossEncoder `BAAI/bge-reranker-large` | Parent reranking (local) |
| Multimodal | CLIP `openai/clip-vit-base-patch32` | Image + text fusion |
| PDF (index) | pypdf | Page text extraction |
| PDF (deep-read) | PyMuPDF (fitz) + pdfplumber | Rich extraction |
| Slides | python-pptx | Deep-Read PPT |
| Async | asyncio | Parallel agents, chat retries |
| Logging | JSONL files | Traces, feedback, healing |
| Tests | unittest | `test_level5.py` |
| Optional cache | Redis | `query_cache.py` |

---

## 14. Configuration & Hyperparameters

From `config.py` and `.env` (defaults shown; `.env` may override):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `LLM_BACKEND` | gemini | Primary LLM |
| `GEMINI_MODEL` | gemini-2.0-flash | Generation + judges |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Dense embeddings |
| `TOP_K_DENSE` | 20 | FAISS candidates |
| `TOP_K_SPARSE` | 20 | BM25 candidates |
| `TOP_K_FINAL` | 6 | Final parent chunks to LLM |
| `RRF_K` | 60 | RRF fusion constant |
| `MIN_RETRIEVAL_SCORE` | 0.35 | Cosine pre-filter (normal) |
| `MIN_RETRIEVAL_SCORE_LOW` | 0.25 | Cosine pre-filter (technical queries) |
| Parent chunk size | 1500 chars | `chunker.py` default |
| Parent overlap | 200 chars | `chunker.py` default |
| `GEMINI_TEMPERATURE` | 0.1 | Judge calls |
| `GEMINI_TEMPERATURE_CREATIVE` | 0.7 | Generation |
| `MAX_RETRIES` | 3 | Groundedness regeneration |
| `MAX_SUB_QUERIES` | 4 | Research planner limit |
| `MAX_TOKENS_ANSWER` | 1200 | Chat answer cap |
| `LEVEL5_ENABLED` | true | User memory, prompt optimizer |

---

## 15. Measured Results (Only Defensible Numbers)

**Source:** `evaluation/orchestrator_report.json` and `evaluation/ragas_results_full.jsonl`  
**Mode:** `full_proxy` (reranker-based context relevancy proxy; **no LLM faithfulness judge**)  
**Corpus:** Single indexed Transformer paper (`1706.03762v7.pdf`)  
**Date:** Last orchestrator run in repo

### Retrieval metrics (30 queries)
| Metric | Value |
|--------|-------|
| Retrieval F1 | **93.88%** (0.9388) |
| In-domain Hit@k | **95.83%** (0.9583) |
| Precision | **92%** (0.92) |
| Recall | **95.83%** (0.9583) |
| Off-topic block rate | **66.67%** (4/6 off-topic correctly blocked) |
| Context relevancy proxy | **93.75%** (reranker≥0.5, NOT LLM judge) |

### Confusion matrix (retrieval abstention task)
| | Predicted pass | Predicted fail |
|--|----------------|----------------|
| Should retrieve (24) | TP=23 | FN=1 |
| Should not retrieve (6) | FP=2 | TN=4 |

### NOT measured (do not claim)
| Metric | Status |
|--------|--------|
| Faithfulness rate (LLM judge) | `null` — 0 queries evaluated |
| First-pass faithfulness | `null` |
| Production verification pass rate | **0%** (2 traces, both failed groundedness or relevance) |
| Multi-paper / multi-domain generalization | Not tested |

### Unit tests
- **13/13 pass** (`tests/test_level5.py`)

---

## 16. Production Logs State (As of Last Run)

**File:** `logs/rag_traces.jsonl` — **2 traces only**

### Trace 1
- Query: `"what is self attention"`
- Result: `"No relevant documents after verification."`
- `relevance_passed: 0` of 6
- Latency: ~17.7s

### Trace 2
- Query: `"What is this document about?"`
- Answer generated about Transformer architecture
- `relevance_fallback: true` (judge rejected all; fallback used top 3)
- `grounded: false` after 3 attempts
- Latency: ~39.5s

**Implication:** Live chat verification is **implemented** but **not yet stable** in production traces. Offline retrieval metrics are much stronger than live groundedness pass rate.

---

## 17. Bugs Fixed During Development

| Bug | Fix | File |
|-----|-----|------|
| `chunks_scores` logged without numeric scores | Added `serialize_retrieved_docs()` with cosine + reranker | `observability/gemini_tracer.py` |
| Cosine threshold applied to reranker scores | `score_is_cosine` flag; only FAISS scores get 0.35 cutoff | `gemini_verifier.py`, `document_chat.py` |
| `grounded_after_retry` not set on success | Set when grounded after retry | `document_chat.py` |
| Gemini quota / response extraction errors | `GeminiQuotaExhaustedError`, safer `_extract_response_text` | `gemini_verifier.py` |
| Eval orchestrator blocked when API down | Auto-fallback to `full_proxy` mode | `eval_orchestrator.py` |
| Simple summarize fails on quota | Dedicated `format_llm_error()` user messages | `simple_summarizer.py` |

---

## 18. Known Limitations & Honest Gaps

1. **Small index** — one paper, ~80 chunks; not multi-document or web-scale
2. **Gemini free tier** — ~20 API calls/day; blocks full 30-query faithfulness eval
3. **Few production traces** — only 2 chat sessions logged; monitoring metrics unstable
4. **Research agent path** — does not write to `rag_traces.jsonl`
5. **Faithfulness not scored** — proxy eval only in last full run
6. **Off-topic blocking imperfect** — 66.7% (2 off-topic queries still retrieved weak matches)
7. **LangGraph not used** — custom asyncio orchestration only
8. **Self-healer / prompt optimizer** — code exists; cron not proven running in dev
9. **API keys in .env** — must never be committed to git; rotate if exposed
10. **q09 BLEU query** — retrieval_passed=false in eval (top cosine 0.287 < 0.35) — one FN in golden set

---

## 19. What This Project Is NOT

| Claim | Truth |
|-------|-------|
| "Uses LangGraph" | **NO** — not in project source code |
| "Uses official RAGAS library" | **NO** — custom RAGAS-aligned metrics |
| "Deployed to production cloud" | **NO** — local Gradio app |
| "93.9% answer accuracy" | **NO** — that is **retrieval F1**, not answer quality |
| "Proven hallucination reduction %" | **NO** — mechanism exists, metric not measured |
| "Multi-agent LangChain Agents framework" | **PARTIAL** — custom SubQueryAgent classes, not LangChain Agent executor |
| "Trained/fine-tuned models" | **NO** — uses pretrained embeddings + API LLM |
| "Web search RAG" | **NO** — PDF-only indexed corpus |

---

## 20. Commands to Run Everything

```bash
cd research_assistant

# Install
pip install -r requirements.txt
pip install -r requirements_level5.txt   # optional Level 5

# Configure .env (never commit)
# LLM_BACKEND=gemini
# GEMINI_API_KEY=AIza...   (Google AI Studio key)
# GEMINI_MODEL=gemini-2.0-flash

# Run app
python app.py
# → http://127.0.0.1:7860

# Offline retrieval eval (no API)
python evaluation/ragas_batch.py

# Full eval with proxy judge (30 queries, no API judges)
python evaluation/ragas_batch.py --full --proxy-judge

# Full eval with LLM judges (needs API quota)
python evaluation/ragas_batch.py --full

# Automated orchestrator (validator + monitor + decision + executor)
python evaluation/eval_orchestrator.py

# Daily production metrics
python evaluation/daily_eval.py

# Unit tests
python tests/test_level5.py
# or: python -m pytest tests/test_level5.py -v

# Level 5 tools (manual)
python optimization/threshold_optimizer.py
python observability/self_healer.py
```

---

## 21. Resume / Interview Claims — Truth Table

| Claim | Valid? | Notes |
|-------|--------|-------|
| Hybrid FAISS + BM25 + RRF + CrossEncoder | ✅ | Core retrieval |
| Parent-child chunking | ✅ | 1500 parent, semantic children |
| 93.9% retrieval F1 | ✅ | Say "retrieval F1", one-paper corpus |
| 95.8% Hit@k | ✅ | In-domain only |
| 92% precision | ✅ | On 30-query golden set |
| LLM-as-judge verification | ✅ | Implemented in chat |
| Parallel agentic sub-queries | ✅ | planner + asyncio.gather |
| "Reduce hallucinations" (measured) | ❌ | Mechanism yes, % no |
| LangGraph | ❌ | Not used |
| Official RAGAS | ❌ | Aligned concepts only |
| Production deployed | ❌ | Prototype |
| Multimodal CLIP | ✅ | Code exists; text path is primary |
| Self-healing + A/B prompts | ⚠️ | Implemented, lightly validated |
| Context relevancy 93.8% | ⚠️ | Proxy (reranker), not LLM judge |

---

## 22. Cost Per Request (Estimated)

**Gemini 2.0 Flash paid tier:** $0.10/1M input, $0.40/1M output tokens  
**Local retrieval (FAISS, BM25, CrossEncoder, embeddings):** $0

| Path | LLM calls | Est. cost (paid) |
|------|-----------|------------------|
| Chat typical | 8 (6 relevance + 1 gen + 1 grounded) | ~$0.0007 |
| Chat worst case | 12 (with 3 retries) | ~$0.002 |
| Research (4 agents) | ~34–40 | ~$0.006 |
| Deep-Read | 6 section calls | ~$0.004 |
| Simple summarize | 1 | ~$0.0006 |

**Free tier bottleneck:** ~20 API calls/day → only 2–3 full chat questions, not dollars.

---

## 23. Suggested Learning Path for the Author

If you are re-learning what you built, study in this order:

1. **Indexing flow** — `pdf_loader.py` → `chunker.py` → `hybrid_retriever.py` → `index_store/`
2. **One retrieval query** — trace `hybrid_retriever.search()` line by line
3. **Chat verification** — `document_chat.py` + `gemini_verifier.py`
4. **Run offline eval** — `python evaluation/ragas_batch.py` and read `ragas_results_full.jsonl`
5. **Agentic path** — `planner.py` → `agent.py` → `synthesizer.py`
6. **Deep-Read** — `deep_read/orchestrator.py` (separate from RAG index)
7. **Level 5** — `LEVEL5_README.md` + `daily_eval.py`

### Questions to ask yourself (self-test)
- What is the difference between a child chunk and a parent chunk?
- Why RRF instead of averaging FAISS and BM25 scores?
- What happens if all 6 chunks fail relevance judge?
- Why is faithfulness null in eval report?
- Is this project LangGraph-based? (Answer: **No**)

---

## APPENDIX A — How to Use This Document with Claude

Paste this entire file at the start of a Claude conversation and add:

> "Answer only from this document. If something is not stated here, say 'not in project scope' or 'not measured.' Do not invent metrics, files, or frameworks."

This prevents hallucination on thesis questions, resume bullets, and architecture explanations.

---

## APPENDIX B — Related Docs in Repo

| File | Purpose |
|------|---------|
| `PROJECT_GUIDE.md` | Beginner-friendly guide + glossary |
| `INTERVIEW_PREP.md` | Interview script, cost, trending Q&A |
| `LEVEL5_README.md` | Autonomous RAG tuning guide |
| `COMPLETE_PROJECT_REPORT.md` | This file — full ground truth |

---

*Report generated from codebase state: 52 Python modules, orchestrator_report.json, ragas_results_full.jsonl, 2 production traces, 13 passing unit tests.*
