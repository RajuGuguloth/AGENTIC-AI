# Resume Defense + Hot Repeated Questions — Agentic RAG Research Assistant

This doc fills the gaps not covered in `INTERVIEW_PREP.md` / `PROJECT_GUIDE.md`:

1. **Line-by-line resume defense** (every claim, backed by a code file)
2. **The "outdated / stale documents" question family** — your #1 repeated interview question, and a known honest gap in the code
3. **"Explain your architecture" — 3 timed scripts** (30 sec / 90 sec / deep)
4. **"How do you handle RAG evaluation" — one complete answer**
5. **Honesty caveats** so you never get caught overclaiming

> **Golden rule:** say "**retrieval** F1 93.9%" — never "accuracy." Say "**RAGAS-style / RAGAS-aligned** metrics" — you did NOT use the official `ragas` library, you computed the same dimensions yourself in `ragas_batch.py`. Getting caught on that one word destroys trust.

---

## PART 1 — LINE-BY-LINE RESUME DEFENSE

For each bullet: **what it claims → where it lives in code → the 20-second spoken defense → the trap to avoid.**

---

### RESUME LINE 1
> *"Agentic RAG Research Assistant: Python, LangChain, Gemini, FAISS, Sentence Transformers"*

**Where it lives:**
| Tech | File / proof |
|------|--------------|
| Python | entire repo |
| LangChain | `orchestration/document_chat.py` (chains, `Document`), `llm/gateway.py` |
| Gemini | `verification/gemini_verifier.py`, `config.py` (`GEMINI_MODEL`) |
| FAISS | `retrieval/dense_retriever.py` (`IndexFlatIP`) |
| Sentence Transformers | `retrieval/dense_retriever.py` (MiniLM embeddings), `retrieval/hybrid_retriever.py` (`CrossEncoder bge-reranker-large`) |

**Spoken defense (20s):**
> *"Python end-to-end. LangChain orchestrates the LLM calls and document objects. Gemini is my generation and judge model. FAISS is the local vector store using inner-product on L2-normalized vectors, which is cosine. Sentence Transformers gives me both the MiniLM bi-encoder for embeddings and the bge cross-encoder for reranking."*

**Trap:** If asked "why is it *agentic* and not just RAG?" → *"Chat is a fixed pipeline. The **Research** tab is the agentic part: a planner decomposes a goal into sub-queries and parallel worker agents each run their own retrieve–verify–answer loop. I use the word precisely."*

---

### RESUME LINE 2
> *"Built a production-oriented RAG pipeline with hybrid retrieval (FAISS + BM25 + Reciprocal Rank Fusion + Cross-Encoder reranking) and parent–child PDF chunking for academic document question answering."*

**Where it lives — every clause is real:**
| Clause | Proof in code |
|--------|---------------|
| FAISS (dense) | `dense_retriever.py`, `TOP_K_DENSE=20` (`config.py:44`) |
| BM25 (sparse) | `sparse_retriever.py`, `TOP_K_SPARSE=20` (`config.py:45`) |
| Reciprocal Rank Fusion | `hybrid_retriever.py:72-80`, `RRF_K=60` (`config.py:47`) |
| Cross-Encoder rerank | `hybrid_retriever.py:116-130` (`BAAI/bge-reranker-large`), `TOP_K_FINAL=6` |
| Parent–child chunking | `etl/chunker.py:45` (`parent_chunk_size=1500`), children link via `parent_id` UUID |
| Score filter (abstain) | `score_filter.py`, `MIN_RETRIEVAL_SCORE=0.35` (`config.py:48`) |

**Spoken defense (30s):**
> *"At query time I run FAISS and BM25 in parallel — top 20 each. I fuse them with Reciprocal Rank Fusion, k=60, which merges the two rankings without having to normalize incompatible score scales. Winning child chunks map to their parent paragraphs, I rerank those parents with the bge cross-encoder, and keep the top 6. Parent–child matters: children are small so search is precise; parents are ~1500 characters so the LLM gets full paragraph context instead of fragments."*

**Trap — "production-oriented" vs "in production":** Be precise. *"The **architecture** is production-shaped — hybrid retrieval, abstention, verification, tracing, offline eval. The **scale** is prototype: one paper, about 80 chunks. I never claim it's serving live traffic."*

---

### RESUME LINE 3
> *"Achieved 93.9% retrieval F1, 95.8% in-domain Hit@k, and 92% precision on a 30-query labeled golden evaluation set using RAGAS-based metrics."*

**Where it lives:**
| Item | Proof |
|------|-------|
| 30-query golden set | `evaluation/golden_set.json` (24 in-domain + 6 off-topic) |
| Metric computation | `evaluation/ragas_batch.py` (replays hybrid retrieval, computes P/R/F1/Hit@k) |
| Results | `evaluation/ragas_results.jsonl`, `ragas_results_full.jsonl` |

**Spoken defense (30s):**
> *"I built a 30-query golden set — 24 in-domain, 6 off-topic to test abstention. `ragas_batch.py` replays the exact hybrid retrieval offline and computes precision, recall, F1, and Hit@k against the labeled relevant sections. On the Transformer index I get retrieval F1 of 93.9%, in-domain Hit@k 95.8%, precision 92%, and a context-relevancy proxy of 93.8% using the reranker score."*

**CRITICAL TRAP — "RAGAS-based":**
> *"To be precise, these are **RAGAS-aligned** metrics — I compute the same dimensions RAGAS defines (context relevancy, context recall → Hit@k), but in my own batch script, not the official `ragas` pip package. I didn't want to claim a library I didn't run."*
This one sentence turns a potential "gotcha" into a **credibility win**.

**Trap — "is 30 queries enough?"** → *"Enough for a prototype regression smoke test, not for production SLAs. For a real claim I'd want 200+ stratified by query type — definitional, comparative, multi-hop, off-topic — with confidence intervals."*

---

### RESUME LINE 4
> *"Implemented LLM-as-judge verification (relevance and groundedness with retry) and parallel sub-query agent orchestration for multi-step research workflows, improving answer reliability and reducing hallucinations."*

**Where it lives:**
| Clause | Proof |
|--------|-------|
| Relevance judge | `gemini_verifier.py:167` `is_relevant()` — YES/NO, temp 0.1, cosine pre-filter |
| Groundedness judge | `gemini_verifier.py:199` `is_grounded()` |
| Retry | `MAX_RETRIES` regeneration loop in `document_chat.py`; backoff in `_generate_with_retry` |
| Parallel sub-query agents | `orchestration/agent.py` (`SubQueryAgent`), `planner.py`, `synthesizer.py`, `asyncio.gather` |
| Fail-closed | `gemini_verifier.py:196,215` — judge error → reject chunk, not pass |

**Spoken defense (30s):**
> *"Two judges. Before generation, a relevance judge grades each chunk YES/NO — this fixes garbage-in. After generation, a groundedness judge checks the answer is fully supported by the context — this fixes garbage-out. If groundedness fails, I regenerate up to three times with a stricter prompt. Judges run at temperature 0.1 and are **fail-closed**: if the API errors, I reject the chunk rather than risk a hallucination. For multi-step research, a planner splits the goal into up to four sub-queries and parallel worker agents each run their own retrieve-verify-answer loop, then a synthesizer merges the report."*

**Trap — "reducing hallucinations — did you measure the reduction?"** Be honest:
> *"I measured retrieval quality rigorously — F1 93.9%. The hallucination reduction is **architectural and design-level**, not yet a measured before/after number, because full faithfulness eval across all 30 queries was blocked by free-tier quota. The right way to prove it is an **ablation**: dense-only → hybrid → +rerank → +relevance judge → +groundedness, reporting F1 and pass rate at each layer. I have the harness to run it."*

---

### RESUME LINE 5
> *"Extended the system with CLIP-based multimodal retrieval, automated evaluation orchestration, retrieval-threshold optimization, monitoring, self-healing, and prompt A/B testing for continuous performance improvement."*

**Where it lives — all six are real modules:**
| Feature | File |
|---------|------|
| CLIP multimodal retrieval | `retrieval/multimodal_retriever.py` (text 0.6 / image 0.4 fusion, unified 512-dim) |
| Automated eval orchestration | `evaluation/eval_orchestrator.py` (Validator→Monitor→Decision→Executor) |
| Retrieval-threshold optimization | `optimization/threshold_optimizer.py` (grid search 0.25–0.50 on traces) |
| Monitoring | `evaluation/daily_eval.py` (pass rate, regen rate, empty retrieval, latency) |
| Self-healing | `observability/self_healer.py` (anomaly detect → auto-recover, token bucket) |
| Prompt A/B testing | `optimization/prompt_optimizer.py` (10% traffic, promote best, rollback) |

**Spoken defense (30s):**
> *"On top of the core RAG, I built a Level-5 continuous-improvement layer. CLIP fuses text and image queries into the same 512-dim vector space. An eval orchestrator decides between LLM judge and proxy judge based on quota, then runs the batch. A threshold optimizer grid-searches the cosine cutoff on logged traces and applies the best-F1 value. Daily eval monitors production traces. A self-healer detects anomalies — verification drop, latency spike, empty-retrieval spike — and auto-recovers, for example lowering the threshold or rolling back a prompt. And prompt A/B testing routes 10% of traffic to variants and promotes the winner."*

**BIGGEST TRAP of the whole resume — data-starved Level-5:**
These modules **exist and run**, but they optimize on **production traces**, and you only have a **handful of traces**. So:
> *"I want to be straight about this: these modules are **implemented and functional**, but they're **data-starved** today — the threshold optimizer and A/B test need volume of production traces to be statistically meaningful, and I have only a few live traces. So I'd describe them as **built and architected for closed-loop improvement**, validated on synthetic and small samples, not yet proven at production traffic. That honesty is deliberate."*

Say the word **"implemented"** (you wrote the code) not **"deployed at scale"** (you didn't).

---

## PART 2 — THE "OUTDATED / STALE DOCUMENTS" QUESTION FAMILY

This is your **#1 repeated question**. It appears as:
- *"How do you handle outdated files / stale documents in your RAG?"*
- *"A document changes — how does your index stay fresh?"*
- *"How do you re-index? Full rebuild or incremental?"*
- *"How do you avoid serving deleted/old content?"*
- *"What's your data freshness / cache invalidation strategy?"*

### First: know YOUR code's honest reality (do not bluff)

In `app.py → build_index_from_pdfs()` (line 46) the flow is:
`load PDFs → chunk → dense.build_index() → sparse.build_index() → save() to index_store/`.

This is a **full rebuild that overwrites `index_store/`**. There is **no** incremental upsert, no per-document versioning, no timestamp/TTL, no delete API, no dedup. So:

**Honest one-liner (memorize):**
> *"Today my freshness model is simple and honest: re-uploading a document triggers a **full re-index that overwrites the store**, so the latest version always wins — but I have **no incremental update, no per-document versioning, and no stale-document detection**. That's a deliberate prototype scope, and I know exactly how I'd productionize it."*

### Then: the production answer (this is where you score points)

Frame it as **four problems**, each with a concrete fix:

**1. Detecting staleness**
> *"Each document gets a **content hash and a `last_modified` / `ingested_at` timestamp** in chunk metadata. On re-ingest I compare hashes — unchanged docs are skipped, changed docs are re-embedded. That turns a full rebuild into an **incremental upsert**."*

**2. Removing old content (the dangerous one)**
> *"The subtle bug in naive RAG is **orphaned chunks** — you update a doc but old embeddings linger and get retrieved. So on update I **delete-then-insert by `doc_id`**: purge all vectors and BM25 postings for that doc_id, then add the new ones. FAISS `IndexFlatIP` doesn't delete in place, so in production I'd move to a store with native deletes — `IndexIDMap`, or pgvector/Qdrant/Weaviate — and delete by ID."*

**3. Serving freshness / cache invalidation**
> *"I already have a `query_cache.py` (Redis, TTL). The trap is a **stale cache after re-index**. Fix: **key the cache on an index version/epoch** — bump the epoch on every reindex so all old cached answers are invalidated automatically. Never cache the final unverified answer forever."*

**4. Freshness at retrieval / ranking time**
> *"For domains where recency matters, add a **`published_date` / `version` field** and either **filter** (only current version) or **boost** recent docs in ranking. For my academic-paper use case the document is static once uploaded, so recency ranking matters less than **version correctness** — the user must never get an answer from a superseded draft."*

**Close with the eval hook (senior move):**
> *"And critically — after any reindex I **re-run the golden set** (`ragas_batch.py`). A reindex that silently drops F1 is a regression, so freshness and evaluation are linked: I don't consider a document 'updated' until retrieval metrics still hold."*

### Related trap: "RAG vs fine-tuning for changing knowledge?"
> *"This is exactly why I chose RAG. When documents change — papers, policies, versions — RAG just re-indexes; fine-tuning would need a retrain and still can't cite a source. RAG when knowledge changes and you need attribution; fine-tune when behavior/style must change."*

---

## PART 3 — "EXPLAIN YOUR ARCHITECTURE" (3 timed scripts)

Pick the length based on how the interviewer asks.

### 3A — 30-second version (they ask casually)
> *"Three layers. **Knowledge**: PDF ingestion with parent-child chunking into FAISS plus BM25. **Retrieval**: hybrid dense-plus-sparse, RRF fusion, cross-encoder rerank, cosine abstention at 0.35. **Validation**: a relevance judge before generation and a groundedness judge after, with regeneration and full trace logging. The validation layer is what separates it from a tutorial RAG."*

### 3B — 90-second version (the default "explain your architecture")
> *"I think of it as a production-shaped RAG in three layers.*
>
> *Layer one, **Knowledge**: a PDF is chunked parent-child — small semantic children for search precision, ~1500-character parents for LLM context, linked by UUID. Children go into two indexes: FAISS for meaning, BM25 for exact tokens like 'self-attention' or 'BLEU'.*
>
> *Layer two, **Retrieval**: at query time I run both retrievers, top 20 each, fuse with Reciprocal Rank Fusion, map children to parents, rerank parents with the bge cross-encoder, keep top 6, and drop anything below cosine 0.35 so weak matches abstain instead of hallucinate.*
>
> *Layer three, **Validation**: before generation a relevance judge grades each chunk YES/NO; Gemini generates only from passed chunks; after generation a groundedness judge checks support and regenerates up to three times if it fails. Everything logs to JSONL with a trace ID, scores, and latency.*
>
> *Beyond chat, a planner plus parallel agents handle multi-step research, and there's a separate Deep-Read pipeline for reports and slides. And I close the loop with a 30-query golden set giving 93.9% retrieval F1, plus threshold optimization, monitoring, and self-healing."*

### 3C — Whiteboard while talking (draw this)
```
KNOWLEDGE   PDF → parent/child chunk → FAISS + BM25   (metadata: source, page, parent_id)
                                 │
RETRIEVAL   query → dense(20) + sparse(20) → RRF(k=60) → parents → rerank → top-6
                                 │  (cosine ≥ 0.35 else abstain)
VALIDATION  relevance judge ×6 → generate → groundedness judge → retry ≤3
                                 │
            citations + trace_id → rag_traces.jsonl → daily_eval / threshold_opt
```

**Design decisions to name-drop (shows you chose, not copied):**
- **Why RRF not score-averaging?** cosine and BM25 live on different scales; RRF fuses *ranks*, no normalization needed.
- **Why parent-child?** decouples search granularity from context granularity.
- **Why abstain at 0.35?** precision/recall tradeoff tuned on the golden set; technical queries drop to 0.25 via `get_effective_retrieval_threshold()`.
- **Why judge twice?** two independent failure modes — wrong chunk (relevance) vs invented answer (groundedness).
- **Why not LangGraph?** bounded custom asyncio orchestration for reliability — max 4 sub-queries, 1 rewrite, ≤3 regen, no infinite loops.

---

## PART 4 — "HOW DO YOU HANDLE RAG EVALUATION?" (one complete answer)

Structure the answer as **four levels** — this framing itself impresses.

> *"I evaluate at four levels.*
>
> ***1. Offline retrieval (the numbers I defend):*** a 30-query labeled golden set — 24 in-domain, 6 off-topic. `ragas_batch.py` replays the real hybrid retrieval and computes precision, recall, F1, and Hit@k against labeled relevant sections. Result: **retrieval F1 93.9%, in-domain Hit@k 95.8%, precision 92%.** These are RAGAS-**aligned** metrics I compute myself, not the official library.*
>
> ***2. Generation quality (relevance + groundedness):*** LLM-as-judge — context relevancy before generation, faithfulness after. When free-tier quota blocks the LLM judge, I fall back to a **proxy judge** (reranker score ≥ 0.5), which tracks relevance at 93.8% on the golden set. My `eval_orchestrator.py` auto-chooses LLM vs proxy based on quota, so eval never fails silently.*
>
> ***3. Production monitoring:*** `daily_eval.py` reads `rag_traces.jsonl` for verification pass rate, regeneration rate, empty-retrieval rate, and latency. Honest gap: only a few live traces so far.*
>
> ***4. Closed-loop tuning:*** `threshold_optimizer.py` grid-searches the cosine cutoff on logged traces and applies the best-F1 value; prompt A/B testing promotes better prompts; the self-healer rolls back on a verification drop.*
>
> *The mindset: retrieval and generation are **separate** eval problems. You can have perfect retrieval and a hallucinated answer — that's exactly why I judge after generation, not just before."*

**The killer follow-up you must be ready for — "prove you're better than vanilla RAG":**
> *"Ablation on the golden set: (1) dense-only, (2) hybrid no rerank, (3) hybrid + rerank, (4) + relevance judge, (5) + groundedness. Report F1, faithfulness proxy, and regeneration rate at each layer. That's evidence, not opinion — and I built the harness to run it."*

**"Retrieval F1 vs faithfulness — difference?"**
> *"Retrieval F1 = did we fetch the right text. Faithfulness = given that text, did the model stay honest. Different failure modes, different fixes: bad F1 → fix retrieval; bad faithfulness → fix the generation prompt or judge harder."*

---

## PART 5 — HONESTY CAVEATS (say these BEFORE they catch you)

Volunteering a limit builds more trust than any metric. Keep 3 ready:

| Caveat | How to say it |
|--------|---------------|
| Scale | *"Single paper, ~80 chunks. Metrics are strong in-domain; multi-document isn't tested yet."* |
| RAGAS wording | *"RAGAS-aligned metrics, computed by me — not the official `ragas` package."* |
| Faithfulness number | *"Full faithfulness eval across all 30 queries is quota-blocked; I report the proxy and don't invent the LLM-judge number."* |
| Level-5 modules | *"Threshold optimizer, self-healer, A/B are implemented but data-starved — few production traces, so not yet statistically validated."* |
| Freshness | *"Re-upload = full rebuild, latest wins; no incremental update or versioning yet — here's how I'd add it."* |
| Agent logging | *"Research/agent path doesn't write to `rag_traces.jsonl` yet; I'd unify observability."* |

**The framing that wins:** *"production-oriented **prototype**"* — architecture is production-shaped, scale is honest.

---

## PART 6 — REPEATED-QUESTION RAPID DRILL (cover the mouth, answer aloud)

| Repeated question | One-breath answer |
|-------------------|-------------------|
| How do you handle outdated/stale docs? | Today: full re-index overwrites, latest wins, no versioning. Prod: content-hash + timestamps → incremental upsert; delete-then-insert by doc_id to kill orphans; version-keyed cache; re-run golden set after reindex. |
| Explain your architecture. | 3 layers — Knowledge (parent-child → FAISS+BM25), Retrieval (RRF → rerank → top-6, abstain <0.35), Validation (relevance judge → generate → groundedness judge → retry). |
| How do you evaluate RAG? | 4 levels — offline golden set (F1 93.9%), LLM/proxy judge, production monitoring, closed-loop tuning. Retrieval and generation evaluated separately. |
| Why hybrid not just vector? | Dense = paraphrase, BM25 = exact tokens/acronyms; RRF fuses ranks without normalizing scales. |
| Why parent-child chunking? | Small children = search precision; big parents = LLM context. Decouple the two. |
| Why judge twice? | Relevance fixes garbage-in; groundedness fixes garbage-out. Two failure modes. |
| Is it really agentic? | Chat = fixed pipeline; Research = planner + parallel bounded worker agents. Word used precisely. |
| RAGAS or your own metrics? | RAGAS-aligned dimensions, computed in my own `ragas_batch.py`, not the pip package. |
| Biggest weakness? | Single-paper index; faithfulness quota-blocked; Level-5 modules data-starved. |
| Cost per chat? | ~$0.0007 on Flash paid; 8–12 API calls; retrieval/embeddings/rerank are local and free. |
| How would you scale to 1M docs? | Swap FAISS pickle for pgvector/Qdrant; shard by tenant; metadata pre-filter; async ingest queue; same retrieve→verify→generate logic. |
| Prompt injection via PDF? | Untrusted PDF = untrusted prompt; mitigate with groundedness judge, trust tiers, injection test cases in golden set. |

---

*Pair with `INTERVIEW_PREP.md` (15 Q&A + cost), `PROJECT_GUIDE.md` (architecture depth), `INTERVIEW_DELIVERY_SCRIPT.md` (timed delivery).*
