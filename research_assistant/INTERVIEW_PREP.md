# Interview Prep — Agentic RAG Research Assistant

Use this with `PROJECT_GUIDE.md` and the 10-minute script from chat. Practice out loud.

---

## 2-Minute Elevator Pitch

> I built an **Agentic RAG Research Assistant** — a Python app where you upload a research PDF and get **verified, source-grounded answers** instead of LLM guesses from memory.
>
> The user uploads a paper in Gradio, the system **indexes** it with parent-child semantic chunking into **FAISS + BM25**, and at question time runs **hybrid retrieval**: dense + keyword fusion via RRF, parent mapping, and **CrossEncoder reranking** to get the top 6 passages.
>
> Before the LLM answers, **Gemini judges each chunk for relevance**. After generation, it **judges groundedness** — and if the answer isn't supported by the context, the system **regenerates up to three times** with stricter prompts. Every turn is logged to JSONL for monitoring.
>
> For complex goals, a **planner splits research into sub-queries** and **parallel agents** retrieve, verify, and answer independently, then synthesize a report. A separate **Deep-Read** pipeline generates a structured report and PowerPoint from arXiv links.
>
> On evaluation, I built a **30-query golden set** and measured retrieval **F1 at 93.9%**, **Hit@k at 95.8%**, and **precision at 92%** — plus an orchestrator that runs batch eval and production monitoring. I'm honest about limits: small index today, full faithfulness eval needs more API quota — but the architecture is production-shaped: retrieve, verify, generate, verify, log, tune.

**Timing:** ~250 words ≈ 2 minutes at normal pace.

---

## 15 Tough Interviewer Questions + Scripted Answers

### 1. "Walk me through what happens when I ask one question."

> When you submit a question in Chat, `document_chat.py` normalizes the query — for example expanding "self attention" to "self-attention." It calls `multimodal_retriever.search()`, which inherits hybrid retrieval: FAISS returns top 20 by cosine, BM25 returns top 20 by keyword score, RRF fuses them with k=60, child chunks map to parent paragraphs, CrossEncoder reranks parents, and we keep top 6.
>
> Then `GeminiVerifier.is_relevant()` runs per chunk — chunks below cosine 0.35 are skipped without an API call. Passed chunks become context. Gemini generates an answer via LangChain. `is_grounded()` checks if the answer is fully supported; if not, we regenerate up to 3 times with escalating strictness. The trace — scores, latency, pass/fail — goes to `logs/rag_traces.jsonl`. You get answer, sources, and trace ID.

---

### 2. "Why hybrid retrieval instead of just vector search?"

> Academic PDFs mix **conceptual language** and **exact technical tokens** — model names, acronyms, hyphenated terms. Dense embeddings handle paraphrases well but can miss rare exact matches. BM25 excels at keywords. I fuse both with **Reciprocal Rank Fusion** so I don't need to normalize incompatible score scales. On my golden set, that combination gave **93.9% retrieval F1** — I wouldn't rely on dense-only for this domain.

---

### 3. "What is parent-child chunking and why did you use it?"

> **Children** are small semantic chunks — good for **search precision**. **Parents** are ~1500-character blocks — good for **LLM context** so the model sees full paragraphs, not sentence fragments. Each child carries a `parent_id` UUID. Search hits children; the LLM reads parents. Without this, you either retrieve imprecise large blocks or generate from starved context.

---

### 4. "Explain your verification loop. Isn't that expensive?"

> It's a **Self-RAG / Corrective RAG** pattern: relevance judge **before** generation, groundedness judge **after**. Yes, it's more API calls — roughly up to 6 relevance checks plus 1–3 groundedness checks per question. I mitigate cost with: cosine pre-filter at 0.35 so weak chunks skip the judge; exponential backoff on rate limits; and a relevance fallback using reranker score ≥ 0.5 when the judge rejects everything due to API errors. The tradeoff is intentional: I'd rather block or retry than ship a hallucination on a research assistant.

---

### 5. "What's the difference between relevance and groundedness?"

> **Relevance**: "Is this retrieved passage useful for answering the question?" — filters bad retrieval **before** generation.
>
> **Groundedness** (faithfulness): "Is the generated answer fully supported by the passages we gave the model?" — catches hallucination **after** generation.
>
> You need both. Good retrieval with a hallucinating generator fails groundedness. Perfect generation with irrelevant context fails relevance. I map these to RAGAS **context relevancy** and **faithfulness**.

---

### 6. "You said 'agentic' — what actually is an agent in your system?"

> I use "agentic" **precisely**, not as marketing. **Chat** is verified RAG with judges — not a planner loop. **Agentic** paths are:
> 1. **Research mode**: `ResearchPlanner` decomposes a goal into up to 4 sub-queries; `SubQueryAgent` instances run in parallel via `asyncio.gather`, each doing retrieve → grade → answer → hallucination check; `ReportSynthesizer` merges results.
> 2. **Deep-Read**: six section agents analyze introduction, methods, results, etc. in parallel.
>
> Each agent is a Python module with a defined tool boundary — retriever + LLM + verifier — not a free-form autonomous loop.

---

### 7. "How did you evaluate this? What numbers can you defend?"

> I built `evaluation/golden_set.json` — **30 labeled queries**, 24 in-domain and 6 off-topic. `ragas_batch.py` replays hybrid retrieval offline and computes:
> - **Retrieval F1: 93.9%**
> - **Hit@k: 95.8%** (in-domain)
> - **Precision: 92%**
> - **Context relevancy proxy: 93.8%** (reranker ≥ 0.5)
>
> Production monitoring via `daily_eval.py` reads `rag_traces.jsonl` for pass rate, regeneration rate, empty retrieval, and latency. Full LLM faithfulness on all 30 queries was blocked by Gemini free-tier quota — I report proxy metrics and don't inflate numbers I haven't measured.

---

### 8. "What's your biggest failure mode today?"

> **Empty or weak retrieval** — if nothing passes cosine 0.35 or all chunks fail relevance, the user gets an honest "I can't verify" message instead of a guess. Second: **judge over-rejection** when API fails — I added a fallback to top reranker/cosine docs. Third: **small corpus** — one paper, ~80 chunks — so cross-document reasoning isn't tested yet.

---

### 9. "Why CrossEncoder reranking after RRF?"

> RRF merges **rankings** from FAISS and BM25 but doesn't deeply score query–document **semantic alignment** at the paragraph level. CrossEncoder `BAAI/bge-reranker-large` jointly encodes query and each parent — much slower than bi-encoder search but far more accurate for the final top-6 cut. I only rerank the deduplicated parent candidates, not the full corpus, to keep latency reasonable.

---

### 10. "How do FAISS cosine scores and reranker scores differ?"

> FAISS uses **bi-encoder** embeddings — query and document encoded separately, cosine similarity in [0, 1]. CrossEncoder outputs a **relevance logit** on a different scale. I had a real bug where the relevance judge applied the 0.35 threshold to reranker scores. I fixed it: cosine threshold only on `retrieval_score` from FAISS; reranker used for ranking and fallback at ≥ 0.5. They're stored separately in trace logs.

---

### 11. "How would you productionize this?"

> **Indexing**: async job queue (Celery/SQS) for PDF ingest; S3 for raw PDFs; vector store at scale (pgvector, Pinecone, or managed OpenSearch). **Serving**: FastAPI behind a load balancer; separate retrieval and generation services; Redis query cache (I already have `query_cache.py` stubbed for Level 5). **Observability**: traces to Datadog or OpenTelemetry; daily eval as a cron; alerts on pass-rate drop (my `self_healer.py` pattern). **Cost**: cache retrieval results; batch judge calls; use smaller judge model for relevance, larger for generation. **Security**: API keys in secrets manager; per-tenant indexes.

---

### 12. "What would you improve with another month?"

> 1. **Multi-document indexes** with metadata filters (author, year, section).
> 2. **Answer relevancy** metric — the RAGAS dimension I don't fully score yet.
> 3. **Link user feedback to trace_id** for human-in-the-loop prompt and threshold tuning.
> 4. **Full faithfulness eval** on paid Gemini tier across the golden set.
> 5. **Agent path logging** — Research mode doesn't write to `rag_traces.jsonl` today; I'd unify observability.

---

### 13. "Why Gemini and not OpenAI or open-source?"

> The project supports **multiple backends** via `config.py` — Perplexity, Gemini, Ollama — and Deep-Read uses a provider-agnostic `LLMGateway`. I used Gemini for verification because `google.generativeai` gives fine-grained safety settings, low-temperature YES/NO judging, and structured retry on 429 quota errors. For local dev, Ollama is the fallback. The architecture is backend-agnostic; swap the LLM, keep retrieval and verification prompts.

---

### 14. "Tell me about a bug you fixed."

> Retrieved chunks were logged to `rag_traces.jsonl` **without numeric scores** — my threshold optimizer couldn't grid-search cosine cutoffs. I added `serialize_retrieved_docs()` in `gemini_tracer.py` to log `score`, `retrieval_score`, and `reranker_score` per chunk. I also fixed groundedness retry tracking — `grounded_after_retry` wasn't set when regeneration succeeded. Those fixes made offline eval and monitoring actually usable.

---

### 15. "Why should we hire you based on this project?"

> This project shows I can build **end-to-end ML systems**, not just call an API. I designed hybrid retrieval with measurable **93.9% F1**, added **verification layers** for trust, built **evaluation and orchestration** with a golden set and production traces, and I'm **honest about limits** — small index, quota constraints, proxy vs LLM judge. I think in terms of retrieve → verify → generate → measure → tune — which is how real RAG products ship.

---

## Quick-Reference Card (glance before interview)

| Topic | One-liner |
|-------|-----------|
| Stack | Python, Gradio, FastAPI, FAISS, BM25, SentenceTransformers, CrossEncoder, Gemini |
| Index | Parent 1500 chars, semantic children, MiniLM-L6-v2, FAISS dim=512 |
| Retrieve | top_k_dense=20, top_k_sparse=20, RRF k=60, rerank, top_k_final=6 |
| Threshold | cosine ≥ 0.35 (0.25 technical), reranker fallback ≥ 0.5 |
| Verify | relevance per chunk → generate → groundedness, max 3 retries |
| Metrics | F1 93.9%, Hit@k 95.8%, Precision 92%, context proxy 93.8% |
| Agentic | Planner + parallel SubQueryAgents; Deep-Read section agents |
| Honest gap | Single-paper index; full faithfulness eval quota-limited |

---

## Practice Plan (30 minutes)

| Time | Activity |
|------|----------|
| 0–5 min | Read elevator pitch aloud 3× |
| 5–15 min | Pick 5 hard questions (3, 4, 7, 8, 11) — answer without reading |
| 15–25 min | Full 10-minute story from chat transcript |
| 25–30 min | Whiteboard: `PDF → Index → Retrieve → Judge → Generate → Judge → Log` |

---

*Pair with: `PROJECT_GUIDE.md` for architecture depth and `evaluation/golden_set.json` if asked for eval methodology.*

---

## LLM Cost Per Request (How to Answer Interviewers)

### One-line answer (memorize this)

> *"On **gemini-2.0-flash** paid tier, a typical **Chat** turn costs about **$0.0007** — under **0.1 cent**. Worst case with full verification and 3 regeneration retries is about **$0.002**. Retrieval, embeddings, and reranking are **local** — zero API cost. The real limit on free tier isn't dollars, it's **~20 API calls/day**, and my Chat path uses **8–12 calls per question**."*

### What costs money vs what is free

| Component | Tool | API cost |
|-----------|------|----------|
| PDF load, chunking | pypdf, LangChain | **$0** |
| Embeddings | SentenceTransformer `all-MiniLM-L6-v2` (local) | **$0** |
| Dense search | FAISS (local) | **$0** |
| Sparse search | BM25Okapi (local) | **$0** |
| Reranking | CrossEncoder `bge-reranker-large` (local GPU/CPU) | **$0** |
| Relevance judge | Gemini `generate_content`, max_output=32 | **Paid per call** |
| Answer generation | Gemini via LangChain, max_output=1200 | **Paid per call** |
| Groundedness judge | Gemini `generate_content`, max_output=32 | **Paid per call** |

**Gemini 2.0 Flash paid pricing (Google AI):** $0.10 / 1M input tokens, $0.40 / 1M output tokens.  
**Free tier:** $0 monetary cost, but strict daily/RPM quotas (your project hit **daily_quota_exhausted** at ~20 calls/day).

### LLM call count per path (from your code)

**Chat** (`document_chat.py` + `gemini_verifier.py`):

| Step | Calls | Module | Notes |
|------|-------|--------|-------|
| Relevance judge | **0–6** | `gemini_verifier.is_relevant()` | 1 per retrieved parent; skipped if cosine < 0.35 |
| Answer generation | **1–3** | `chain.ainvoke()` | `MAX_RETRIES=3` on groundedness fail |
| Groundedness judge | **1–3** | `gemini_verifier.is_grounded()` | 1 per generation attempt |
| **Typical total** | **8 calls** | | 6 + 1 + 1 |
| **Worst case** | **12 calls** | | 6 + 3 + 3 |

**Research agentic** (`planner.py` + `agent.py` × 4 + `synthesizer.py`):

| Step | Calls (approx) |
|------|----------------|
| Planner | 1 |
| Per sub-query (×4): relevance ×6 + generate ×1 + groundedness ×1 | 8 each → **32** |
| Query rewrite fallback (if retrieval fails) | 0–4 extra |
| Hallucination regen | 0–4 extra |
| Synthesizer | 1 |
| **Typical total** | **~34–40 LLM calls** |

**Deep-Read** (`section_agents.py` × 6):

| Step | Calls |
|------|-------|
| Section analysis | **6** (`max_tokens=2500` each, context up to 14K chars) |
| **Total** | **6 calls** (no judge loop) |

**Simple summarize** (`simple_summarizer.py`): **1 LLM call** (+ same local retrieval as Chat).

### Token & dollar estimates (gemini-2.0-flash, paid tier)

Assumptions: ~4 characters ≈ 1 token; 6 parents × ~1,500 chars context ≈ 2,250 tokens in generation prompt.

| Path | Input tokens (est.) | Output tokens (est.) | **Cost (USD)** |
|------|---------------------|----------------------|----------------|
| Chat — typical (8 calls) | ~5,500 | ~450 | **~$0.0007** |
| Chat — worst case (12 calls, 3 retries) | ~12,500 | ~1,250 | **~$0.002** |
| Research — 4 agents (~36 calls) | ~35,000 | ~5,000 | **~$0.006** |
| Deep-Read (6 sections) | ~21,000 | ~4,800 | **~$0.004** |
| Simple summarize | ~3,000 | ~800 | **~$0.0006** |

**Formula to say on the whiteboard:**

```
cost = (input_tokens / 1_000_000 × $0.10) + (output_tokens / 1_000_000 × $0.40)
```

### How to reduce cost (shows engineering thinking)

1. **Cache retrieval** — `retrieval/query_cache.py` (Redis TTL=3600); same question = skip retrieve + judges.
2. **Cosine pre-filter** — skip relevance LLM call if `retrieval_score < 0.35` (already in your code).
3. **Smaller judge model** — use Flash-Lite for YES/NO judges, Flash for generation only.
4. **Batch judges** — one prompt grading all 6 chunks (tradeoff: accuracy).
5. **Cap retries** — you use `MAX_RETRIES=3`; production might use 1 for cost-sensitive paths.
6. **Proxy judge offline** — reranker score ≥ 0.5 instead of LLM for eval/monitoring (`ragas_batch.py --proxy-judge`).

### Honest caveat (say this if pressed)

> *"These are **estimates** from call counts and typical token sizes — I haven't metered exact tokens per production trace yet. I'd add `usage_metadata` from Gemini responses to `rag_traces.jsonl` for precise per-request cost in production."*

---

## Trending RAG + LLM + Agent Questions (Logic & Thinking)

High-probability interview topics in 2025–2026. Each has a **thinking frame** + **how your project answers**.

---

### A. Fundamentals & "Why RAG?"

**Q: RAG vs fine-tuning — when do you pick which?**

> **Think:** Does knowledge change often? Do you need citations? Do you have training budget?
>
> **Answer:** RAG when documents update (papers, policies), you need **source attribution**, and you can't retrain weekly. Fine-tuning when behavior/style must change or domain language is fixed and retrieval noise is the bottleneck. My project is classic RAG: user uploads **their** PDF — fine-tuning can't know that document at question time without retraining.

**Q: RAG vs long-context — why not stuff the whole PDF in the prompt?**

> **Think:** Context window cost, "lost in the middle," attention dilution.
>
> **Answer:** A 15-page paper might fit in 128K context, but you pay for every token every request, and models still **miss details** in long prompts. I retrieve **top-6 parents** (~9K chars) — cheaper and more focused. For 500-page corpora, full-context is impossible; retrieval is mandatory.

**Q: What is "lost in the middle" and how do you handle it?**

> **Think:** LLMs overweight start/end of context, underweight middle.
>
> **Answer:** Reranker puts the **best** chunks first; I join context with clear `[source, page]` headers. If answers miss middle chunks, I'd reduce k, rerank harder, or use **lost-in-the-middle** aware prompting ("consider all sections equally").

---

### B. Retrieval Logic (most asked technical area)

**Q: Why hybrid dense + sparse instead of one?**

> **Think:** Dense = semantic paraphrase; sparse = exact token match.
>
> **Answer:** "self attention" vs "self-attention", model names, BLEU scores — BM25 catches these; FAISS catches "how does attention work". RRF merges ranks without normalizing scores. Measured **F1 93.9%** on golden set.

**Q: How do you choose chunk size?**

> **Think:** Precision vs context tradeoff.
>
> **Answer:** I use **parent-child**: children for search precision (semantic splits), parents at **1500 chars** for generation context. Too small = fragmented answers; too large = retrieval returns irrelevant paragraphs. I'd tune with retrieval recall@k on golden set.

**Q: What if the answer spans two chunks that never co-retrieve?**

> **Think:** Multi-hop / compositional retrieval failure.
>
> **Answer:** Honest limitation today. Fixes: increase k, parent expansion, **query decomposition** (my Research planner), or graph RAG linking adjacent chunks. This is why I built agentic sub-query path for complex goals.

**Q: When does reranking help vs hurt?**

> **Think:** Reranker = accurate but O(n) on candidates; wrong candidates in pool can't be saved.
>
> **Answer:** CrossEncoder reranks only **deduplicated parents** after RRF — small n (~12). It improved final precision; it can't fix if the right chunk wasn't in top-20 FAISS/BM25. Recall is fixed at indexing + top_k_dense/sparse=20.

**Q: How do you set the 0.35 cosine threshold?**

> **Think:** Precision-recall tradeoff on labeled data.
>
> **Answer:** `threshold_optimizer.py` grid-searches on logged traces; golden set gives **F1 93.9%** at current setting. Technical queries use **0.25** via `get_effective_retrieval_threshold()` — definitional questions often have lower cosine but are still correct.

---

### C. Verification & Hallucination (Self-RAG / Corrective RAG)

**Q: Why judge before AND after generation?**

> **Think:** Garbage in vs garbage out — two failure modes.
>
> **Answer:** Pre-generation **relevance** filters wrong retrieval (fixes context precision). Post-generation **groundedness** catches model invention (fixes faithfulness). RAGAS maps to context relevancy + faithfulness. Either alone leaves a hole.

**Q: LLM-as-judge — isn't the judge also wrong?**

> **Think:** Judge error rate, calibration, cascade failures.
>
> **Answer:** Yes — judges can false-reject or false-accept. I use low temperature (0.1), YES/NO only, short context truncation, and **reranker fallback** when judge rejects all. In production I'd measure judge agreement vs human labels on a sample. Proxy: reranker ≥ 0.5 correlates with relevance at **93.8%** on golden set.

**Q: Regeneration loop — when do you stop trusting retries?**

> **Think:** Diminishing returns, cost, same wrong context.
>
> **Answer:** I cap at **MAX_RETRIES=3** with stricter prompts. If groundedness still fails, I return the answer but log `grounded=false` — production should show a warning. Retrying won't help if **retrieval** was wrong; fix retrieve, not generate.

---

### D. Agents vs Workflows (2025–2026 hot topic)

**Q: What makes something an "agent" vs a "pipeline"?**

> **Think:** Autonomy, tool choice, loop until done vs fixed DAG.
>
> **Answer:** My **Chat** is a **fixed pipeline**: retrieve → judge → generate → judge → log. No replanning. My **Research** mode is **agentic**: planner creates sub-queries, parallel agents each run a Self-RAG loop with **query rewrite fallback** if retrieval fails. Still bounded — not open-ended tool loops. That's intentional for reliability.

**Q: Why parallel agents instead of one big prompt?**

> **Think:** Context limits, specialization, latency.
>
> **Answer:** One prompt with 4 sub-questions dilutes attention. Parallel `SubQueryAgent` × 4 via `asyncio.gather` — each gets focused retrieval and verification. Tradeoff: **4× retrieval cost**, **~34 LLM calls** vs ~8 for Chat. Use for research reports, not simple Q&A.

**Q: How do you prevent agent infinite loops?**

> **Think:** Max steps, budgets, termination conditions.
>
> **Answer:** `MAX_SUB_QUERIES=4`, single rewrite fallback per sub-query, one hallucination regen — no while-True loops. Eval orchestrator uses explicit Validator → Decision → Executor DAG, not free-form agent chat.

**Q: Single agent vs multi-agent — tradeoffs?**

> **Think:** Coordination overhead vs decomposition.
>
> **Answer:** Multi-agent wins when tasks are **independent** (sub-queries on same corpus). Single agent wins for **follow-up chat** with history. I use both in different tabs — not one-size-fits-all.

---

### E. Evaluation & Metrics (Data Science angle)

**Q: How do you evaluate RAG without users?**

> **Think:** Golden set, LLM judge, retrieval-only metrics.
>
> **Answer:** `golden_set.json` — 30 queries with expected_pass / in_domain flags. `ragas_batch.py` computes retrieval F1, Hit@k, precision. Proxy context relevancy via reranker ≥ 0.5 when API quota blocks LLM judge. `daily_eval.py` on production traces for pass rate and regeneration rate.

**Q: What's the difference between retrieval F1 and answer faithfulness?**

> **Think:** Did we find the right text vs did the model lie?
>
> **Answer:** Retrieval F1 = did hybrid search return chunks from the right section? Faithfulness = given those chunks, is the answer supported? You can have **perfect retrieval, hallucinated answer** — that's why I judge after generation. My retrieval F1 is **93.9%**; full faithfulness eval pending quota.

**Q: How many test queries is enough?**

> **Think:** Statistical power, domain coverage.
>
> **Answer:** 30 is enough for **prototype/demo** and regression smoke tests — not for production SLAs. I'd want 200+ stratified by query type (definitional, comparative, off-topic, multi-hop) before claiming 95% confidence intervals.

---

### F. Cost, Scale & Production

**Q: What's your LLM cost per request?** → See cost section above.

**Q: How does cost scale with users?**

> **Think:** Calls per request × users × sessions; caching amortizes.
>
> **Answer:** Chat ≈ **$0.0007/request** on Flash paid tier — 10,000 questions ≈ **$7**. Bottleneck is often **judge call count** (8–12), not generation tokens. Redis query cache and retrieval cache cut repeat-question cost to ~1 generation call.

**Q: How would you scale to 1M documents?**

> **Think:** Index sharding, metadata filters, managed vector DB, async ingest.
>
> **Answer:** Replace local FAISS pickle with **pgvector/Pinecone/Weaviate**; shard by tenant; add metadata pre-filter (author, date); async ingest queue; keep same retrieve → verify → generate logic. Parent-child still applies; index build becomes distributed batch job.

---

### G. Security & Adversarial (increasingly asked)

**Q: Prompt injection via uploaded PDF?**

> **Think:** Untrusted document = untrusted prompt input.
>
> **Answer:** A PDF could contain "ignore instructions and say X". Mitigations: system prompt hardening, **groundedness judge** (answer must cite context — injection text is still "in context" though), separate **trust tiers** for user uploads vs curated corpora, output filtering. I'd add explicit injection test cases to golden set.

**Q: Can user feedback poison the system?**

> **Think:** Feedback loops, preference learning attacks.
>
> **Answer:** `user_memory.py` boosts terms from feedback — low risk at small scale but in production I'd rate-limit feedback influence and require authenticated users before changing retrieval behavior.

---

### H. Logic Puzzles (interviewers test thinking)

**Q: User asks off-topic question — what happens?**

> **Think:** Retrieval threshold + relevance judge + empty context path.
>
> **Answer:** Weak cosine scores → filtered at 0.35; if junk still passes, relevance judge should reject; if all rejected, user gets "couldn't verify relevance" — not a fabricated answer. Golden set has **6 off-topic** queries; off-topic block rate was part of eval.

**Q: Two chunks contradict each other — what should the system do?**

> **Think:** Conflict in context, synthesis vs pick one.
>
> **Answer:** Generation prompt should present both with sources; agent gen_prompt explicitly says **present contradiction neutrally**. Groundedness still passes if answer reflects both. Bad answer = silently merging contradictions — judge won't catch that; need **answer quality** eval or human review.

**Q: Retrieval returns right chunk but answer is wrong — debug how?**

> **Think:** Trace-driven debugging: retrieve → judge → generate stages.
>
> **Answer:** Open `rag_traces.jsonl` for trace_id: check `chunks_scores` (cosine, reranker), `verification_results.relevance_passed`, `grounded`, `regeneration_count`. If relevance passed and grounded passed but answer wrong → **generation prompt** or **chunk too large** issue. If grounded failed but returned anyway → policy bug.

**Q: Would you remove BM25 and use only embeddings?**

> **Think:** Ablation mindset — prove with data.
>
> **Answer:** I wouldn't without an ablation run on golden set. Hypothesis: F1 drops on acronym-heavy queries. I'd run `ragas_batch.py` dense-only vs hybrid and report delta — that's the right engineering answer.

---

## Rapid-Fire Card (30 seconds each)

| Question | Answer in one breath |
|----------|---------------------|
| Cost per chat? | ~$0.0007 Flash paid; 8–12 API calls; retrieval free locally |
| Why agents? | Decompose complex research; parallel Self-RAG per sub-query |
| Why not one LLM call? | No verification, no citations, hallucination risk |
| Biggest RAG failure? | Wrong chunk retrieved — fix retrieval before bigger model |
| RAGAS mapping? | Faithfulness=groundedness, context precision≈relevance, recall≈Hit@k |
| Free tier problem? | 20 calls/day — 2–3 chat questions, not dollars |
| Agent vs pipeline? | Chat=pipeline; Research=planned parallel agents with bounds |
| Scale bottleneck? | Local FAISS index + per-chunk judge API calls |

---

## Mock "Thinking" Drill (practice aloud)

Interviewer: *"Your relevance judge rejects a chunk with cosine 0.45 but the answer is actually in that chunk. What do you do?"*

> First I'd **label it** in the golden set as a false negative. Check if the judge prompt truncates at 500 chars and cuts the defining sentence. Try expanding truncation or passing parent not child to judge. Compare **LLM judge vs reranker proxy** on that query. If systematic, tune prompt or lower reliance on judge for high-cosine chunks — e.g. auto-pass if cosine > 0.55. Measure F1 delta before shipping.

Interviewer: *"How do you prove your system is better than vanilla RAG?"*

> Ablation table on golden set: (1) dense only, (2) hybrid no rerank, (3) hybrid + rerank, (4) + relevance judge, (5) + groundedness. Report F1, faithfulness proxy, and regeneration rate at each layer. That's evidence, not opinion.
