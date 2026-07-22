# Interview Delivery Script — Agentic RAG Research Assistant

**Goal:** Sound like a **systems thinker**, not a tutorial follower.  
**Total time:** 8–12 minutes core + 3–5 minutes for follow-ups.  
**Rule:** Only say numbers you measured. Say "retrieval F1," not "accuracy."

---

## BEFORE YOU SPEAK (30 seconds mental checklist)

- [ ] I built **3 layers**: Knowledge → Retrieval → Validation
- [ ] My headline metric: **93.9% retrieval F1** on 30-query golden set (one paper corpus)
- [ ] My differentiator: **LLM-as-judge** before AND after generation
- [ ] I did **not** use LangGraph — custom asyncio + LangChain
- [ ] Honest gap: faithfulness % not fully measured (API quota); 2 production traces

---

## OPENING HOOK (20–30 seconds) — memorize verbatim

> *"Most RAG demos stop at vector search — upload a PDF, embed it, ask a question, hope the LLM doesn't hallucinate. I built the opposite: an **enterprise-shaped pipeline** where every answer has to pass **retrieval, relevance verification, generation, and groundedness verification** — with citations, trace IDs, and offline evaluation on a labeled golden set. It's called an Agentic RAG Research Assistant, and I'll walk you through the architecture in under ten minutes."*

**Why this works:** You immediately signal validation + production thinking, not "I used FAISS."

---

## 2-MINUTE VERSION (if they say "give me the overview")

> *"Users upload an academic PDF — I tested on the Transformer paper. The system chunks it parent-child: small semantic children for search, large parents for LLM context. It builds **dual indexes** — FAISS for meaning, BM25 for exact terms — fuses them with RRF, reranks with a CrossEncoder, and keeps the top six passages.*
>
> *For chat, before Gemini generates anything, a **relevance judge** filters each chunk. After generation, a **groundedness judge** checks the answer is supported — with up to three retries and stricter prompts each time. Everything logs to JSONL for monitoring.*
>
> *For complex research goals, a **planner** splits the goal into sub-queries and **parallel agents** each run their own retrieve-verify-answer loop, then a synthesizer merges a report. There's also a separate Deep-Read pipeline for structured reports and PowerPoint from arXiv links.*
>
> *On evaluation, I built a 30-query golden set and measured **93.9% retrieval F1**, **95.8% in-domain Hit@k**, and **92% precision**. The validation layer is designed like enterprise RAG — the index is still prototype-scale, one paper, but the architecture is production-oriented."*

---

## 10-MINUTE FULL DELIVERY (timed script)

### MINUTE 0–1 — Problem & why it matters

> *"The problem is trust. If you're reading a research paper and you ask an LLM a technical question, you get fluent answers that might be wrong — with no source, no score, no audit trail.*
>
> *I wanted NotebookLM-style Q&A, but with **measurable retrieval quality** and **verification gates** — so the system can say 'I don't know' instead of guessing. That's the product goal."*

**Pause.** Let them nod or ask a question.

---

### MINUTE 1–2 — Enterprise 3-layer framing (senior move)

> *"I think about it in three layers — this is how I'd explain it in a production design review.*
>
> ***Layer 1 — Knowledge:*** PDF ingestion, parent-child chunking, metadata — source, page, parent_id — persisted to disk.*
>
> ***Layer 2 — Retrieval:*** hybrid FAISS plus BM25, RRF fusion, CrossEncoder rerank, cosine thresholding. Not just 'we have a vector DB.'*
>
> ***Layer 3 — Validation:*** relevance judge per chunk, groundedness judge on the answer, regeneration, citations, trace logging, user feedback. **This layer is what separates enterprise RAG from a tutorial.***
>
> *Everything runs in a Gradio app on port 7860 with a FastAPI feedback endpoint — Python, LangChain, Gemini, Sentence Transformers, FAISS."*

---

### MINUTE 2–4 — Walk the indexing path (whiteboard this)

> *"Let me walk indexing first — nothing works without a good knowledge layer.*
>
> *`pdf_loader.py` extracts text page by page. `chunker.py` creates **parents** at fifteen hundred characters and **semantic children** using MiniLM embeddings — children link to parents via UUID.*
>
> *Children go into two indexes in parallel: **FAISS** with IndexFlatIP — cosine via inner product on L2-normalized vectors — and **BM25** for keyword matches like 'self-attention' and BLEU scores.*
>
> *At query time I don't pick one. I run both — top twenty from each — **RRF fuse** with k equals sixty, map winning children to **parent paragraphs**, rerank parents with **bge-reranker-large**, return top six.*
>
> *Why parent-child? Search needs precision on small chunks; the LLM needs paragraph context. That split is deliberate."*

**Whiteboard:** `PDF → chunk → FAISS + BM25 → RRF → parents → rerank → top-6`

---

### MINUTE 4–6 — Walk the chat path (your strongest section)

> *"Chat is a fixed pipeline — I call it Self-RAG or Corrective RAG, not a free-form agent loop.*
>
> *Step one: normalize the query — 'self attention' becomes 'self-attention.' Optional user memory boosts repeated terms.*
>
> *Step two: hybrid retrieve six parents.*
>
> *Step three: **relevance judge** — Gemini YES/NO per chunk. Chunks below cosine zero-point-three-five skip the API call entirely. That's a latency and cost lever.*
>
> *Step four: assemble context with source and page headers.*
>
> *Step five: Gemini generates — temperature zero-point-seven for generation, zero-point-one for judges.*
>
> *Step six: **groundedness judge**. If it fails, regenerate up to three times with escalating strictness and exponential backoff.*
>
> *Step seven: log to `rag_traces.jsonl` — trace ID, chunk scores, verification pass/fail, latency. User can thumbs up/down, which feeds user memory.*
>
> *If retrieval is weak or nothing passes verification, the system **refuses** — it doesn't hallucinate through."*

**Impressive line:**
> *"I separate FAISS cosine scores from CrossEncoder logits — they live on different scales. I learned that the hard way when the threshold optimizer broke."*

---

### MINUTE 6–7 — Agentic path (only when goal is complex)

> *"I use 'agentic' precisely. Chat is a pipeline. **Agentic** is the Research tab:*
>
> *`ResearchPlanner` uses structured LLM output to split a goal into up to four sub-queries. `SubQueryAgent` workers run in **parallel** via asyncio — each does retrieve, relevance grade, answer, hallucination check, with query rewrite fallback if retrieval fails. `ReportSynthesizer` merges into one report.*
>
> *This is **not LangGraph** — it's bounded custom orchestration. I chose that for reliability: max four sub-queries, one rewrite, one regen — no infinite loops.*
>
> *There's also **simple summarize** — one retrieval, one LLM call — for when quota or latency matters. Two speeds, same index."*

---

### MINUTE 7–8 — Evaluation & metrics (data science credibility)

> *"I don't claim vibes — I built evaluation infrastructure.*
>
> *`golden_set.json` — thirty labeled queries, twenty-four in-domain, six off-topic. `ragas_batch.py` replays hybrid retrieval and computes precision, recall, F1, Hit@k, off-topic block rate.*
>
> *Results on the Transformer index: **retrieval F1 ninety-three point nine percent**, **in-domain Hit@k ninety-five point eight**, **precision ninety-two percent**. Context relevancy proxy at reranker threshold: **ninety-three point eight**.*
>
> *`eval_orchestrator.py` runs Validator, Monitor, Decision, Executor — if Gemini quota is exhausted, it auto-falls back to proxy-judge mode instead of failing silently.*
>
> *`daily_eval.py` reads production traces for pass rate, regeneration rate, empty retrieval. Honest gap: only two live traces so far — faithfulness on all thirty queries blocked by free-tier quota. I report what I measured."*

---

### MINUTE 8–9 — Senior trade-offs (this impresses staff engineers)

> *"Three trade-offs I'd highlight.*
>
> ***Latency vs accuracy:*** full chat uses eight to twelve Gemini calls per question — I saw seventeen to forty seconds end-to-end in traces. I prioritized validation over speed. In production I'd add **conditional reranking** — skip CrossEncoder when FAISS and BM25 agree — and batch judges.*
>
> ***Latency vs cost:*** judges burn free-tier quota — twenty calls a day. That's why simple summarize exists and why the orchestrator has proxy mode.*
>
> ***Accuracy vs corpus size:*** eighty chunks, one paper. Retrieval metrics are strong in-domain; multi-document and ACL filtering would be the next enterprise layer.*
>
> *Level Five hooks exist — threshold grid search, prompt A/B, self-healer, Redis query cache — architecture for closed-loop improvement."*

---

### MINUTE 9–10 — Close strong

> *"To summarize: I built end-to-end RAG with hybrid retrieval, parent-child chunking, LLM-as-judge verification, parallel agents for research, Deep-Read for reports, and a golden-set evaluation harness with defensible retrieval metrics.*
>
> *If I had another month: conditional reranking for latency, faithfulness eval on paid API tier, link feedback to trace IDs for human-in-the-loop tuning, and managed vector DB for multi-tenant scale.*
>
> *I'm happy to go deeper on retrieval fusion, the verification loop, the eight-stage latency breakdown, or how I'd productionize this on AWS. What would be most useful?"*

**End with a question** — shows confidence.

---

## WHITEBOARD SEQUENCE (draw while talking — 90 seconds)

```
┌──────────── KNOWLEDGE LAYER ────────────┐
│ PDF → parent/child chunk → FAISS+BM25  │
│ metadata: source, page, parent_id       │
└──────────────────┬────────────────────┘
                   ▼
┌──────────── RETRIEVAL LAYER ────────────┐
│ query norm → dense(20) + sparse(20)     │
│ → RRF(k=60) → parents → rerank → top-6  │
│ cosine filter ≥ 0.35                    │
└──────────────────┬────────────────────┘
                   ▼
┌──────────── VALIDATION LAYER ───────────┐
│ relevance judge (×6) → generate         │
│ → groundedness judge → retry ≤3         │
│ → citations + trace_id → JSONL log      │
└─────────────────────────────────────────┘
```

---

## POWER PHRASES (sprinkle naturally — don't force all)

| Phrase | When to use |
|--------|-------------|
| "validated retrieval" | Opening, enterprise framing |
| "abstention on weak scores" | Threshold 0.35, off-topic queries |
| "confidence scoring" | cosine, reranker, judge pass/fail |
| "audit trail" | trace_id, JSONL |
| "human-in-the-loop" | thumbs up/down, feedback |
| "golden set regression" | offline eval |
| "fail closed" | judge API error → reject chunk |
| "bounded agent orchestration" | planner + parallel agents |
| "RAGAS-aligned metrics" | eval (not official RAGAS lib) |
| "production-oriented prototype" | honest scale caveat |

---

## IF INTERRUPTED — modular 60-second deep dives

### "Tell me more about hybrid retrieval"
> *"Dense catches paraphrases — 'how does attention work.' BM25 catches exact tokens — 'BLEU', 'self-attention.' RRF merges rankings without normalizing incompatible scores. CrossEncoder reranks only deduplicated parents — small n. Measured F1 ninety-three point nine on thirty queries."*

### "Tell me more about verification"
> *"Two failure modes: wrong chunk retrieved — relevance judge fixes garbage-in. Right chunks, wrong answer — groundedness judge fixes garbage-out. RAGAS maps to context relevancy and faithfulness. Either alone leaves a hole."*

### "Why not LangGraph?"
> *"Agentic means planner plus parallel workers with bounded steps — not LangGraph state machines. I used asyncio and LangChain. LangGraph would be a natural upgrade for checkpointing and visual routing — not what I shipped."*

### "How reduce latency?"
> *"Profile eight stages. My bottleneck is Stage five — reranking — and Stage eight — six relevance plus up to three groundedness calls. I'd add conditional reranking, batch judges, route simple queries to one-call summarize path, cache embedding and retrieval layers separately — never cache unvalidated final answers."*

### "What failed in production?"
> *"Two traces. One: all chunks failed relevance on 'self attention' — returned abstention. Two: answer generated but groundedness failed three times — logged grounded false. Tells me judges need tuning and I need more trace volume — retrieval offline metrics are much better than live verification pass rate today."*

---

## TOUGH FOLLOW-UPS — one-line answers

| Question | Answer |
|----------|--------|
| Cost per request? | ~0.07 cents chat typical on Gemini Flash paid; eight to twelve API calls; retrieval is local and free |
| Biggest weakness? | Single-paper index; faithfulness not fully scored; few production traces |
| Prove better than vanilla RAG? | Ablation: dense only → hybrid → plus rerank → plus relevance judge → plus groundedness; report F1 and pass rate at each layer |
| Multi-tenant? | Not built; would add metadata ACL pre-filter in index and per-tenant shards |
| Prompt injection? | Untrusted PDF is untrusted prompt; mitigations: groundedness, injection test cases in golden set |
| Why Gemini? | Multi-backend config; used for judges with low temp and safety settings; Ollama fallback exists |

---

## BODY LANGUAGE & DELIVERY TIPS

1. **Slow down on metrics** — "ninety-three point nine percent **retrieval** F1" (emphasize retrieval)
2. **Draw the 3 layers** — interviewers remember diagrams
3. **Say one honest gap unprompted** — builds trust ("faithfulness pending quota")
4. **Don't apologize** for prototype scale — say "architecture is production-oriented"
5. **End with a question** — "retrieval, verification, or eval?"
6. **If nervous** — use 2-minute version first, ask "want the full ten-minute architecture walk?"

---

## PRACTICE SCHEDULE (do this twice before interview)

| Round | Time | Focus |
|-------|------|-------|
| 1 | 5 min | Opening hook + 2-minute version only |
| 2 | 10 min | Full script with whiteboard |
| 3 | 15 min | Full script + answer 3 interrupts cold |
| 4 | 5 min | Metrics section only — no hesitation |
| 5 | 2 min | Closing + "why hire you" from INTERVIEW_PREP.md Q15 |

---

## "WHY SHOULD WE HIRE YOU?" — 30-second closer

> *"I don't just call an API — I built retrieve, verify, generate, measure, tune. I have defensible retrieval metrics, a validation layer enterprises care about, honest reporting of gaps, and I think in trade-offs — latency versus accuracy, cost versus trust. That's the engineer who ships RAG to production, not demo day."*

---

*Pair with: `COMPLETE_PROJECT_REPORT.md` (facts), `INTERVIEW_PREP.md` (15 Q&A + cost).*
