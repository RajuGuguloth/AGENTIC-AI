# Level 5 Autonomous RAG — Architecture & Tuning Guide

## Overview

Level 5 extends the Level 4 production RAG with **closed-loop self-improvement**:

| Capability | Module | Schedule |
|------------|--------|----------|
| Dynamic threshold optimization | `optimization/threshold_optimizer.py` | Weekly |
| Cross-session user memory | `memory/user_memory.py` | Every chat turn |
| Self-improving prompts | `optimization/prompt_optimizer.py` | Daily + A/B |
| Multi-modal retrieval | `retrieval/multimodal_retriever.py` | On demand |
| Self-healing pipeline | `observability/self_healer.py` | Hourly |

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ User Query  │────▶│ User Memory      │────▶│ Multimodal      │
│ + Session   │     │ (preferences)    │     │ Hybrid Search   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
┌─────────────┐     ┌──────────────────┐              ▼
│ Feedback    │◀────│ GeminiVerifier   │◀──── Prompt Optimizer
│ JSONL       │     │ + Regeneration   │      (A/B templates)
└──────┬──────┘     └──────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Autonomous Loop                                              │
│  daily_eval → threshold_optimizer → prompt_optimizer          │
│  self_healer (anomaly detect → auto-recover)                  │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd research_assistant
chmod +x migration_level5.sh
./migration_level5.sh
python app.py
```

## Capability Details

### 1. Dynamic Threshold Optimization

Grid-searches `MIN_RETRIEVAL_SCORE` ∈ [0.25 … 0.50] using weekly traces + feedback.

- **Precision@5**: thumbs-up traces that still retrieve at threshold
- **Recall@5**: relevance-found traces that survive threshold
- **F1**: selection metric

Updates `optimization/runtime_config.json` and `.env`. Alerts on Δ > 0.1.

```bash
python optimization/threshold_optimizer.py
python optimization/threshold_optimizer.py  # dry_run via code: optimize_threshold(dry_run=True)
```

### 2. Cross-Session User Memory

Profiles stored in `memory/user_profiles.jsonl`. Signals:

| Signal | Preference learned |
|--------|-------------------|
| Thumbs up on short answers | `prefers_concise` ↑ |
| Follow-ups about pages/sources | `wants_page_citations` ↑ |
| Repeated query terms | `boost_terms` for retrieval |

Injected as system-message block in `document_chat.py`.

### 3. Self-Improving Prompts

When `verification_pass_rate < 0.7` for 3 consecutive days:

1. Gemini generates 3 stricter prompt variants
2. 10% traffic A/B tested
3. Best variant promoted after 100 samples
4. History in `optimization/prompt_history.jsonl`

```bash
python -c "import asyncio; from optimization.prompt_optimizer import optimize_prompts; asyncio.run(optimize_prompts())"
```

### 4. Multi-Modal Retrieval

`MultimodalRetriever` extends hybrid search with CLIP fusion:

```python
from retrieval.multimodal_retriever import MultimodalRetriever

retriever = MultimodalRetriever(dense, sparse)
docs = retriever.search_multimodal("diagram of architecture", image_query="path/to/query.png")
```

Fusion: `0.6 × text_embedding + 0.4 × image_embedding` (L2-normalized).

### 5. Self-Healing

Hourly scan of last hour's traces:

| Anomaly | Recovery |
|---------|----------|
| Empty retrieval > 50% | Lower threshold by 0.1 |
| Latency > 2× baseline | Enable query cache |
| Verification drop > 20% | Rollback generation prompt |

Logs: `logs/self_healing_log.jsonl`. Uses Gemini token bucket for rate safety.

## Configuration

Add to `.env`:

```env
LEVEL5_ENABLED=true
REDIS_URL=redis://localhost:6379/0
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX
QUERY_CACHE_TTL=3600
GEMINI_TOKEN_BUCKET_RATE=10
GEMINI_IMAGE_SAFETY_THRESHOLD=BLOCK_MEDIUM_AND_ABOVE
```

Runtime overrides: `optimization/runtime_config.json`

## API

Feedback (unchanged, now updates user memory):

```bash
curl -X POST http://127.0.0.1:7860/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"trace_id":"...", "rating":"positive", "metadata":{"session_id":"abc"}}'
```

## Monitoring

```bash
python evaluation/daily_eval.py          # daily metrics + alert
python observability/self_healer.py      # hourly healing
tail -f logs/self_healing_log.jsonl
tail -f optimization/optimization_history.jsonl
```

## Tuning Guide

| Symptom | Action |
|---------|--------|
| Too many empty retrievals | Lower `MIN_RETRIEVAL_SCORE` or wait for self-healer |
| Hallucinations persist | Force prompt optimization: `optimize_prompts(force=True)` |
| High latency | Enable Redis cache via `REDIS_URL` or self-healer auto-enables |
| Poor multimodal results | Index image chunks via `universal_loader` + ensure CLIP paths valid |
| Rate limits on Gemini | Increase `GEMINI_TOKEN_BUCKET_CAPACITY`, `RETRY_BACKOFF` |

## Gemini-Specific Notes

- **Verification**: temperature 0.1, `BLOCK_MEDIUM_AND_ABOVE` safety
- **Prompt optimization**: uses Gemini with retry backoff
- **Image safety**: separate `GEMINI_IMAGE_SAFETY_THRESHOLD` for multimodal
- **Context window**: Gemini 1M tokens — for docs < 100 pages consider full-doc mode (future)

## File Map

```
optimization/
  common.py                 # runtime config, alerts
  threshold_optimizer.py    # weekly grid search
  prompt_optimizer.py       # A/B prompt evolution
memory/
  user_memory.py            # cross-session preferences
retrieval/
  multimodal_retriever.py   # CLIP fusion search
  query_cache.py              # Redis/in-memory cache
observability/
  self_healer.py              # hourly anomaly recovery
evaluation/
  daily_eval.py               # quality monitoring
tests/
  test_level5.py              # unit tests
```
