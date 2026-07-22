"""
Self-improving prompt registry with Gemini-generated variants and A/B testing.
"""

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Config
from evaluation.daily_eval import compute_metrics, _load_jsonl as load_traces_jsonl
from optimization.common import (
    PROMPT_HISTORY_PATH,
    append_jsonl,
    load_jsonl,
    load_runtime_config,
    save_runtime_config,
    send_alert,
)

FEEDBACK_PATH = Path("./logs/feedback.jsonl")
TRACES_PATH = Path("./logs/rag_traces.jsonl")
PERFORMANCE_THRESHOLD = 0.7
UNDERPERFORM_DAYS = 3
AB_TRAFFIC_FRACTION = 0.10
AB_MIN_SAMPLES = 100
DAY_SECONDS = 86400

PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "generation": {
        "default_v1": (
            "You are a document assistant. Answer using ONLY the document context below. "
            "Use simple, clear language. Mention page/source when helpful. "
            "If the answer is not in the context, say you don't know — do not guess."
        ),
        "strict_grounded_v1": (
            "You are a grounded document assistant. EVERY claim must appear verbatim or "
            "paraphrased from the context. If uncertain, respond: 'I cannot find that in the documents.' "
            "Include page numbers for all citations. Never use outside knowledge."
        ),
    },
    "verification": {
        "default_v1": (
            "You are a hallucination detector. Answer ONLY 'YES' or 'NO'. "
            "Does the answer contain ANY information NOT in the context?"
        ),
    },
    "relevance": {
        "default_v1": (
            "You are a strict relevance grader. Answer ONLY 'YES' or 'NO'. "
            "Is the context directly relevant to answering the question?"
        ),
    },
}


def _daily_verification_rates(traces: List[Dict[str, Any]], days: int = 7) -> List[float]:
    """Compute verification pass rate per day for last N days."""
    from evaluation.daily_eval import _is_verification_pass

    now = time.time()
    rates: List[float] = []
    for d in range(days):
        day_start = now - (d + 1) * DAY_SECONDS
        day_end = now - d * DAY_SECONDS
        day_traces = [
            t for t in traces
            if day_start <= float(t.get("ts", 0)) < day_end
        ]
        outcomes = []
        for t in day_traces:
            outcome = _is_verification_pass(t)
            if outcome is not None:
                outcomes.append(outcome)
        if outcomes:
            rates.append(sum(outcomes) / len(outcomes))
    return rates


def should_optimize_prompts() -> bool:
    traces = load_traces_jsonl(TRACES_PATH)
    rates = _daily_verification_rates(traces, days=UNDERPERFORM_DAYS)
    if len(rates) < UNDERPERFORM_DAYS:
        return False
    return all(r < PERFORMANCE_THRESHOLD for r in rates[:UNDERPERFORM_DAYS])


async def _generate_variants_with_gemini(
    task: str,
    old_prompt: str,
    failure_examples: List[str],
) -> List[Dict[str, str]]:
    """Use Gemini to propose 3 improved prompt variants."""
    if not Config.GEMINI_API_KEY:
        return []

    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory

    genai.configure(api_key=Config.GEMINI_API_KEY)
    threshold = getattr(
        HarmBlockThreshold,
        Config.GEMINI_SAFETY_THRESHOLD,
        HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    )
    safety = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
    }
    model = genai.GenerativeModel(
        model_name=Config.GEMINI_MODEL,
        safety_settings=safety,
        generation_config=genai.GenerationConfig(
            temperature=Config.GEMINI_TEMPERATURE,
            max_output_tokens=2048,
        ),
    )

    examples_text = "\n---\n".join(failure_examples[:5]) or "No examples available."
    meta_prompt = f"""I have a prompt template that is underperforming:

Old prompt: {old_prompt}
Failure examples: {examples_text}

Generate 3 improved versions that:
- Are stricter about groundedness
- Explicitly forbid hallucinations
- Include "I don't know" encouragement

Format as a JSON array with objects having 'name' and 'prompt' keys only."""

    for attempt in range(Config.MAX_RETRIES):
        try:
            response = await asyncio.to_thread(model.generate_content, meta_prompt)
            text = (response.text or "").strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                variants = json.loads(match.group())
                return [
                    {"name": v["name"], "prompt": v["prompt"]}
                    for v in variants
                    if "name" in v and "prompt" in v
                ][:3]
        except Exception as exc:
            delay = Config.RETRY_BACKOFF * (2 ** attempt)
            print(f"[prompt_optimizer] Gemini retry {attempt + 1}: {exc}")
            await asyncio.sleep(delay)
    return []


def _collect_failure_examples(traces: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    from evaluation.daily_eval import _is_verification_pass

    examples = []
    for trace in reversed(traces):
        if _is_verification_pass(trace) is False:
            examples.append(
                f"Query: {trace.get('query', '')[:200]}\n"
                f"Answer: {trace.get('answer_preview', '')[:300]}"
            )
        if len(examples) >= limit:
            break
    return examples


def get_prompt(task: str, trace_id: str = "") -> tuple[str, str]:
    """
    Select prompt template for task. 10% A/B traffic when variants exist.
    Returns (template_name, prompt_text).
    """
    runtime = load_runtime_config()
    ab_test = runtime.get("ab_test") or {}
    templates = PROMPT_TEMPLATES.get(task, {})
    active = runtime.get("active_prompts", {}).get(task, "default_v1")

    if task in ab_test and ab_test[task].get("variants"):
        bucket = hash(trace_id or str(random.random())) % 100
        if bucket < int(AB_TRAFFIC_FRACTION * 100):
            variant_name = random.choice(list(ab_test[task]["variants"].keys()))
            return variant_name, ab_test[task]["variants"][variant_name]

    return active, templates.get(active, templates.get("default_v1", ""))


def record_ab_outcome(task: str, template_name: str, success: bool) -> None:
    runtime = load_runtime_config()
    ab_test = runtime.get("ab_test") or {}
    if task not in ab_test:
        return
    stats = ab_test[task].setdefault("stats", {})
    entry = stats.setdefault(template_name, {"samples": 0, "successes": 0})
    entry["samples"] += 1
    if success:
        entry["successes"] += 1
    save_runtime_config({"ab_test": ab_test})

    total = entry["samples"]
    if total >= AB_MIN_SAMPLES:
        _promote_best_variant(task)


def _promote_best_variant(task: str) -> None:
    runtime = load_runtime_config()
    ab_test = runtime.get("ab_test") or {}
    if task not in ab_test:
        return

    stats = ab_test[task].get("stats") or {}
    if not stats:
        return

    best_name = max(
        stats.keys(),
        key=lambda n: stats[n]["successes"] / max(stats[n]["samples"], 1),
    )
    best_rate = stats[best_name]["successes"] / max(stats[best_name]["samples"], 1)
    old_active = runtime.get("active_prompts", {}).get(task, "default_v1")

    variants = ab_test[task].get("variants", {})
    if best_name in variants:
        PROMPT_TEMPLATES.setdefault(task, {})[best_name] = variants[best_name]

    active_prompts = runtime.get("active_prompts", {})
    active_prompts[task] = best_name
    save_runtime_config({"active_prompts": active_prompts, "ab_test": {**ab_test, task: {}}})

    append_jsonl(PROMPT_HISTORY_PATH, {
        "ts": time.time(),
        "event": "prompt_promoted",
        "task": task,
        "old_template": old_active,
        "new_template": best_name,
        "success_rate": round(best_rate, 4),
        "samples": stats[best_name]["samples"],
    })
    send_alert(f"Prompt promoted for {task}: {old_active} → {best_name} (rate={best_rate:.2%})")


async def optimize_prompts(task: str = "generation", force: bool = False) -> Dict[str, Any]:
    """Generate variants and start A/B test when performance drops."""
    if not force and not should_optimize_prompts():
        return {"status": "skipped", "reason": "performance_ok"}

    traces = load_traces_jsonl(TRACES_PATH)
    feedback = load_jsonl(FEEDBACK_PATH)
    metrics = compute_metrics(traces, feedback)

    runtime = load_runtime_config()
    active = runtime.get("active_prompts", {}).get(task, "default_v1")
    old_prompt = PROMPT_TEMPLATES.get(task, {}).get(active, "")

    failures = _collect_failure_examples(traces)
    variants = await _generate_variants_with_gemini(task, old_prompt, failures)

    if not variants:
        # Fallback variants without Gemini
        variants = [
            {"name": f"{task}_fallback_strict", "prompt": old_prompt + " Say 'I don't know' when unsure."},
            {"name": f"{task}_fallback_cite", "prompt": old_prompt + " Always cite page numbers."},
            {"name": f"{task}_fallback_bullets", "prompt": old_prompt + " Use bullet points only."},
        ]

    variant_map = {v["name"]: v["prompt"] for v in variants}
    for name, prompt in variant_map.items():
        PROMPT_TEMPLATES.setdefault(task, {})[name] = prompt

    ab_test = runtime.get("ab_test") or {}
    ab_test[task] = {
        "started_at": time.time(),
        "variants": variant_map,
        "stats": {name: {"samples": 0, "successes": 0} for name in variant_map},
    }
    save_runtime_config({"ab_test": ab_test})

    record = {
        "ts": time.time(),
        "event": "ab_test_started",
        "task": task,
        "old_template": active,
        "variants": list(variant_map.keys()),
        "verification_pass_rate": metrics.get("verification_pass_rate"),
        "user_satisfaction": metrics.get("user_satisfaction"),
    }
    append_jsonl(PROMPT_HISTORY_PATH, record)
    send_alert(
        f"Prompt A/B test started for {task}. "
        f"verification_pass_rate={metrics.get('verification_pass_rate', 0):.2%}"
    )
    return record


def rollback_prompt(task: str) -> Optional[str]:
    """Rollback to previous prompt from history."""
    if not PROMPT_HISTORY_PATH.exists():
        return None

    history = load_jsonl(PROMPT_HISTORY_PATH)
    for record in reversed(history):
        if record.get("event") == "prompt_promoted" and record.get("task") == task:
            old_template = record.get("old_template")
            if old_template:
                runtime = load_runtime_config()
                active = runtime.get("active_prompts", {})
                active[task] = old_template
                save_runtime_config({"active_prompts": active})
                append_jsonl(PROMPT_HISTORY_PATH, {
                    "ts": time.time(),
                    "event": "prompt_rollback",
                    "task": task,
                    "restored_template": old_template,
                })
                return old_template
    return None


if __name__ == "__main__":
    asyncio.run(optimize_prompts())
