"""
Shared JSONL utilities and runtime configuration for Level 5 optimization.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNTIME_CONFIG_PATH = Path("./optimization/runtime_config.json")
OPTIMIZATION_HISTORY_PATH = Path("./optimization/optimization_history.jsonl")
PROMPT_HISTORY_PATH = Path("./optimization/prompt_history.jsonl")
SELF_HEALING_LOG_PATH = Path("./logs/self_healing_log.jsonl")

DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "min_retrieval_score": 0.35,
    "query_cache_enabled": False,
    "active_prompts": {
        "generation": "default_v1",
        "verification": "default_v1",
        "relevance": "default_v1",
    },
    "ab_test": {},
    "updated_at": None,
}


def load_jsonl(path: Path, since_ts: Optional[float] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_ts is not None:
                ts = record.get("ts") or record.get("timestamp")
                if ts is not None and float(ts) < since_ts:
                    continue
            records.append(record)
    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_runtime_config() -> Dict[str, Any]:
    if not RUNTIME_CONFIG_PATH.exists():
        return dict(DEFAULT_RUNTIME_CONFIG)
    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(DEFAULT_RUNTIME_CONFIG)
    merged.update(data)
    return merged


def save_runtime_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    config = load_runtime_config()
    config.update(updates)
    config["updated_at"] = time.time()
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


def update_env_min_retrieval_score(value: float) -> None:
    """Persist threshold to .env for restart durability."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    key = "MIN_RETRIEVAL_SCORE"
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def send_alert(message: str, channel: str = "console") -> None:
    webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if channel == "slack" and webhook:
        try:
            import requests

            requests.post(webhook, json={"text": message}, timeout=10)
            return
        except Exception as exc:
            print(f"[alert] Slack failed: {exc}")
    print(f"[ALERT] {message}")
