"""
Cross-session user memory with exponential moving average preference learning.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

USER_PROFILES_PATH = Path("./memory/user_profiles.jsonl")
EMA_ALPHA = 0.3

DEFAULT_PROFILE: Dict[str, Any] = {
    "user_id": "",
    "prefers_concise": 0.5,
    "wants_page_citations": 0.5,
    "avoids_jargon": 0.5,
    "boost_terms": {},
    "session_count": 0,
    "positive_short_answers": 0,
    "positive_long_answers": 0,
    "citation_followups": 0,
    "rephrase_terms": {},
    "updated_at": None,
}


def _load_all_profiles() -> Dict[str, Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    if not USER_PROFILES_PATH.exists():
        return profiles
    with USER_PROFILES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                profile = json.loads(line)
                uid = profile.get("user_id")
                if uid:
                    profiles[uid] = profile
            except json.JSONDecodeError:
                continue
    return profiles


def _save_all_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    USER_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USER_PROFILES_PATH.open("w", encoding="utf-8") as f:
        for profile in profiles.values():
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")


def _ema(old: float, new_signal: float, alpha: float = EMA_ALPHA) -> float:
    return alpha * new_signal + (1 - alpha) * old


def get_or_create_profile(user_id: str) -> Dict[str, Any]:
    profiles = _load_all_profiles()
    if user_id not in profiles:
        profile = dict(DEFAULT_PROFILE)
        profile["user_id"] = user_id
        profiles[user_id] = profile
        _save_all_profiles(profiles)
    return profiles[user_id]


def _extract_terms(query: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", query.lower())
    stop = {"what", "when", "where", "which", "about", "explain", "please", "does", "that", "this", "with", "from"}
    return [t for t in tokens if t not in stop][:8]


def update_from_feedback(
    user_id: str,
    rating: str,
    answer: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Update preferences from explicit thumbs up/down."""
    profiles = _load_all_profiles()
    profile = profiles.get(user_id) or dict(DEFAULT_PROFILE)
    profile["user_id"] = user_id
    metadata = metadata or {}

    answer_len = len(answer or "")
    is_short = answer_len < 400

    if rating == "positive":
        if is_short:
            profile["positive_short_answers"] = profile.get("positive_short_answers", 0) + 1
            profile["prefers_concise"] = _ema(profile.get("prefers_concise", 0.5), 1.0)
        else:
            profile["positive_long_answers"] = profile.get("positive_long_answers", 0) + 1
            profile["prefers_concise"] = _ema(profile.get("prefers_concise", 0.5), 0.0)

        if metadata.get("had_citations") or "page" in (answer or "").lower():
            profile["wants_page_citations"] = _ema(profile.get("wants_page_citations", 0.5), 1.0)

    elif rating == "negative":
        if "jargon" in (metadata.get("comment") or "").lower():
            profile["avoids_jargon"] = _ema(profile.get("avoids_jargon", 0.5), 1.0)
        if not is_short:
            profile["prefers_concise"] = _ema(profile.get("prefers_concise", 0.5), 0.8)

    profile["updated_at"] = time.time()
    profiles[user_id] = profile
    _save_all_profiles(profiles)
    return profile


def update_from_interaction(
    user_id: str,
    query: str,
    is_follow_up: bool = False,
    prior_query: str = "",
) -> Dict[str, Any]:
    """Learn from implicit signals: follow-ups on citations, rephrased terms."""
    profiles = _load_all_profiles()
    profile = profiles.get(user_id) or dict(DEFAULT_PROFILE)
    profile["user_id"] = user_id

    q_lower = query.lower()
    if is_follow_up and any(k in q_lower for k in ("page", "source", "cite", "reference", "where")):
        profile["citation_followups"] = profile.get("citation_followups", 0) + 1
        profile["wants_page_citations"] = _ema(profile.get("wants_page_citations", 0.5), 1.0)

    if prior_query:
        prior_terms = set(_extract_terms(prior_query))
        new_terms = [t for t in _extract_terms(query) if t not in prior_terms]
        rephrase = profile.get("rephrase_terms") or {}
        boost = profile.get("boost_terms") or {}
        for term in new_terms:
            rephrase[term] = rephrase.get(term, 0) + 1
            if rephrase[term] >= 2:
                boost[term] = boost.get(term, 0) + 1
        profile["rephrase_terms"] = rephrase
        profile["boost_terms"] = boost

    profile["updated_at"] = time.time()
    profiles[user_id] = profile
    _save_all_profiles(profiles)
    return profile


def start_session(user_id: str, session_id: str) -> None:
    profile = get_or_create_profile(user_id)
    profile["last_session_id"] = session_id
    profile["session_count"] = profile.get("session_count", 0) + 1
    profiles = _load_all_profiles()
    profiles[user_id] = profile
    _save_all_profiles(profiles)


def format_preference_prompt(user_id: str) -> str:
    """Build system-message block from learned preferences."""
    profile = get_or_create_profile(user_id)
    lines = ["User preferences detected from history:", ""]

    if profile.get("prefers_concise", 0.5) >= 0.6:
        lines.append("- Prefers: bullet-point, concise answers")
    elif profile.get("prefers_concise", 0.5) <= 0.4:
        lines.append("- Prefers: detailed, comprehensive answers")

    if profile.get("wants_page_citations", 0.5) >= 0.55:
        lines.append("- Always wants: page number citations and source references")

    if profile.get("avoids_jargon", 0.5) >= 0.55:
        lines.append("- Avoids: technical jargon — use plain language")

    boost = profile.get("boost_terms") or {}
    top_terms = sorted(boost.items(), key=lambda x: x[1], reverse=True)[:5]
    if top_terms:
        terms = ", ".join(t for t, _ in top_terms)
        lines.append(f"- Frequently uses terms (boost in answers): {terms}")

    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


def boost_query(user_id: str, query: str) -> str:
    """Append frequently-used user terms to retrieval query."""
    profile = get_or_create_profile(user_id)
    boost = profile.get("boost_terms") or {}
    extras = [t for t, count in sorted(boost.items(), key=lambda x: -x[1])[:3] if count >= 2]
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


def resolve_user_id(session_id: str = "", explicit_user_id: str = "") -> str:
    if explicit_user_id:
        return explicit_user_id
    if session_id:
        return f"session:{session_id}"
    return "anonymous"
