"""
Query result cache — in-memory with optional Redis backend.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from config import Config

try:
    import redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class QueryCache:
    """TTL cache for retrieval results. Redis when available, else in-memory."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        redis_url: Optional[str] = None,
        prefix: str = "rag:query:",
    ):
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self._memory: Dict[str, tuple[float, Any]] = {}
        self._redis = None

        url = redis_url or getattr(Config, "REDIS_URL", None)
        if url and HAS_REDIS:
            try:
                self._redis = redis.from_url(url, decode_responses=True)
                self._redis.ping()
                print("[query_cache] Connected to Redis")
            except Exception as exc:
                print(f"[query_cache] Redis unavailable, using in-memory: {exc}")

    def _key(self, query: str, extra: str = "") -> str:
        raw = f"{query.strip().lower()}|{extra}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
        return f"{self.prefix}{digest}"

    def get(self, query: str, extra: str = "") -> Optional[Any]:
        key = self._key(query, extra)
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    return json.loads(raw)
            except Exception:
                pass

        entry = self._memory.get(key)
        if entry:
            expires, value = entry
            if time.time() < expires:
                return value
            del self._memory[key]
        return None

    def set(self, query: str, value: Any, extra: str = "", ttl: Optional[int] = None) -> None:
        key = self._key(query, extra)
        ttl = ttl or self.ttl_seconds
        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value, default=str))
                return
            except Exception:
                pass
        self._memory[key] = (time.time() + ttl, value)

    def invalidate(self, query: str, extra: str = "") -> None:
        key = self._key(query, extra)
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        self._memory.pop(key, None)

    def clear(self) -> None:
        self._memory.clear()
        if self._redis:
            try:
                for key in self._redis.scan_iter(f"{self.prefix}*"):
                    self._redis.delete(key)
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        from optimization.common import load_runtime_config

        return bool(load_runtime_config().get("query_cache_enabled", False))


# Module-level singleton
_cache: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    global _cache
    if _cache is None:
        _cache = QueryCache(ttl_seconds=getattr(Config, "QUERY_CACHE_TTL", 3600))
    return _cache
