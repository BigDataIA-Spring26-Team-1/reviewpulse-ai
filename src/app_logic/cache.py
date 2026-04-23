"""Optional Redis-backed JSON cache for API responses."""

from __future__ import annotations

import hashlib
import importlib
import json
from typing import Any


def make_cache_key(namespace: str, **parts: Any) -> str:
    canonical = json.dumps(parts, ensure_ascii=True, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"reviewpulse:{namespace}:{digest}"


class ApiCache:
    """Small cache wrapper that degrades cleanly when Redis is unavailable."""

    def __init__(
        self,
        redis_url: str,
        *,
        socket_timeout_seconds: float = 0.2,
    ) -> None:
        self.redis_url = redis_url.strip()
        self.client: Any | None = None
        self.disabled_reason: str | None = None

        if not self.redis_url:
            self.disabled_reason = "REDIS_URL is not configured."
            return

        try:
            redis_module = importlib.import_module("redis")
            self.client = redis_module.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=socket_timeout_seconds,
                socket_timeout=socket_timeout_seconds,
            )
        except Exception as exc:
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            self.client = None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def get_json(self, key: str) -> Any | None:
        if self.client is None:
            return None
        try:
            payload = self.client.get(key)
        except Exception as exc:
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            return None
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        if self.client is None or ttl_seconds <= 0:
            return
        try:
            payload = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
            self.client.setex(key, ttl_seconds, payload)
        except Exception as exc:
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
