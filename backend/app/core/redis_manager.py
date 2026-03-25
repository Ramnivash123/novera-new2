"""
Redis manager with graceful in-memory fallback.

On Render free tier or when Redis is unavailable, the app falls back
to a thread-safe in-memory store so nothing hard-crashes. Production
deployments with Redis get full cross-worker session sharing.

Usage:
    from app.core.redis_manager import redis_manager

    await redis_manager.set("key", "value", ttl=3600)
    val = await redis_manager.get("key")
    await redis_manager.delete("key")
    await redis_manager.exists("key")
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional
from loguru import logger


class InMemoryFallback:
    """Thread-safe, TTL-aware in-memory cache used when Redis is absent."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        async with self._lock:
            expires_at = (time.monotonic() + ttl) if ttl else None
            self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def keys(self, pattern: str) -> list[str]:
        """Return all keys matching a prefix (strips trailing '*')."""
        prefix = pattern.rstrip("*")
        async with self._lock:
            now = time.monotonic()
            return [
                k for k, (_, exp) in self._store.items()
                if k.startswith(prefix) and (exp is None or now <= exp)
            ]

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class RedisManager:
    """
    Async Redis client wrapper with automatic in-memory fallback.

    Call `await redis_manager.init()` during application startup.
    After that, all get/set/delete calls are safe regardless of whether
    Redis is actually reachable.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._fallback = InMemoryFallback()
        self._use_fallback = True

    async def init(self, url: str) -> None:
        """
        Try to connect to Redis. Falls back silently if unavailable.
        Safe to call multiple times (idempotent).
        """
        if not self._use_fallback:
            return  # already connected

        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(
                url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            await client.ping()
            self._client = client
            self._use_fallback = False
            logger.info("✅ Redis connected — cross-worker sessions enabled")
        except Exception as e:
            logger.warning(
                f"⚠️ Redis unavailable ({e}). "
                "Using in-memory fallback — conversations will NOT persist across restarts."
            )
            self._use_fallback = True

    # ── Core operations ──────────────────────────────────────────────────────

    async def get(self, key: str) -> Optional[str]:
        if self._use_fallback:
            return await self._fallback.get(key)
        try:
            return await self._client.get(key)
        except Exception as e:
            logger.warning(f"Redis GET error ({key}): {e} — using fallback")
            return await self._fallback.get(key)

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if self._use_fallback:
            await self._fallback.set(key, value, ttl)
            return
        try:
            if ttl:
                await self._client.setex(key, ttl, value)
            else:
                await self._client.set(key, value)
            # Keep fallback in sync so a mid-flight Redis failure doesn't lose data
            await self._fallback.set(key, value, ttl)
        except Exception as e:
            logger.warning(f"Redis SET error ({key}): {e} — writing to fallback only")
            await self._fallback.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        if self._use_fallback:
            await self._fallback.delete(key)
            return
        try:
            await self._client.delete(key)
        except Exception as e:
            logger.warning(f"Redis DELETE error ({key}): {e}")
        await self._fallback.delete(key)

    async def exists(self, key: str) -> bool:
        if self._use_fallback:
            return await self._fallback.exists(key)
        try:
            return bool(await self._client.exists(key))
        except Exception as e:
            logger.warning(f"Redis EXISTS error ({key}): {e}")
            return await self._fallback.exists(key)

    async def keys(self, pattern: str) -> list[str]:
        if self._use_fallback:
            return await self._fallback.keys(pattern)
        try:
            return await self._client.keys(pattern)
        except Exception as e:
            logger.warning(f"Redis KEYS error ({pattern}): {e}")
            return await self._fallback.keys(pattern)

    # ── JSON helpers ─────────────────────────────────────────────────────────

    async def get_json(self, key: str) -> Optional[Any]:
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        await self.set(key, json.dumps(value, default=str), ttl)

    # ── Rate-limiting helper ──────────────────────────────────────────────────

    async def increment(self, key: str, ttl: int = 60) -> int:
        """
        Increment a counter, creating it with TTL if it doesn't exist.
        Used for sliding-window rate limiting.
        Returns the new count.
        """
        if self._use_fallback:
            raw = await self._fallback.get(key)
            count = (int(raw) + 1) if raw else 1
            await self._fallback.set(key, str(count), ttl)
            return count
        try:
            pipe = self._client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.warning(f"Redis INCR error ({key}): {e}")
            raw = await self._fallback.get(key)
            count = (int(raw) + 1) if raw else 1
            await self._fallback.set(key, str(count), ttl)
            return count

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def ping(self) -> bool:
        if self._use_fallback:
            return True
        try:
            return await self._client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass

    @property
    def is_redis_active(self) -> bool:
        return not self._use_fallback


# Singleton — import this everywhere
redis_manager = RedisManager()

__all__ = ["redis_manager"]
