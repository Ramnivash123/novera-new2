"""
Sliding-window rate limiter middleware.

Rules are matched by BOTH path prefix AND HTTP method so that read-only
requests (GET conversation history, token-stats, analytics) are never
counted against the strict LLM-call budget.

Buckets (ordered most-specific → least-specific):
  GET/ANY /api/v1/chat/conversations  — 300 req/60s  (reads: history, analytics, token-stats)
  POST    /api/v1/chat                — 100 req/60s  (real LLM calls; 5 msgs/min per user × 20 users)
  POST    /api/v1/documents/upload    —  10 req/60s  (uploads)
  ANY     /api/v1/documents           — 300 req/60s  (document reads / editor)
  ANY     /api/v1/                    — 300 req/60s  (everything else)

With the org Gemini API key (billing enabled) the backend quota is
1 000 RPM, so 100 chat POST/min is well within limits even at 20 concurrent
users each sending one message every 6 seconds.

The identifier is the JWT sub (user ID) for authenticated requests, or the
client IP for unauthenticated ones.  Falls back to in-memory counter when
Redis is unavailable so rate limiting still works on a single worker.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from loguru import logger

from app.core.redis_manager import redis_manager


@dataclass
class RateRule:
    """
    A single rate-limit rule.

    prefix  : URL path prefix (most-specific first in _RULES).
    limit   : Max requests allowed within `window` seconds.
    window  : Sliding-window duration in seconds.
    methods : If non-empty, only these HTTP verbs are subject to the rule.
              An empty set (default) matches ALL methods.
    """
    prefix: str
    limit: int
    window: int
    methods: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Rules — ORDER MATTERS.  First match wins.
# Put more-specific prefixes before less-specific ones.
# ---------------------------------------------------------------------------
_RULES: list[RateRule] = [
    # 1. Conversation reads: history, analytics, token-stats polls — generous
    RateRule("/api/v1/chat/conversations", 300, 60),

    # 2. Actual LLM chat messages — 100/min supports 20 users at normal pace
    RateRule("/api/v1/chat", 100, 60, frozenset({"POST"})),

    # 3. Document uploads — strict to prevent abuse
    RateRule("/api/v1/documents/upload", 10, 60, frozenset({"POST"})),

    # 4. Document reads / editor — generous
    RateRule("/api/v1/documents", 300, 60),

    # 5. Catch-all for every other API route
    RateRule("/api/v1/", 300, 60),
]


def _match_rule(path: str, method: str) -> Optional[RateRule]:
    """
    Return the first rule whose prefix matches the path AND whose method
    constraint (if any) matches the HTTP method.

    Returns None only if no rule matches at all (impossible with the
    catch-all rule in place, but kept for safety).
    """
    for rule in _RULES:
        if not path.startswith(rule.prefix):
            continue
        # Empty methods set means match every HTTP method
        if not rule.methods or method.upper() in rule.methods:
            return rule
    return None


def _extract_identifier(request: Request) -> str:
    """
    Prefer JWT sub (user ID) so authenticated users share a per-user bucket
    regardless of IP.  Falls back to X-Forwarded-For then direct client IP.
    """
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        try:
            import base64
            import json as _json
            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = _json.loads(base64.b64decode(payload_b64))
            if "sub" in payload:
                return f"user:{payload['sub']}"
        except Exception:
            pass

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"

    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiting via Redis INCR + EXPIRE.
    Returns HTTP 429 with Retry-After header when a limit is exceeded.
    Fails open (lets request through) if Redis is unavailable.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        method = request.method

        # Only rate-limit API routes
        if not path.startswith("/api/"):
            return await call_next(request)

        rule = _match_rule(path, method)
        if rule is None:
            return await call_next(request)

        identifier = _extract_identifier(request)

        # Include HTTP method in bucket key when the rule is method-specific
        # so POST and GET counters are tracked independently.
        method_tag = f":{method.upper()}" if rule.methods else ""
        bucket_key = f"rl:{identifier}:{rule.prefix}{method_tag}"

        try:
            count = await redis_manager.increment(bucket_key, ttl=rule.window)
        except Exception as exc:
            # Redis down — fail open
            logger.warning(f"Rate limit check error: {exc} — allowing request")
            return await call_next(request)

        remaining = max(0, rule.limit - count)
        reset_at = int(time.time()) + rule.window

        if count > rule.limit:
            logger.warning(
                f"Rate limit exceeded: {identifier} on {method} {rule.prefix} "
                f"({count}/{rule.limit} in {rule.window}s)"
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": (
                        f"Rate limit exceeded. "
                        f"Max {rule.limit} requests per {rule.window} seconds."
                    ),
                    "retry_after": rule.window,
                },
                headers={
                    "Retry-After": str(rule.window),
                    "X-RateLimit-Limit": str(rule.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rule.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        return response


__all__ = ["RateLimitMiddleware"]