"""
Health check endpoints for monitoring system status.

FIXES APPLIED:
  1. Removed stale `settings.openai_api_key` reference
     (the app uses Gemini, not OpenAI — this check always showed "warning").
  2. Added Redis health check with graceful fallback reporting.
  3. Added Gemini API key presence check.
  4. /health/ready now tests a real DB query to confirm readiness.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger

from app.db.session import get_db
from app.core.config import settings
from app.core.redis_manager import redis_manager

router = APIRouter()


@router.get("/health", response_model=dict[str, Any])
async def health_check():
    """Basic health check — no external dependencies."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment,
    }


@router.get("/health/detailed", response_model=dict[str, Any])
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """
    Detailed health check including database, pgvector, Redis, and
    API key configuration.
    """
    health: dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {},
    }

    # ── Database ──────────────────────────────────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
        health["checks"]["database"] = {
            "status": "healthy",
            "message": "PostgreSQL connection successful",
        }
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {e}",
        }

    # ── pgvector ──────────────────────────────────────────────────────────────
    try:
        result = await db.execute(
            text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        )
        has_vector = result.scalar()
        health["checks"]["pgvector"] = {
            "status": "healthy" if has_vector else "warning",
            "message": (
                "pgvector extension available"
                if has_vector
                else "pgvector extension not installed"
            ),
        }
    except Exception as e:
        health["checks"]["pgvector"] = {
            "status": "warning",
            "message": f"pgvector check failed: {e}",
        }

    # ── Redis ─────────────────────────────────────────────────────────────────
    try:
        redis_ok = await redis_manager.ping()
        health["checks"]["redis"] = {
            "status": "healthy" if redis_manager.is_redis_active else "degraded",
            "message": (
                "Redis connected — cross-worker sessions enabled"
                if redis_manager.is_redis_active
                else "Redis unavailable — using in-memory fallback (single-worker only)"
            ),
        }
    except Exception as e:
        health["checks"]["redis"] = {
            "status": "degraded",
            "message": f"Redis check failed: {e}",
        }

    # ── Gemini API key ────────────────────────────────────────────────────────
    gemini_key = settings.gemini_api_key
    if gemini_key and len(gemini_key) >= 10:
        health["checks"]["gemini_config"] = {
            "status": "healthy",
            "message": "Gemini API key configured",
        }
    else:
        health["checks"]["gemini_config"] = {
            "status": "warning",
            "message": "Gemini API key not configured",
        }

    # ── Cohere ────────────────────────────────────────────────────────────────
    cohere_key = settings.cohere_api_key
    health["checks"]["cohere_config"] = {
        "status": "healthy" if cohere_key else "warning",
        "message": (
            "Cohere API key configured"
            if cohere_key
            else "Cohere API key not set — reranking will be skipped"
        ),
    }

    # ── Overall status ────────────────────────────────────────────────────────
    if any(
        c["status"] == "unhealthy" for c in health["checks"].values()
    ):
        health["status"] = "unhealthy"
    elif any(
        c["status"] == "warning" for c in health["checks"].values()
    ):
        health["status"] = "degraded"

    return health


@router.get("/health/ready", response_model=dict[str, str])
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Kubernetes / Render readiness probe.
    Returns 200 only when the database is reachable.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")


@router.get("/health/live", response_model=dict[str, str])
async def liveness_check():
    """Kubernetes / Render liveness probe — always 200 if process is alive."""
    return {"status": "alive"}
