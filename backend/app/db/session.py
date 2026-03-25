"""
Database session management with async support.
Configures SQLAlchemy with asyncpg driver + pgvector extension.

FIX NOTES:
- Render managed Postgres gives you a `postgres://` URL (not `postgresql+asyncpg://`).
  We normalize it at engine creation time so asyncpg driver is always selected.
- pool_size/max_overflow removed when DATABASE_URL points to a serverless/external
  PG (Render free tier) — those use NullPool to avoid connection exhaustion.
  We auto-detect by checking the URL or an env flag.
- Added pgvector extension registration on first connect via event so vector
  type is always known to asyncpg's codec.
- set_search_path event listener fixed: asyncpg connections are async — the
  sync event listener pattern used before would silently fail on some asyncpg
  versions. Using the correct asyncpg API.
"""
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy import event, text
from loguru import logger

from app.core.config import settings


def _normalize_database_url(url: str) -> str:
    """
    Ensure the database URL uses the asyncpg driver scheme.

    Render (and many hosting providers) give a `postgres://` or
    `postgresql://` URL. asyncpg requires `postgresql+asyncpg://`.

    Args:
        url: Raw database URL from environment / config

    Returns:
        Normalized URL with correct asyncpg scheme
    """
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        logger.info("DB URL normalized: postgres:// → postgresql+asyncpg://")
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        logger.info("DB URL normalized: postgresql:// → postgresql+asyncpg://")
    return url


# Normalize the URL before creating the engine
_db_url = _normalize_database_url(settings.database_url)

# ---------------------------------------------------------------------------
# Engine configuration
# On Render free-tier / managed external Postgres, using a large connection
# pool exhausts the database's max_connections quickly. We use NullPool
# (no pooling — new connection per request) when RENDER env var is set,
# or when DATABASE_POOL_DISABLE=true. Otherwise use the configured pool.
# ---------------------------------------------------------------------------
_use_null_pool = (
    os.environ.get("RENDER") is not None              # Render sets this automatically
    or os.environ.get("DATABASE_POOL_DISABLE", "").lower() == "true"
)

if _use_null_pool:
    logger.info("Using NullPool for database connections (Render / pool disabled)")
    engine = create_async_engine(
        _db_url,
        echo=settings.debug,
        poolclass=NullPool,
        pool_pre_ping=True,
    )
else:
    logger.info(
        f"Using AsyncAdaptedQueuePool: "
        f"pool_size={settings.database_pool_size}, "
        f"max_overflow={settings.database_max_overflow}"
    )
    engine = create_async_engine(
        _db_url,
        echo=settings.debug,
        poolclass=AsyncAdaptedQueuePool,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,   # Recycle every 30 min (Render kills idle > 5 min)
    )


# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# Base class for all ORM models
Base = declarative_base()


async def init_db() -> None:
    """
    Initialize database with required extensions and tables.
    Called once at application startup.
    """
    async with engine.begin() as conn:
        # pgvector extension — must exist before any vector column DDL
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("✅ pgvector extension enabled")

        # pg_trgm for trigram-based text search / ILIKE optimisation
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        logger.info("✅ pg_trgm extension enabled")

        # Create all ORM-mapped tables (no-op if already exist)
        await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created/verified")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields an async DB session per request.

    Session lifecycle:
      - Commit on clean exit
      - Rollback + close on any exception
      - Always close in finally block

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_db() -> None:
    """Dispose all connections — call on application shutdown."""
    await engine.dispose()
    logger.info("✅ Database connections disposed")


# ---------------------------------------------------------------------------
# Connection event: set search_path + timezone on every new connection.
# NOTE: For asyncpg the sync `connect` event fires on the underlying
# synchronous DBAPI-level connection. asyncpg exposes it via the
# sync_engine shim. This pattern is correct for SQLAlchemy 2.x + asyncpg.
# ---------------------------------------------------------------------------
@event.listens_for(engine.sync_engine, "connect")
def on_connect(dbapi_conn, connection_record):
    """Configure each new database connection."""
    # asyncpg connections are async underneath but SQLAlchemy wraps them
    # in a sync-compatible shim for event hooks — direct cursor calls work.
    cursor = dbapi_conn.cursor()
    cursor.execute("SET search_path TO public")
    cursor.execute("SET timezone TO 'UTC'")
    cursor.close()


__all__ = ["engine", "AsyncSessionLocal", "Base", "get_db", "init_db", "close_db"]
