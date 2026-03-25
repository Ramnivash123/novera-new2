"""
Main FastAPI application entry point.

FIXES APPLIED:
  1. Redis manager initialised during startup so all services share
     the same connected client from the first request onward.
  2. RateLimitMiddleware added to protect against traffic spikes.
  3. Removed the `print()` loop that dumped all routes to stdout
     on every cold start (noise in production logs).
  4. GZipMiddleware minimum_size raised to 2000 bytes — gzipping tiny
     JSON responses wastes CPU and adds latency.
  5. Removed duplicate `from app.core.config import settings` inside
     the lifespan that shadowed the module-level import.
  6. /api/docs and /api/redoc hidden in production (already was, kept).
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException
from loguru import logger

from app.core.config import settings
from app.core.redis_manager import redis_manager
from app.db.session import init_db, close_db
from app.api.endpoints import (
    health, documents, auth, chat, search,
    admin, document_editor, customization, organization,
)
from app.middleware.rate_limit import RateLimitMiddleware


# ── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level=settings.log_level,
    colorize=True,
)
logger.add(
    settings.log_file,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=settings.log_level,
    rotation=settings.log_rotation,
    retention=settings.log_retention,
    compression="zip",
    enqueue=True,          # non-blocking file writes
)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"   Environment : {settings.environment}")
    logger.info(f"   Debug mode  : {settings.debug}")

    try:
        # 1. Redis (non-fatal — falls back to in-memory automatically)
        await redis_manager.init(settings.redis_url)

        # 2. Database + extensions
        await init_db()
        logger.info("✅ Database ready")

        # Quick embedding dimension sanity-check (log only, don't block startup)
        try:
            from sqlalchemy import text as sa_text
            from app.db.session import AsyncSessionLocal
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    sa_text(
                        "SELECT array_length(embedding::real[], 1) AS dim, COUNT(*) AS cnt "
                        "FROM chunks GROUP BY dim"
                    )
                )
                for row in result.fetchall():
                    logger.info(f"📐 Embeddings: {row.dim}D × {row.cnt} chunks")
        except Exception as e:
            logger.warning(f"Embedding dimension check skipped: {e}")

        # 3. Upload directories
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.upload_dir + "/branding").mkdir(parents=True, exist_ok=True)

        # 4. Pre-warm embedding service
        try:
            from app.services.embedding.embedding_service import embedding_service
            logger.info(
                f"✅ Embedding service ready "
                f"(local_fallback={embedding_service.use_local_fallback})"
            )
        except Exception as e:
            logger.warning(f"Embedding service pre-warm failed: {e}")

        logger.info("🎉 Startup complete")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield  # ← app runs here

    logger.info("🛑 Shutting down...")
    try:
        await redis_manager.close()
        await close_db()
        logger.info("👋 Shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered knowledge assistant with RAG capabilities",
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
    openapi_url="/api/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)


# ── Frontend path resolution ──────────────────────────────────────────────────
_frontend_candidates = [
    Path(__file__).resolve().parent.parent / "frontend" / "dist",
    Path(__file__).resolve().parents[2] / "frontend" / "dist",
    Path("/app/frontend/dist"),
]
frontend_path: Path | None = next(
    (p for p in _frontend_candidates if p.exists() and (p / "index.html").exists()),
    None,
)
if frontend_path:
    logger.info(f"✅ Frontend found at {frontend_path}")
else:
    frontend_path = _frontend_candidates[0]
    logger.warning(
        f"⚠️ Frontend build not found. Tried: "
        + ", ".join(str(p) for p in _frontend_candidates)
    )


# ── Middleware (order matters — CORS must be outermost) ───────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=3600,
)
app.add_middleware(GZipMiddleware, minimum_size=2000)
app.add_middleware(RateLimitMiddleware)
logger.info("✅ Middleware configured (CORS → GZip → RateLimit)")


# ── API routers ───────────────────────────────────────────────────────────────
_pfx = settings.api_v1_prefix

for _router, _tag in [
    (health.router,          "Health"),
    (auth.router,            "Authentication"),
    (documents.router,       "Documents"),
    (search.router,          "Search"),
    (chat.router,            "Chat"),
    (customization.router,   "Customization"),
    (organization.router,    "Super Admin - Organizations"),
    (admin.router,           "Admin"),
    (document_editor.router, "Document Editor"),
]:
    app.include_router(_router, prefix=_pfx, tags=[_tag])

logger.info(f"✅ {len(app.routes)} routes registered")


# ── Static mounts (after API routers) ────────────────────────────────────────
upload_path = Path(settings.upload_dir)
upload_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_path)), name="uploads")

if frontend_path.exists() and (frontend_path / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(frontend_path / "assets")),
        name="frontend_assets",
    )


# ── Root / health ─────────────────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root_handler(request: Request):
    if request.method == "HEAD":
        return JSONResponse({"status": "ok"})
    if frontend_path and frontend_path.exists():
        idx = frontend_path / "index.html"
        if idx.exists():
            return FileResponse(idx)
    return JSONResponse({
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
    })


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if frontend_path and frontend_path.exists():
        f = frontend_path / "favicon.ico"
        if f.exists():
            return FileResponse(f)
    raise HTTPException(status_code=404)


# ── SPA fallback (must be last) ───────────────────────────────────────────────
@app.api_route("/{full_path:path}", methods=["GET"], include_in_schema=False)
async def spa_fallback(request: Request, full_path: str):
    for reserved in ("api", "uploads", "assets"):
        if full_path.startswith(reserved):
            raise HTTPException(status_code=404, detail="Not Found")

    if frontend_path and frontend_path.exists():
        static_file = frontend_path / full_path
        try:
            static_file.resolve().relative_to(frontend_path.resolve())
            if static_file.exists() and static_file.is_file():
                return FileResponse(static_file)
        except ValueError:
            pass
        idx = frontend_path / "index.html"
        if idx.exists():
            return FileResponse(idx)

    raise HTTPException(status_code=404, detail="Not Found")


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code} on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "message": "Invalid request data",
            "details": exc.errors() if settings.debug else None,
        },
    )


@app.exception_handler(IntegrityError)
async def integrity_exception_handler(request: Request, exc: IntegrityError):
    logger.error(f"Integrity error on {request.url.path}: {exc}")
    msg = str(exc.orig) if hasattr(exc, "orig") else str(exc)
    if "duplicate key" in msg.lower() or "unique constraint" in msg.lower():
        return JSONResponse(
            status_code=409,
            content={"error": "Conflict", "message": "A record with this information already exists"},
        )
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid data", "message": "The provided data violates database constraints"},
    )


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(f"Database error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Database error",
            "message": str(exc) if settings.debug else "A database error occurred",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
            "type": type(exc).__name__ if settings.debug else None,
        },
    )


# ── Local dev runner ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower(),
    )
