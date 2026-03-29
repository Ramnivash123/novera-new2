
# ============================================================
# STAGE 1 — Build frontend
# ============================================================
FROM node:20-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
# ci is faster and more reproducible than install in CI/CD
RUN npm ci
COPY frontend/ .
RUN npm run build


# ============================================================
# STAGE 2 — Backend (Python 3.11-slim)
# ============================================================
FROM python:3.11-slim

# System dependencies — grouped to minimise layer count
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libpq-dev postgresql-client \
    poppler-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Prevent pip from writing to /root which may be read-only
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python dependencies first (layer-cached if requirements unchanged)
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY backend .

# Remove any __pycache__ carried over from the build context
RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name "*.pyc" -delete 2>/dev/null || true

# Frontend build
COPY --from=frontend-build /frontend/dist ./frontend/dist

# Runtime directories
RUN mkdir -p data/uploads/branding data/processed logs

# Non-root user for security
RUN addgroup --system novera && adduser --system --group novera && \
    chown -R novera:novera /app
USER novera

EXPOSE 8000

# ── Gunicorn with uvicorn workers ─────────────────────────────────────────────
# WEB_CONCURRENCY defaults to 2. Render sets this automatically based on
# the instance type. Uvicorn workers handle async I/O properly.
# --timeout 120 gives slow LLM calls room to complete.
# --graceful-timeout 30 ensures in-flight requests finish before shutdown.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
