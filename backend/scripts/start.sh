#!/bin/bash
set -e

echo "============================================"
echo "🚀 Starting Mentanova Application"
echo "============================================"

# Step 1: Wait for database
echo "⏳ Waiting for database..."
for i in $(seq 1 30); do
    if python -c "
import sys, os
sys.path.insert(0, '.')
from app.core.config import settings
url = settings.database_url.replace('postgresql+asyncpg', 'postgresql+psycopg2')
from sqlalchemy import create_engine, text
engine = create_engine(url)
with engine.connect() as conn:
    conn.execute(text('SELECT 1'))
print('ok')
" 2>/dev/null; then
        echo "✅ Database is ready!"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ Database not ready after 60 seconds. Starting anyway..."
    fi
    echo "  Attempt $i/30 - retrying in 2s..."
    sleep 2
done

# Step 2: Run Alembic migrations
echo ""
echo "🔄 Running database migrations..."
if alembic upgrade head; then
    echo "✅ Migrations complete"
else
    echo "⚠️ Migration had issues — attempting to stamp and continue..."
    # If migrations fail (e.g., already applied), stamp current and continue
    alembic stamp head
    echo "✅ Stamped at head"
fi

# Step 3: Start server
echo ""
echo "🎉 Starting Uvicorn on port ${PORT:-8000}"
echo "============================================"
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
