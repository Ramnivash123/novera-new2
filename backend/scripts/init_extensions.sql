-- Initialize PostgreSQL extensions for Mentanova RAG system
-- This script runs automatically when the database container starts

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable pg_trgm for efficient text search and similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable uuid-ossp for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable btree_gin for composite indexes
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Set timezone to UTC
SET timezone = 'UTC';

-- Create initial schema
CREATE SCHEMA IF NOT EXISTS public;

-- Grant privileges
GRANT ALL ON SCHEMA public TO mentanova_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mentanova_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mentanova_user;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE '✅ Mentanova database extensions initialized successfully';
    RAISE NOTICE '   - pgvector: enabled for vector similarity search';
    RAISE NOTICE '   - pg_trgm: enabled for text search';
    RAISE NOTICE '   - uuid-ossp: enabled for UUID generation';
END $$;