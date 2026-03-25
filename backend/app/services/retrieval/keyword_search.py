"""
Keyword-based search using PostgreSQL full-text search.
Implements BM25-like ranking for exact match queries.
"""
from typing import List, Dict, Any, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger


class KeywordSearchService:
    """
    Service for keyword-based search using PostgreSQL full-text search.
    Complements vector search for exact matches and specific terms.
    """

    def __init__(self):
        self.default_top_k = 10

    def _safe_extract_metadata(self, metadata_obj: Any) -> dict:
        """Safely extract metadata from any object type."""
        if metadata_obj is None:
            return {}
        if isinstance(metadata_obj, dict):
            return metadata_obj
        if hasattr(metadata_obj, '__dict__'):
            try:
                return {
                    k: v for k, v in metadata_obj.__dict__.items()
                    if not k.startswith('_')
                }
            except Exception:
                pass
        if hasattr(metadata_obj, 'keys') and hasattr(metadata_obj, '__getitem__'):
            try:
                return {k: metadata_obj[k] for k in metadata_obj.keys()}
            except Exception:
                pass
        return {}

    async def search_keywords(
        self,
        query: str,
        db: AsyncSession,
        top_k: Optional[int] = None,
        doc_type: Optional[str] = None,
        department: Optional[str] = None,
        document_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search chunks using keyword matching with optional document filtering.
        
        FIXED: Added session rollback on error to prevent cascade failures.
        """
        k = top_k or self.default_top_k

        try:
            sql_parts = [
                """
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_type,
                    c.chunk_index,
                    c.page_numbers,
                    c.section_title,
                    c.token_count,
                    c.metadata as chunk_metadata,
                    d.id as doc_id,
                    d.filename,
                    d.doc_type,
                    d.department,
                    ts_rank_cd(
                        to_tsvector('english', c.content),
                        plainto_tsquery('english', :query)
                    ) as rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.status = 'completed'
                AND to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)
                """
            ]

            params = {"query": query}

            if doc_type:
                sql_parts.append("AND d.doc_type = :doc_type")
                params["doc_type"] = doc_type

            if department:
                sql_parts.append("AND d.department = :department")
                params["department"] = department

            if document_filter:
                filter_conditions = []
                for i, doc_name in enumerate(document_filter):
                    filter_conditions.append(
                        f"d.filename ILIKE :doc_filter_{i}"
                    )
                    params[f"doc_filter_{i}"] = f"%{doc_name}%"
                sql_parts.append(
                    f"AND ({' OR '.join(filter_conditions)})"
                )
                logger.info(
                    f"  Keyword search filtering to: {document_filter}"
                )

            sql_parts.append("ORDER BY rank DESC")
            sql_parts.append("LIMIT :limit")
            params["limit"] = k

            full_sql = "\n".join(sql_parts)

            result = await db.execute(text(full_sql), params)
            rows = result.fetchall()

            chunks = []
            for row in rows:
                chunk_metadata = self._safe_extract_metadata(
                    row.chunk_metadata
                )

                chunk_dict = {
                    'chunk_id': str(row.chunk_id),
                    'id': str(row.chunk_id),
                    'document_id': str(row.doc_id),
                    'document_name': row.filename,
                    'content': row.content,
                    'chunk_type': row.chunk_type,
                    'chunk_index': row.chunk_index,
                    'page_numbers': row.page_numbers,
                    'section_title': row.section_title,
                    'token_count': row.token_count,
                    'keyword_score': float(row.rank),
                    'chunk_metadata': chunk_metadata,
                    'metadata': {
                        'document_title': row.filename,
                        'doc_type': row.doc_type,
                        'department': row.department,
                        **chunk_metadata,
                    }
                }
                chunks.append(chunk_dict)

            logger.info(
                f"Keyword search found {len(chunks)} matching chunks"
            )
            return chunks

        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            # ============================================================
            # FIX: Rollback session to prevent cascade failures
            # ("current transaction is aborted" errors)
            # ============================================================
            try:
                await db.rollback()
                logger.info(
                    "Session rolled back after keyword search error"
                )
            except Exception as rollback_err:
                logger.error(
                    f"Rollback also failed: {str(rollback_err)}"
                )
            return []

    async def search_exact_phrase(
        self,
        phrase: str,
        db: AsyncSession,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for exact phrase matches."""
        k = top_k or self.default_top_k

        try:
            sql = """
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_type,
                    c.page_numbers,
                    c.section_title,
                    c.token_count,
                    c.metadata as chunk_metadata,
                    d.id as doc_id,
                    d.filename,
                    d.doc_type
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.status = 'completed'
                AND c.content ILIKE :phrase
                LIMIT :limit
            """

            result = await db.execute(
                text(sql),
                {"phrase": f"%{phrase}%", "limit": k}
            )
            rows = result.fetchall()

            chunks = []
            for row in rows:
                chunk_metadata = self._safe_extract_metadata(
                    row.chunk_metadata
                )

                chunk_dict = {
                    'chunk_id': str(row.chunk_id),
                    'id': str(row.chunk_id),
                    'document_id': str(row.doc_id),
                    'document_name': row.filename,
                    'content': row.content,
                    'chunk_type': row.chunk_type,
                    'page_numbers': row.page_numbers,
                    'section_title': row.section_title,
                    'token_count': row.token_count,
                    'exact_match': True,
                    'chunk_metadata': chunk_metadata,
                    'metadata': {
                        'document_title': row.filename,
                        'doc_type': row.doc_type,
                        **chunk_metadata,
                    }
                }
                chunks.append(chunk_dict)

            logger.info(
                f"Exact phrase search found {len(chunks)} chunks"
            )
            return chunks

        except Exception as e:
            logger.error(f"Exact phrase search failed: {str(e)}")
            try:
                await db.rollback()
            except Exception:
                pass
            return []

    async def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        db: AsyncSession,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search chunks by metadata fields."""
        k = top_k or self.default_top_k

        try:
            sql_parts = [
                """
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_type,
                    c.metadata as chunk_metadata,
                    d.id as doc_id,
                    d.filename
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.status = 'completed'
                """
            ]

            params = {}

            for i, (key, value) in enumerate(metadata_filters.items()):
                sql_parts.append(
                    f"AND c.metadata->>:key_{i} = :val_{i}"
                )
                params[f"key_{i}"] = key
                params[f"val_{i}"] = str(value)

            sql_parts.append("LIMIT :limit")
            params["limit"] = k

            full_sql = "\n".join(sql_parts)

            result = await db.execute(text(full_sql), params)
            rows = result.fetchall()

            chunks = []
            for row in rows:
                chunk_metadata = self._safe_extract_metadata(
                    row.chunk_metadata
                )

                chunk_dict = {
                    'chunk_id': str(row.chunk_id),
                    'id': str(row.chunk_id),
                    'document_id': str(row.doc_id),
                    'document_name': row.filename,
                    'content': row.content,
                    'chunk_type': row.chunk_type,
                    'chunk_metadata': chunk_metadata,
                    'metadata': {
                        'document_title': row.filename,
                        **chunk_metadata,
                    }
                }
                chunks.append(chunk_dict)

            return chunks

        except Exception as e:
            logger.error(f"Metadata search failed: {str(e)}")
            try:
                await db.rollback()
            except Exception:
                pass
            return []


# Global instance
keyword_search_service = KeywordSearchService()

__all__ = ['KeywordSearchService', 'keyword_search_service']
