"""
Vector similarity search using pgvector.
Implements cosine similarity search with filtering and ranking.

FIX NOTES:
- asyncpg driver does NOT support SQLAlchemy :name style params with PostgreSQL
  type casts (::vector). The ORM's cosine_distance() method emits
  `c.embedding <=> :embedding::vector` which breaks asyncpg's $1 positional
  binding. Fix: serialize the embedding list to a pgvector-compatible string
  and pass it via sqlalchemy text() with explicit cast, OR use func.cast().
  We use the safest approach: convert embedding to string and use text() SQL
  with $1 positional param via connection.execute() raw style — but keeping
  ORM for joins. The cleanest ORM-compatible fix is to pass the embedding
  as a typed bindparam with explicit Vector type.
- Added explicit rollback on exception so callers don't get a broken session.
- Fixed doc_metadata attribute name (was referencing wrong alias).
"""
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from sqlalchemy import bindparam, String
from loguru import logger

from app.models.document import Chunk, Document
from app.core.config import settings


class VectorSearchService:
    """
    Service for semantic similarity search using vector embeddings.
    Leverages pgvector's cosine similarity for efficient retrieval.
    """

    def __init__(self):
        self.top_k = settings.retrieval_top_k
        self.similarity_threshold = settings.similarity_threshold

    def _safe_extract_metadata(self, metadata_obj: Any) -> dict:
        """
        Safely extract metadata from SQLAlchemy object or dict.

        Args:
            metadata_obj: Metadata object (could be dict, SQLAlchemy object, etc.)

        Returns:
            Plain Python dictionary
        """
        if metadata_obj is None:
            return {}
        if isinstance(metadata_obj, dict):
            return metadata_obj
        if hasattr(metadata_obj, '__dict__'):
            try:
                return {k: v for k, v in metadata_obj.__dict__.items() if not k.startswith('_')}
            except Exception:
                pass
        if hasattr(metadata_obj, 'keys') and hasattr(metadata_obj, '__getitem__'):
            try:
                return {k: metadata_obj[k] for k in metadata_obj.keys()}
            except Exception:
                pass
        logger.warning(f"Could not extract metadata from {type(metadata_obj)}")
        return {}

    def _embedding_to_pg_string(self, embedding: List[float]) -> str:
        """
        Convert a Python float list to PostgreSQL vector literal string.
        e.g. [0.1, 0.2, 0.3] -> '[0.1,0.2,0.3]'

        This bypasses the asyncpg :name::vector cast conflict by letting us
        do an explicit CAST($1::text AS vector) in raw SQL without SQLAlchemy
        named-param interpolation.
        """
        return "[" + ",".join(str(v) for v in embedding) + "]"

    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        db: AsyncSession,
        top_k: Optional[int] = None,
        doc_type: Optional[str] = None,
        department: Optional[str] = None,
        document_ids: Optional[List[UUID]] = None,
        document_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to query embedding with document filtering.

        KEY FIX: We build a raw SQL string for the vector similarity expression
        to avoid asyncpg's incompatibility with SQLAlchemy :name::vector params.
        The embedding is serialized to a PG vector literal and embedded directly
        into the SQL as a string cast — safe because it's a float list we control,
        not user input.

        Args:
            query_embedding: Vector embedding of search query
            db: Database session
            top_k: Number of results (default from config)
            doc_type: Filter by document type
            department: Filter by department
            document_ids: Filter by specific document IDs
            document_filter: Filter by document names

        Returns:
            List of chunk dictionaries with similarity scores
        """
        k = top_k or self.top_k

        logger.info(f"Vector search: top_k={k}, threshold={self.similarity_threshold}")
        logger.info(f"Query embedding dimensions: {len(query_embedding)}")

        try:
            # ---------------------------------------------------------------
            # THE FIX: Serialize embedding as a PG vector literal string.
            # asyncpg breaks on `<=> :embedding::vector` (named param + cast).
            # By inlining the literal we avoid all param binding for the vector.
            # This is safe — query_embedding is a float list from our own model.
            # ---------------------------------------------------------------
            embedding_literal = self._embedding_to_pg_string(query_embedding)

            # Build the similarity expression as a raw text column so SQLAlchemy
            # doesn't try to bind it as a named parameter
            similarity_expr = text(
                f"1 - (c.embedding <=> '{embedding_literal}'::vector)"
            ).columns()

            # Use raw SQL for the full query to avoid any ORM param binding issues
            # with the vector type — but keep Python-side filtering logic
            base_sql = f"""
                SELECT
                    c.id                    AS chunk_id,
                    c.document_id,
                    c.content,
                    c.chunk_type,
                    c.chunk_index,
                    c.page_numbers,
                    c.section_title,
                    c.token_count,
                    c.metadata              AS chunk_metadata,
                    d.id                    AS doc_id,
                    d.filename,
                    d.doc_type,
                    d.department,
                    d.metadata              AS doc_metadata,
                    1 - (c.embedding <=> '{embedding_literal}'::vector) AS similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.status = 'completed'
            """

            # Build WHERE clauses dynamically — using positional params for
            # all non-vector values (asyncpg-safe: :name for text/uuid is fine,
            # only the ::vector cast was the problem)
            conditions = []
            params = {}

            if doc_type:
                conditions.append("d.doc_type = :doc_type")
                params["doc_type"] = doc_type

            if department:
                conditions.append("d.department = :department")
                params["department"] = department

            if document_ids:
                # UUID list — cast each to string for asyncpg compatibility
                uuid_list = ", ".join(f"'{str(uid)}'" for uid in document_ids)
                conditions.append(f"d.id IN ({uuid_list})")

            if document_filter:
                like_clauses = " OR ".join(
                    f"d.filename ILIKE :doc_filter_{i}"
                    for i in range(len(document_filter))
                )
                conditions.append(f"({like_clauses})")
                for i, name in enumerate(document_filter):
                    params[f"doc_filter_{i}"] = f"%{name}%"
                logger.info(f"  Filtering to documents matching: {document_filter}")

            if conditions:
                base_sql += " AND " + " AND ".join(conditions)

            base_sql += f"""
                ORDER BY similarity DESC
                LIMIT {k * 2}
            """

            # Execute via raw SQL — asyncpg handles :name params for text fine
            result = await db.execute(text(base_sql), params)
            rows = result.mappings().all()

            # Format results
            chunks = []
            for row in rows:
                similarity = float(row["similarity"])
                if similarity < self.similarity_threshold:
                    continue

                # Safely parse metadata JSON columns
                chunk_metadata = self._safe_extract_metadata(row["chunk_metadata"])
                doc_metadata = self._safe_extract_metadata(row["doc_metadata"])

                chunk_dict = {
                    'chunk_id':       str(row["chunk_id"]),
                    'id':             str(row["chunk_id"]),
                    'document_id':    str(row["document_id"]),
                    'document_name':  row["filename"],
                    'content':        row["content"],
                    'chunk_type':     row["chunk_type"],
                    'chunk_index':    row["chunk_index"],
                    'page_numbers':   row["page_numbers"],
                    'section_title':  row["section_title"],
                    'token_count':    row["token_count"],
                    'similarity_score': similarity,
                    'chunk_metadata': chunk_metadata,
                    'metadata': {
                        'document_title': row["filename"],
                        'doc_type':       row["doc_type"],
                        'department':     row["department"],
                        **chunk_metadata,
                    },
                }
                chunks.append(chunk_dict)

            logger.info(
                f"Vector search found {len(chunks)} chunks "
                f"above threshold {self.similarity_threshold}"
            )
            return chunks[:k]

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}", exc_info=True)
            # CRITICAL: rollback so the session isn't left in an aborted-transaction
            # state — without this, every subsequent query on the same session will
            # also fail with InFailedSQLTransactionError
            try:
                await db.rollback()
                logger.info("Vector search: session rolled back after error")
            except Exception as rb_err:
                logger.warning(f"Rollback also failed: {rb_err}")
            return []

    async def search_by_document(
        self,
        query_embedding: List[float],
        document_id: UUID,
        db: AsyncSession,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search within a specific document only."""
        return await self.search_similar_chunks(
            query_embedding=query_embedding,
            db=db,
            top_k=top_k,
            document_ids=[document_id],
        )

    async def get_chunk_neighbors(
        self,
        chunk_id: UUID,
        db: AsyncSession,
        n_before: int = 1,
        n_after: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring chunks for context expansion.

        Args:
            chunk_id: Target chunk ID
            db: Database session
            n_before: Number of chunks before
            n_after: Number of chunks after

        Returns:
            List of neighboring chunks in order
        """
        try:
            # Get target chunk — plain ORM query, no vector ops, no issue
            result = await db.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                .where(Chunk.id == chunk_id)
            )
            row = result.first()

            if not row:
                logger.warning(f"Chunk {chunk_id} not found for neighbor expansion")
                return []

            target_chunk, document = row
            target_index = target_chunk.chunk_index

            # Get neighbors
            neighbor_query = (
                select(Chunk)
                .where(
                    Chunk.document_id == target_chunk.document_id,
                    Chunk.chunk_index >= target_index - n_before,
                    Chunk.chunk_index <= target_index + n_after,
                )
                .order_by(Chunk.chunk_index)
            )

            result = await db.execute(neighbor_query)
            chunks = result.scalars().all()

            neighbor_chunks = []
            for chunk in chunks:
                chunk_metadata = self._safe_extract_metadata(chunk.chunk_metadata)
                chunk_dict = {
                    'chunk_id':      str(chunk.id),
                    'id':            str(chunk.id),
                    'document_id':   str(chunk.document_id),
                    'document_name': document.filename,
                    'content':       chunk.content,
                    'chunk_type':    chunk.chunk_type,
                    'chunk_index':   chunk.chunk_index,
                    'page_numbers':  chunk.page_numbers,
                    'section_title': chunk.section_title,
                    'token_count':   chunk.token_count,
                    'is_target':     chunk.id == chunk_id,
                    'chunk_metadata': chunk_metadata,
                    'metadata': {
                        'document_title': document.filename,
                        **chunk_metadata,
                    },
                }
                neighbor_chunks.append(chunk_dict)

            return neighbor_chunks

        except Exception as e:
            logger.error(f"Failed to get chunk neighbors: {str(e)}", exc_info=True)
            try:
                await db.rollback()
            except Exception:
                pass
            return []


# Global instance
vector_search_service = VectorSearchService()

__all__ = ['VectorSearchService', 'vector_search_service']
