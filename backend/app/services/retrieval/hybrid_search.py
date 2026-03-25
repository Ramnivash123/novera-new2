"""
Hybrid search combining semantic (vector) and keyword (BM25) search.
Implements Reciprocal Rank Fusion for result merging.

FIX NOTES:
- When vector_search fails and returns [], the DB session may be in an
  aborted-transaction state (asyncpg raises InFailedSQLTransactionError on
  any subsequent query in the same transaction). We now explicitly rollback
  the session between the semantic and keyword search steps so keyword search
  always runs on a clean transaction.
- Added a helper _safe_rollback() to avoid swallowing secondary errors.
- Debug count query wrapped in its own try/except so it never blocks search.
"""
from typing import List, Dict, Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from loguru import logger

from app.core.config import settings
from app.services.retrieval.vector_search import vector_search_service
from app.services.retrieval.keyword_search import keyword_search_service
from app.services.embedding.embedding_service import embedding_service


class HybridSearchService:
    """
    Combines semantic and keyword search for optimal retrieval.
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    """

    def __init__(self):
        self.alpha = settings.hybrid_alpha      # 0.7 = 70% semantic weight
        self.top_k = settings.retrieval_top_k
        self.rrf_k = 60                         # RRF constant

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _safe_rollback(self, db: AsyncSession, context: str = "") -> None:
        """
        Attempt a session rollback without raising.
        Ensures the next query in the same session starts on a clean transaction.
        """
        try:
            await db.rollback()
            logger.debug(f"Session rolled back cleanly [{context}]")
        except Exception as e:
            logger.warning(f"Rollback attempt failed [{context}]: {e}")

    async def _count_available_chunks(self, db: AsyncSession) -> int:
        """
        Debug helper: count completed chunks in DB.
        Isolated in its own try/except so it never breaks the main search.
        """
        try:
            from app.models.document import Document, Chunk as ChunkModel
            count_query = (
                select(func.count(ChunkModel.id))
                .join(Document)
                .where(Document.status == "completed")
            )
            result = await db.execute(count_query)
            return result.scalar() or 0
        except Exception as e:
            logger.warning(f"Could not count chunks (non-fatal): {e}")
            await self._safe_rollback(db, "chunk count fallback")
            return -1   # -1 signals "unknown"

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        db: AsyncSession,
        top_k: Optional[int] = None,
        use_semantic: bool = True,
        use_keyword: bool = True,
        doc_type: Optional[str] = None,
        department: Optional[str] = None,
        document_filter: Optional[List[str]] = None,
        boost_documents: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with optional document scoping and boosting.

        Execution order:
          1. (debug) count available chunks
          2. Semantic (vector) search   — rollback session on failure
          3. Keyword (FTS) search       — rollback session on failure
          4. RRF fusion + sort
        """
        k = top_k or self.top_k

        logger.info(f"Hybrid search: query='{query[:50]}...', k={k}")
        logger.info(f"  Filters: doc_type={doc_type}, department={department}")
        logger.info(f"  Document filter: {document_filter}")
        logger.info(f"  Boost documents: {boost_documents}")

        # Debug: total chunk count (best-effort, non-blocking)
        total_available = await self._count_available_chunks(db)
        logger.info(f"  Total chunks available in DB: {total_available}")

        results: Dict[str, Dict[str, Any]] = {}

        # ------------------------------------------------------------------
        # Step 1 — Semantic search
        # ------------------------------------------------------------------
        if use_semantic:
            logger.info("Running semantic search...")
            try:
                query_embedding = await embedding_service.embed_query(query)

                semantic_results = await vector_search_service.search_similar_chunks(
                    query_embedding=query_embedding,
                    db=db,
                    top_k=k * 2,
                    doc_type=doc_type,
                    department=department,
                    document_filter=document_filter,
                )

                for idx, chunk in enumerate(semantic_results, 1):
                    chunk_id = chunk['chunk_id']
                    if chunk_id not in results:
                        results[chunk_id] = chunk.copy()
                    results[chunk_id]['semantic_rank'] = idx
                    results[chunk_id]['semantic_score'] = chunk['similarity_score']

                logger.info(f"Semantic search returned {len(semantic_results)} results")

            except Exception as e:
                logger.error(f"Semantic search step failed: {e}", exc_info=True)
                # -------------------------------------------------------
                # KEY FIX: rollback before keyword search so the session
                # isn't stuck in an aborted-transaction state.
                # Without this, keyword search always gets
                # InFailedSQLTransactionError regardless of its own SQL.
                # -------------------------------------------------------
                await self._safe_rollback(db, "post-semantic-failure")

        # ------------------------------------------------------------------
        # Step 2 — Keyword search
        # ------------------------------------------------------------------
        if use_keyword:
            logger.info("Running keyword search...")
            try:
                keyword_results = await keyword_search_service.search_keywords(
                    query=query,
                    db=db,
                    top_k=k,
                    doc_type=doc_type,
                    department=department,
                    document_filter=document_filter,
                )

                for idx, chunk in enumerate(keyword_results, 1):
                    chunk_id = chunk['chunk_id']
                    if chunk_id not in results:
                        results[chunk_id] = chunk.copy()
                    results[chunk_id]['keyword_rank'] = idx
                    results[chunk_id]['keyword_score'] = chunk.get('keyword_score', 0)

                logger.info(f"Keyword search returned {len(keyword_results)} results")

            except Exception as e:
                logger.error(f"Keyword search step failed: {e}", exc_info=True)
                await self._safe_rollback(db, "post-keyword-failure")

        # ------------------------------------------------------------------
        # Step 3 — RRF fusion + sort
        # ------------------------------------------------------------------
        logger.info("Fusing results...")
        fused_results = self._reciprocal_rank_fusion(results, boost_documents)

        sorted_results = sorted(
            fused_results.values(),
            key=lambda x: x['fused_score'],
            reverse=True,
        )[:k]

        logger.info(f"Hybrid search completed: {len(sorted_results)} final results")
        return sorted_results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        results: Dict[str, Dict[str, Any]],
        boost_documents: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion with soft document boosting.

        RRF score = Σ weight_i / (rrf_k + rank_i)
        Boost: multiply score by boost_factor for specified documents.
        """
        boost_factor = 1.3

        for chunk_id, chunk in results.items():
            rrf_score = 0.0

            if 'semantic_rank' in chunk:
                rrf_score += self.alpha * (1.0 / (self.rrf_k + chunk['semantic_rank']))

            if 'keyword_rank' in chunk:
                rrf_score += (1 - self.alpha) * (1.0 / (self.rrf_k + chunk['keyword_rank']))

            if boost_documents:
                doc_name = (
                    chunk.get('document_name')
                    or chunk.get('metadata', {}).get('document_title', '')
                )
                is_boosted = any(
                    bd.lower() in doc_name.lower() or doc_name.lower() in bd.lower()
                    for bd in boost_documents
                )
                if is_boosted:
                    rrf_score *= boost_factor
                    chunk['boosted'] = True
                    logger.debug(f"Boosted chunk from {doc_name}: {rrf_score:.4f}")
                else:
                    chunk['boosted'] = False

            chunk['fused_score'] = rrf_score
            chunk['retrieval_method'] = []
            if 'semantic_rank' in chunk:
                chunk['retrieval_method'].append('semantic')
            if 'keyword_rank' in chunk:
                chunk['retrieval_method'].append('keyword')

        return results

    # ------------------------------------------------------------------
    # Context expansion
    # ------------------------------------------------------------------

    async def search_with_context_expansion(
        self,
        query: str,
        db: AsyncSession,
        top_k: Optional[int] = None,
        expand_neighbors: bool = True,
        document_filter: Optional[List[str]] = None,
        boost_documents: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with automatic neighboring-chunk context expansion.

        Args:
            query: Search query
            db: Database session
            top_k: Number of base results before expansion
            expand_neighbors: Whether to pull n±1 neighbor chunks
            document_filter: Restrict search to these document names
            boost_documents: Soft-boost these document names in RRF

        Returns:
            List of chunks (original + expanded neighbors), de-duplicated
        """
        initial_results = await self.search(
            query=query,
            db=db,
            top_k=top_k,
            document_filter=document_filter,
            boost_documents=boost_documents,
        )

        if not expand_neighbors:
            return initial_results

        expanded_results: List[Dict[str, Any]] = []
        seen_chunk_ids: set = set()

        for result in initial_results[:5]:
            chunk_id_raw = result.get('chunk_id') or result.get('id')

            if not chunk_id_raw:
                logger.warning(f"Chunk missing ID, skipping expansion: {result}")
                continue

            try:
                chunk_id = UUID(str(chunk_id_raw))
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid chunk ID format: {chunk_id_raw} — {e}")
                continue

            neighbors = await vector_search_service.get_chunk_neighbors(
                chunk_id=chunk_id,
                db=db,
                n_before=1,
                n_after=1,
            )

            for neighbor in neighbors:
                nid = neighbor['chunk_id']
                if nid in seen_chunk_ids:
                    continue

                neighbor['is_expanded_context'] = not neighbor['is_target']
                neighbor['parent_chunk_id'] = (
                    result['chunk_id'] if not neighbor['is_target'] else None
                )

                if neighbor['is_target']:
                    neighbor.update({
                        'fused_score':      result['fused_score'],
                        'semantic_score':   result.get('semantic_score'),
                        'keyword_score':    result.get('keyword_score'),
                        'retrieval_method': result['retrieval_method'],
                        'boosted':          result.get('boosted', False),
                    })

                expanded_results.append(neighbor)
                seen_chunk_ids.add(nid)

        # Append remaining results that weren't in the top-5 expansion set
        for result in initial_results[5:]:
            if result['chunk_id'] not in seen_chunk_ids:
                result['is_expanded_context'] = False
                expanded_results.append(result)
                seen_chunk_ids.add(result['chunk_id'])

        logger.info(
            f"Context expansion: {len(initial_results)} → {len(expanded_results)} chunks"
        )
        return expanded_results


# Global instance
hybrid_search_service = HybridSearchService()

__all__ = ['HybridSearchService', 'hybrid_search_service']
