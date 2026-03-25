"""
Complete retrieval pipeline orchestrator.
Coordinates query processing, hybrid search, reranking, and context assembly.
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import settings
from app.services.retrieval.query_processor import query_processor
from app.services.retrieval.hybrid_search import hybrid_search_service
from app.services.retrieval.reranker import reranking_service

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.generation.context_manager import ConversationContext


class RetrievalPipeline:
    """
    Complete retrieval pipeline for RAG system.
    Orchestrates: Query Processing → Hybrid Search → Reranking → Context Assembly
    """
    
    def __init__(self):
        self.max_context_tokens = settings.max_context_tokens
        self.rerank_enabled = True
    
    async def retrieve(
        self,
        query: str,
        db: AsyncSession,
        top_k: Optional[int] = None,
        doc_type: Optional[str] = None,
        department: Optional[str] = None,
        include_context: bool = True,
        conversation_context: Optional['ConversationContext'] = None,
        force_global: bool = False
    ) -> Dict[str, Any]:
        """
        Complete retrieval pipeline with conversation context awareness and dynamic scoping.
        
        Args:
            query: User's search query
            db: Database session
            top_k: Number of final chunks to return
            doc_type: Filter by document type
            department: Filter by department
            include_context: Whether to expand with neighboring chunks
            conversation_context: Conversation context for document scoping
            force_global: Force global search (skip scoping)
            
        Returns:
            Dictionary with retrieved chunks and metadata
        """
        logger.info(f"🔍 Starting retrieval pipeline for query: '{query[:50]}...'")
        
        # Step 1: Process and enhance query
        logger.info("Step 1: Processing query...")
        processed_query = query_processor.process_query(query)
        
        logger.info(f"  Intent: {processed_query['intent']}")
        logger.info(f"  Complexity: {processed_query['complexity']}")
        
        # Step 2: Apply conversation context - BOOST not FILTER
        boost_documents = []
        use_scoped_search = False
        document_filter_for_boosting = None

        if conversation_context and not force_global:
            logger.info("Step 2: Applying conversation context...")
            
            # Get documents to BOOST (not hard filter)
            document_filter_for_boosting = conversation_context.get_document_filter()
            
            if document_filter_for_boosting:
                boost_documents = document_filter_for_boosting
                use_scoped_search = True
                logger.info(f"  ⬆️ Will boost results from: {boost_documents}")
            else:
                logger.info("  🌐 No document boosting - global search")
        else:
            logger.info("Step 2: Global search (no context or forced)")

        # Step 3: Determine search strategy
        use_semantic = True
        use_keyword = True

        if query_processor.should_use_semantic_only(processed_query):
            logger.info("  Strategy: Semantic only")
            use_keyword = False
        elif query_processor.should_use_keyword_only(processed_query):
            logger.info("  Strategy: Keyword only")
            use_semantic = False
        else:
            logger.info("  Strategy: Hybrid (semantic + keyword)")

        # Step 4: Execute search with dynamic scoping
        initial_top_k = settings.scoped_search_top_k if use_scoped_search else settings.global_search_top_k

        logger.info(f"Step 3: Performing {'scoped' if use_scoped_search else 'global'} hybrid search (k={initial_top_k})...")

        if include_context:
            search_results = await hybrid_search_service.search_with_context_expansion(
                query=query,
                db=db,
                top_k=initial_top_k,
                expand_neighbors=True,
                document_filter=None,  # CRITICAL: No hard filtering
                boost_documents=boost_documents  # Only boosting
            )
        else:
            search_results = await hybrid_search_service.search(
                query=query,
                db=db,
                top_k=initial_top_k,
                use_semantic=use_semantic,
                use_keyword=use_keyword,
                doc_type=doc_type,
                department=department,
                document_filter=None,  # CRITICAL: No hard filtering
                boost_documents=boost_documents  # Only boosting
            )

        logger.info(f"  Initial search: {len(search_results)} chunks")

        # Step 5: DYNAMIC EXPANSION - Check if we need global search
        should_expand = False

        if use_scoped_search and conversation_context and settings.enable_dynamic_scope:
            should_expand = conversation_context.should_expand_search(search_results)
            
            if should_expand:
                logger.info("🔄 Scoped search insufficient - performing GLOBAL search...")
                
                global_results = await hybrid_search_service.search(
                    query=query,
                    db=db,
                    top_k=settings.global_search_top_k,
                    use_semantic=use_semantic,
                    use_keyword=use_keyword,
                    doc_type=doc_type,
                    department=department,
                    document_filter=None,
                    boost_documents=[]  # No boosting in global search
                )
                
                logger.info(f"  Global search: {len(global_results)} chunks")
                
                # Merge results: prioritize scoped but include global
                seen_ids = set()
                merged_results = []
                
                # Add scoped results first (boosted)
                for chunk in search_results:
                    chunk_id = chunk.get('chunk_id')
                    if chunk_id and chunk_id not in seen_ids:
                        chunk['source_type'] = 'scoped'
                        merged_results.append(chunk)
                        seen_ids.add(chunk_id)
                
                # Add global results
                for chunk in global_results:
                    chunk_id = chunk.get('chunk_id')
                    if chunk_id and chunk_id not in seen_ids:
                        chunk['source_type'] = 'global'
                        merged_results.append(chunk)
                        seen_ids.add(chunk_id)
                
                search_results = merged_results
                logger.info(f"  Merged results: {len(search_results)} total chunks")

        # Step 6: Rerank results
        reranked_results = []

        # Decide if reranking is needed
        should_rerank = (
            self.rerank_enabled and
            len(search_results) > 1 and
            len(search_results) >= 3
        )

        # Perform reranking if needed
        if should_rerank:
            logger.info("Step 4: Reranking results...")
            try:
                reranked_results = await reranking_service.rerank(

        # Skip reranking for very high confidence results
        # if should_rerank and len(search_results) >= 3:
        #     # Check if top 3 results are already very confident
        #     top_scores = []
        #     for chunk in search_results[:3]:
        #         score = (
        #             chunk.get('similarity_score', 0) or 
        #             chunk.get('fused_score', 0) or 
        #             0
        #         )
        #         top_scores.append(score)
            
        #     # If all top 3 have score > 0.75, skip reranking
        #     if all(score > 0.75 for score in top_scores):
        #         logger.info("Step 4: Skipping reranking (high confidence results - all top 3 above 0.75)")
        #         reranked_results = search_results[:top_k or settings.rerank_top_k]
        #         should_rerank = False

        # Perform reranking if needed
                    query=query,
                    chunks=search_results,
                    top_n=top_k or settings.rerank_top_k
                )
                
                stats = reranking_service.calculate_score_statistics(reranked_results)
                logger.info(f"  Rerank scores: min={stats['min_score']:.3f}, max={stats['max_score']:.3f}, avg={stats['avg_score']:.3f}")
            except Exception as e:
                logger.error(f"Reranking failed: {str(e)}, using original order")
                reranked_results = search_results[:top_k or settings.rerank_top_k]
        else:
            if not reranked_results:  # If not already set by smart skip
                logger.info("Step 4: Skipping reranking (disabled or insufficient results)")
                reranked_results = search_results[:top_k or settings.rerank_top_k]

        logger.info(f"  After reranking: {len(reranked_results)} chunks")
        
        # Step 7: Assemble final context
        logger.info("Step 5: Assembling context...")
        final_context = self._assemble_context(
            chunks=reranked_results,
            processed_query=processed_query
        )
        
        logger.info(f"✅ Retrieval complete: {len(final_context['chunks'])} chunks, {final_context['total_tokens']} tokens")
        
        return {
            'query': query,
            'processed_query': processed_query,
            'chunks': final_context['chunks'],
            'context_text': final_context['context_text'],
            'total_tokens': final_context['total_tokens'],
            'sources': final_context['sources'],
            'retrieval_metadata': {
                'search_strategy': 'hybrid' if (use_semantic and use_keyword) else ('semantic' if use_semantic else 'keyword'),
                'total_retrieved': len(search_results),
                'after_reranking': len(reranked_results),
                'final_chunks': len(final_context['chunks']),
                'intent': processed_query['intent'],
                'complexity': processed_query['complexity'],
                'used_scoped_search': use_scoped_search,
                'expanded_to_global': should_expand,
                'boosted_documents': boost_documents,
                'search_type': 'global' if should_expand else ('scoped' if use_scoped_search else 'global')
            }
        }
    
    def _safe_get_metadata(self, chunk: Dict[str, Any]) -> dict:
        """
        Safely extract metadata from chunk, handling SQLAlchemy objects.
        
        Args:
            chunk: Chunk dictionary (may contain SQLAlchemy objects)
            
        Returns:
            Plain dictionary with metadata
        """
        metadata = {}
        
        # Try different metadata keys
        for meta_key in ['chunk_metadata', 'metadata', 'doc_metadata']:
            if meta_key in chunk:
                meta_obj = chunk[meta_key]
                
                # Handle different types
                if isinstance(meta_obj, dict):
                    metadata = meta_obj
                    break
                elif hasattr(meta_obj, '__dict__'):
                    # SQLAlchemy object - convert to dict
                    try:
                        metadata = {k: v for k, v in meta_obj.__dict__.items() if not k.startswith('_')}
                        break
                    except Exception as e:
                        logger.debug(f"Could not extract __dict__ from metadata: {e}")
                        pass
                elif hasattr(meta_obj, 'keys') and hasattr(meta_obj, '__getitem__'):
                    # Mapping-like object
                    try:
                        metadata = {k: meta_obj[k] for k in meta_obj.keys()}
                        break
                    except Exception as e:
                        logger.debug(f"Could not extract keys from metadata: {e}")
                        pass
        
        # Fallback: Extract from chunk itself if metadata is empty
        if not metadata:
            metadata = {
                'document_title': chunk.get('document_name') or chunk.get('filename', 'Unknown'),
                'doc_type': chunk.get('doc_type'),
                'department': chunk.get('department'),
            }
        
        return metadata
    
    def _prioritize_chunks(
        self,
        chunks: List[Dict[str, Any]],
        processed_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Prioritize chunks based on type and relevance.
        
        Priority order:
        1. Tables (for financial/data queries)
        2. High rerank score chunks
        3. Exact matches
        4. Regular text chunks
        """
        if not chunks:
            return []
        
        intent = processed_query.get('intent', 'general')
        entities = processed_query.get('entities', {})
        has_financial_entities = 'amount' in entities or 'financial' in intent.lower()
        
        def get_priority(chunk: Dict[str, Any]) -> tuple:
            # Higher number = higher priority (sort descending)
            priority = 0
            
            # Boost tables for financial/analytical queries
            chunk_type = chunk.get('chunk_type', 'text')
            if chunk_type == 'table' and (intent in ['financial', 'analytical'] or has_financial_entities):
                priority += 1000
            
            # Use rerank score as primary sort
            rerank_score = chunk.get('rerank_score', chunk.get('fused_score', chunk.get('similarity_score', 0)))
            
            return (priority, rerank_score)
        
        # Sort by priority
        try:
            sorted_chunks = sorted(chunks, key=get_priority, reverse=True)
        except Exception as e:
            logger.warning(f"Error sorting chunks: {e}, returning unsorted")
            sorted_chunks = chunks
        
        return sorted_chunks
    
    def _assemble_context(
        self,
        chunks: List[Dict[str, Any]],
        processed_query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assemble final context with intelligent truncation.
        """
        context_parts = []
        total_tokens = 0
        sources = []
        final_chunks = []
        
        if not chunks:
            logger.warning("No chunks to assemble context from")
            return {
                'chunks': [],
                'context_text': '',
                'total_tokens': 0,
                'sources': []
            }
        
        sorted_chunks = self._prioritize_chunks(chunks, processed_query)
        
        # max_tokens = min(self.max_context_tokens, 6000)
        max_tokens = self.max_context_tokens  # use the configured 12,000 — Gemini handles it

        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.get('token_count', 0)
            
            # if chunk_tokens > 800:
            #     chunk['content'] = chunk['content'][:3200]
            #     chunk_tokens = 800

            if chunk_tokens > 1200:
                chunk['content'] = chunk['content'][:4800]
                chunk_tokens = 1200
            
            if total_tokens + chunk_tokens > max_tokens:
                logger.warning(f"Reached token limit ({max_tokens}), stopping context assembly")
                break
            
            chunk_text = self._format_chunk_for_context(chunk)
            context_parts.append(chunk_text)
            
            chunk_metadata = self._safe_get_metadata(chunk)
            
            source_info = {
                'document': chunk_metadata.get('document_title', chunk.get('document_name', 'Unknown Document')),
                'page': chunk.get('page_numbers', [None])[0] if chunk.get('page_numbers') else None,
                'section': chunk.get('section_title'),
                'chunk_id': str(chunk.get('chunk_id', chunk.get('id', '')))
            }
            
            if source_info not in sources:
                sources.append(source_info)
            
            final_chunks.append(chunk)
            total_tokens += chunk_tokens
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Assembled context: {len(final_chunks)} chunks, {total_tokens} tokens, {len(sources)} sources")
        
        return {
            'chunks': final_chunks,
            'context_text': context_text,
            'total_tokens': total_tokens,
            'sources': sources
        }
    
    def _format_chunk_for_context(self, chunk: Dict[str, Any]) -> str:
        """
        Format chunk with metadata for LLM context.
        
        Returns:
            Formatted chunk text with source information
        """
        # ✅ Use safe metadata extraction
        chunk_metadata = self._safe_get_metadata(chunk)
        
        # Build header
        header_parts = []
        
        # Try to get document title from multiple sources
        doc_title = (
            chunk_metadata.get('document_title') or 
            chunk_metadata.get('filename') or
            chunk.get('document_name') or 
            chunk.get('filename')
        )
        
        if doc_title:
            header_parts.append(f"Document: {doc_title}")
        
        section_title = chunk.get('section_title')
        if section_title:
            header_parts.append(f"Section: {section_title}")
        
        page_numbers = chunk.get('page_numbers', [])
        if page_numbers and isinstance(page_numbers, list) and len(page_numbers) > 0:
            if len(page_numbers) == 1:
                header_parts.append(f"Page: {page_numbers[0]}")
            else:
                header_parts.append(f"Pages: {page_numbers[0]}-{page_numbers[-1]}")
        
        chunk_type = chunk.get('chunk_type', 'text')
        if chunk_type and chunk_type != 'text':
            header_parts.append(f"Type: {chunk_type.title()}")
        
        # Get content
        content = chunk.get('content', '')
        
        # Format final output
        if header_parts:
            header = " | ".join(header_parts)
            return f"[{header}]\n{content}"
        else:
            return content
    
    async def retrieve_from_document(
        self,
        query: str,
        document_id: str,
        db: AsyncSession,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve from a specific document only.
        Useful for document-specific questions.
        """
        from uuid import UUID
            
        logger.info(f"Retrieving from document {document_id}")
        
        # Process query
        processed_query = query_processor.process_query(query)
        
        # Search within document
        from app.services.embedding.embedding_service import embedding_service
        from app.services.retrieval.vector_search import vector_search_service
        
        query_embedding = await embedding_service.embed_query(query)
        
        search_results = await vector_search_service.search_by_document(
            query_embedding=query_embedding,
            document_id=UUID(document_id),
            db=db,
            top_k=top_k or settings.retrieval_top_k
        )
        
        # Rerank
        if self.rerank_enabled and search_results:
            reranked_results = await reranking_service.rerank(
                query=query,
                chunks=search_results,
                top_n=top_k or settings.rerank_top_k
            )
        else:
            reranked_results = search_results
        
        # Assemble context
        final_context = self._assemble_context(
            chunks=reranked_results,
            processed_query=processed_query
        )
        
        return {
            'query': query,
            'document_id': document_id,
            'chunks': final_context['chunks'],
            'context_text': final_context['context_text'],
            'total_tokens': final_context['total_tokens'],
            'sources': final_context['sources']
        }


# Global instance
retrieval_pipeline = RetrievalPipeline()

__all__ = ['RetrievalPipeline', 'retrieval_pipeline']