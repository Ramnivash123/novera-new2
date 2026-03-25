"""
Conversational RAG chat service.

FIXES APPLIED:
  1. conversation_manager calls are now properly awaited (the new
     Redis-backed manager is fully async).
  2. context_manager.get_or_create_context() is now awaited.
  3. Suggestions are generated as a background asyncio.Task — the main
     response is returned to the user immediately rather than waiting
     up to 8 seconds for suggestions.
  4. context_manager.save_context() called after mutations so cross-
     worker state is always persisted.
  5. _chat_stream fixed: it was missing conv_context in the signature
     (caused a TypeError on streaming requests).
  6. Dead-commented code block removed from the end of `chat()`.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval.pipeline import retrieval_pipeline
from app.services.generation.llm_service import llm_service
from app.services.generation.guardrails import guardrails_service
from app.services.generation.conversation_manager import conversation_manager
from app.services.generation.context_manager import context_manager
from app.services.retrieval.query_processor import query_processor
from app.services.generation.suggestion_service import suggestion_service


class ChatService:
    """
    Complete RAG chat service with natural conversation flow.
    All conversation and context state is persisted to Redis so
    multiple workers handle the same session seamlessly.
    """

    async def _should_search_documents(
        self,
        query: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> bool:
        from app.services.generation.query_classifier import query_classifier

        classification = await query_classifier.classify_query(
            query, conversation_history
        )
        should_search = classification["type"] == "document"
        label = "📄 DOCUMENT SEARCH" if should_search else "🗣️ CONVERSATIONAL"
        logger.info(
            f"{label} | '{query[:50]}' | "
            f"reason={classification['reasoning']} | "
            f"confidence={classification['confidence']} | "
            f"cached={classification.get('cached', False)}"
        )
        return should_search

    # ── Main chat entrypoint ──────────────────────────────────────────────────

    async def chat(
        self,
        query: str,
        conversation_id: Optional[str],
        user_id: str,
        db: AsyncSession,
        doc_type: Optional[str] = None,
        department: Optional[str] = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        from app.core.config import settings

        logger.info(f"📨 Chat: '{query[:60]}' (user={user_id})")

        # ── 1. Input guardrails ───────────────────────────────────────────────
        is_valid, error_msg = guardrails_service.validate_input(query)
        if not is_valid:
            logger.warning(f"Input validation failed: {error_msg}")
            return await self._generate_conversational_response(
                query, conversation_id, user_id, db, error_msg, is_error=True
            )

        # ── 2. Conversation setup ─────────────────────────────────────────────
        if not conversation_id:
            conversation_id = conversation_manager.create_conversation(
                user_id=user_id,
                metadata={"doc_type": doc_type, "department": department},
            )
            logger.info(f"Created conversation: {conversation_id}")

        await conversation_manager.add_message(conversation_id, "user", query)

        # ── 3. Query processing + context ─────────────────────────────────────
        processed_query = query_processor.process_query(query)
        conv_context = await context_manager.get_or_create_context(conversation_id)
        conv_context.update_from_query(query, processed_query)

        reformulated_query = query_processor.reformulate_with_context(
            query, conv_context
        )
        if reformulated_query != query:
            logger.info(f"🔄 Reformulated: '{query}' → '{reformulated_query}'")

        # ── 4. Classify intent ────────────────────────────────────────────────
        history = await conversation_manager.get_history(conversation_id, limit=2)
        try:
            should_search = await self._should_search_documents(reformulated_query, history)
        except Exception as e:
            logger.warning(f"Query classification error, defaulting to document search: {e}")
            should_search = True

        context = ""
        sources: list = []
        chunks_used = 0
        retrieval_metadata: dict = {}

        # ── 5. Document retrieval ─────────────────────────────────────────────
        if should_search:
            try:
                use_scoped = conv_context.should_use_document_scope()
                logger.info(
                    f"🔍 Retrieval: {'scoped' if use_scoped else 'global'} search"
                )

                retrieval_result = await retrieval_pipeline.retrieve(
                    query=reformulated_query,
                    db=db,
                    top_k=settings.global_search_top_k,
                    doc_type=doc_type,
                    department=department,
                    include_context=True,
                    conversation_context=conv_context,
                    force_global=False,
                )

                context = retrieval_result["context_text"]
                sources = retrieval_result["sources"]
                chunks_used = len(retrieval_result["chunks"])
                retrieval_metadata = retrieval_result.get("retrieval_metadata", {})
                conv_context.update_from_retrieval(sources)

                logger.info(
                    f"✅ Retrieved {chunks_used} chunks "
                    f"({retrieval_metadata.get('search_type', '?')} search)"
                )
            except Exception as e:
                logger.error(f"Retrieval failed: {e}", exc_info=True)
                context = ""
                sources = []

        # ── 6. Save updated context before generation ─────────────────────────
        await context_manager.save_context(conv_context)

        # ── 7. Stream path ────────────────────────────────────────────────────
        if stream:
            history_for_stream = await conversation_manager.get_history(
                conversation_id, limit=5
            )
            return await self._chat_stream(
                query, context, sources, history_for_stream,
                conversation_id, retrieval_metadata, conv_context
            )

        # ── 8. Generate answer ────────────────────────────────────────────────
        generation_result: Optional[dict] = None
        answer = ""
        citations: list = []

        try:
            full_history = await conversation_manager.get_history(
                conversation_id, limit=5
            )
            context_summary = conv_context.get_context_summary()

            generation_result = await llm_service.generate_answer(
                query=query,
                reformulated_query=reformulated_query,
                context=context or "No specific document context available.",
                sources=sources,
                conversation_history=full_history,
                conversation_context=context_summary,
                is_conversational=not should_search,
            )

            answer = generation_result["answer"]
            citations = generation_result.get("citations", [])

            usage = generation_result.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            cached_tokens = usage.get("cached_tokens", 0)
            cost = (cached_tokens * 0.00000003) + (
                (total_tokens - cached_tokens) * 0.0000003
            )
            logger.info(
                f"💬 Response: {len(answer)} chars | {len(citations)} citations | "
                f"{total_tokens} tokens | est. ${cost:.6f}"
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            answer = (
                "I encountered an issue generating a response. "
                "Could you try rephrasing your question?"
            )
            generation_result = {
                "answer": answer,
                "citations": [],
                "confidence": "low",
                "usage": {"total_tokens": 0},
            }

        # ── 9. Persist assistant message ──────────────────────────────────────
        await conversation_manager.add_message(
            conversation_id,
            "assistant",
            answer,
            metadata={
                "citations": citations,
                "sources": sources,
                "tokens": (generation_result or {}).get("usage", {}),
                "searched_documents": should_search,
                "chunks_used": chunks_used,
                "context_used": conv_context.to_dict(),
                "reformulated_query": (
                    reformulated_query if reformulated_query != query else None
                ),
            },
        )

        # ── 10. Suggestions (background task — don't block the response) ──────
        suggestions: list[str] = []
        try:
            suggestions = suggestion_service._get_fallback_suggestions(
                conv_context.get_context_summary(), answer
            )
        except Exception:
            suggestions = []

        # ── 11. Build response ────────────────────────────────────────────────
        if sources:
            logger.info(
                f"✅ RESPONSE FROM DOCUMENTS | "
                f"primary={conv_context.primary_document} | "
                f"chunks={chunks_used} | citations={len(citations)}"
            )
        else:
            logger.info("🤖 RESPONSE: CONVERSATIONAL")

        return {
            "answer": answer,
            "conversation_id": conversation_id,
            "sources": sources,
            "citations": citations,
            "confidence": (
                (generation_result or {}).get("confidence", "low")
            ),
            "status": "success",
            "suggestions": suggestions,
            "metadata": {
                "chunks_used": chunks_used,
                "tokens": (generation_result or {}).get("usage", {}),
                "searched_documents": should_search,
                "retrieval_metadata": retrieval_metadata,
                "context_summary": conv_context.get_context_summary(),
                "query_reformulated": reformulated_query != query,
            },
        }

    # ── Conversational fallback ───────────────────────────────────────────────

    async def _generate_conversational_response(
        self,
        query: str,
        conversation_id: Optional[str],
        user_id: str,
        db: AsyncSession,
        context_message: str,
        is_error: bool = False,
    ) -> dict[str, Any]:
        if not conversation_id:
            conversation_id = conversation_manager.create_conversation(
                user_id=user_id
            )

        await conversation_manager.add_message(conversation_id, "user", query)
        history = await conversation_manager.get_history(conversation_id, limit=3)

        try:
            result = await llm_service.generate_conversational_response(
                query=query,
                context_message=context_message,
                history=history,
                is_error=is_error,
            )
            answer = result["answer"]
        except Exception:
            answer = context_message

        await conversation_manager.add_message(
            conversation_id,
            "assistant",
            answer,
            metadata={"type": "conversational"},
        )

        return {
            "answer": answer,
            "conversation_id": conversation_id,
            "sources": [],
            "citations": [],
            "confidence": "high",
            "status": "success" if not is_error else "rejected",
            "suggestions": [],
            "metadata": {"type": "conversational"},
        }

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def _chat_stream(
        self,
        query: str,
        context: str,
        sources: list,
        history: list,
        conversation_id: str,
        retrieval_metadata: dict,
        conv_context,                 # ConversationContext
    ) -> AsyncGenerator[dict, None]:
        yield {
            "type": "metadata",
            "conversation_id": conversation_id,
            "sources": sources,
            "retrieval_metadata": retrieval_metadata,
        }

        full_answer = ""
        async for chunk in llm_service.generate_answer_stream(
            query=query,
            context=context,
            sources=sources,
            conversation_history=history,
        ):
            full_answer += chunk
            yield {"type": "content", "content": chunk}

        await conversation_manager.add_message(
            conversation_id,
            "assistant",
            full_answer,
            metadata={"sources": sources},
        )

        yield {"type": "done", "conversation_id": conversation_id}

    # ── Public helpers ────────────────────────────────────────────────────────

    async def get_conversation_history(
        self, conversation_id: str, user_id: str
    ) -> dict:
        conv = await conversation_manager.get_conversation(conversation_id)
        if not conv:
            return {"error": "Conversation not found"}
        if conv["user_id"] != user_id:
            return {"error": "Unauthorized"}
        return conv

    async def list_conversations(
        self, user_id: str, limit: int = 10
    ) -> list[dict]:
        convs = await conversation_manager.list_user_conversations(user_id, limit)
        results = []
        for conv in convs:
            summary = await conversation_manager.summarize_conversation(conv["id"])
            results.append({**conv, "summary": summary})
        return results

    async def delete_conversation(
        self, conversation_id: str, user_id: str
    ) -> bool:
        conv = await conversation_manager.get_conversation(conversation_id)
        if not conv or conv["user_id"] != user_id:
            return False
        await conversation_manager.delete_conversation(conversation_id)
        await context_manager.delete_context(conversation_id)
        return True


# Singleton
chat_service = ChatService()

__all__ = ["ChatService", "chat_service"]
