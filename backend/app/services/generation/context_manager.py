"""
Redis-backed context manager for conversational RAG.

CRITICAL FIX: The original stored ConversationContext objects as Python
class instances in a dict. With multiple Render workers these objects
are invisible across processes, breaking follow-up queries.

This version serialises context to Redis (or in-memory fallback) so
every worker can read/write the same conversation context.

ConversationContext itself is now a plain dataclass-style object that
is cheaply created from a dict. The persistence layer is completely
separated from the context logic.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from app.core.config import settings
from app.core.redis_manager import redis_manager

_PREFIX = "ctx:"
_CTX_TTL = settings.context_timeout_hours * 3600


# ── Context data object ──────────────────────────────────────────────────────

class ConversationContext:
    """
    Represents conversation context for a single session.
    Loaded from and saved to Redis on every mutation.
    """

    def __init__(self, conversation_id: str, data: Optional[dict] = None) -> None:
        self.conversation_id = conversation_id
        d = data or {}

        self.created_at: str = d.get("created_at", datetime.utcnow().isoformat())
        self.updated_at: str = d.get("updated_at", datetime.utcnow().isoformat())

        self.active_documents: set[str] = set(d.get("active_documents", []))
        self.document_references: list[dict] = d.get("document_references", [])
        self.primary_document: Optional[str] = d.get("primary_document")

        self.entities: dict[str, list[str]] = defaultdict(
            list, d.get("entities", {})
        )
        self.financial_context: dict = d.get("financial_context", {})
        self.time_periods: list[str] = d.get("time_periods", [])
        self.last_time_reference: Optional[str] = d.get("last_time_reference")

        self.topics: list[str] = d.get("topics", [])
        self.current_topic: Optional[str] = d.get("current_topic")

        self.message_count: int = d.get("message_count", 0)
        self.last_intent: Optional[str] = d.get("last_intent")
        self.expecting_clarification: bool = d.get("expecting_clarification", False)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active_documents": list(self.active_documents),
            "document_references": self.document_references[-20:],
            "primary_document": self.primary_document,
            "entities": dict(self.entities),
            "financial_context": self.financial_context,
            "time_periods": self.time_periods[-10:],
            "last_time_reference": self.last_time_reference,
            "topics": self.topics[-10:],
            "current_topic": self.current_topic,
            "message_count": self.message_count,
            "last_intent": self.last_intent,
            "expecting_clarification": self.expecting_clarification,
        }

    def get_context_summary(self) -> dict:
        return {
            "primary_document": self.primary_document,
            "active_documents": list(self.active_documents),
            "message_count": self.message_count,
            "current_topic": self.current_topic,
            "last_intent": self.last_intent,
            "last_time_reference": self.last_time_reference,
        }
    
    def enhance_query_with_context(self, query: str) -> str:
        """
        Enhance a follow-up query with document and time context.
        Called by query_processor.reformulate_with_context().
        """
        import re
        enhancements = []

        # Add primary document context for scoped conversations
        if self.primary_document and self.should_use_document_scope():
            enhancements.append(f"in document '{self.primary_document}'")

        # Add time period if query doesn't already mention one
        if self.last_time_reference:
            if not re.search(r'\b(q[1-4]|fy|20\d{2}|quarter|year)\b', query.lower()):
                enhancements.append(f"for {self.last_time_reference}")

        if enhancements:
            enhanced = f"{query} ({' '.join(enhancements)})"
            logger.debug(f"Enhanced query: '{query}' → '{enhanced}'")
            return enhanced

        return query

    # ── Mutation helpers ─────────────────────────────────────────────────────

    def update_from_query(self, query: str, processed_query: dict) -> None:
        self.updated_at = datetime.utcnow().isoformat()
        self.message_count += 1
        self.last_intent = processed_query.get("intent")

        for etype, vals in processed_query.get("entities", {}).items():
            self.entities[etype].extend(vals)
            self.entities[etype] = self.entities[etype][-10:]

        tp = self._extract_time_periods(query)
        if tp:
            self.time_periods.extend(tp)
            self.last_time_reference = tp[-1]

    def update_from_retrieval(self, sources: list[dict]) -> None:
        if not sources:
            return

        for source in sources:
            doc_name = source.get("document")
            if doc_name:
                self.active_documents.add(doc_name)
                self.document_references.append({
                    "document": doc_name,
                    "page": source.get("page"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "message_index": self.message_count,
                })

        if self.active_documents:
            from collections import Counter
            doc_counts = Counter(
                ref["document"] for ref in self.document_references
            )
            self.primary_document = doc_counts.most_common(1)[0][0]

    def get_document_filter(self) -> Optional[list[str]]:
        if (
            self.primary_document
            and self.message_count >= settings.document_scope_message_threshold
        ):
            return list(self.active_documents)
        return None

    def should_use_document_scope(self) -> bool:
        return bool(
            self.primary_document
            and self.message_count >= settings.document_scope_message_threshold
        )

    def should_expand_search(self, results: list[dict]) -> bool:
        if not results:
            return True
        top_score = max(
            (
                r.get("similarity_score", 0) or r.get("fused_score", 0)
                for r in results[:3]
            ),
            default=0,
        )
        return top_score < settings.min_relevance_for_scoped_search

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_time_periods(text: str) -> list[str]:
        patterns = [
            r"\b(Q[1-4]\s*\d{4})\b",
            r"\b(FY\s*\d{4})\b",
            r"\b(January|February|March|April|May|June|July|August"
            r"|September|October|November|December)\s+\d{4}\b",
            r"\b(\d{4}-\d{4})\b",
            r"\b(FY\d{2,4}|H[12]\s*\d{4})\b",
        ]
        matches = []
        for pat in patterns:
            matches.extend(re.findall(pat, text, re.IGNORECASE))
        return matches


# ── Manager ──────────────────────────────────────────────────────────────────

class ContextManager:
    """
    Stores and retrieves ConversationContext objects via Redis.
    """

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"{_PREFIX}{conversation_id}"

    async def get_or_create_context(
        self, conversation_id: str
    ) -> ConversationContext:
        raw = await redis_manager.get_json(self._key(conversation_id))
        if raw:
            return ConversationContext(conversation_id, raw)

        ctx = ConversationContext(conversation_id)
        await self._save(ctx)
        return ctx

    async def save_context(self, ctx: ConversationContext) -> None:
        await self._save(ctx)

    async def _save(self, ctx: ConversationContext) -> None:
        await redis_manager.set_json(
            self._key(ctx.conversation_id), ctx.to_dict(), ttl=_CTX_TTL
        )

    async def delete_context(self, conversation_id: str) -> None:
        await redis_manager.delete(self._key(conversation_id))


# Singleton
context_manager = ContextManager()

__all__ = ["ConversationContext", "ContextManager", "context_manager"]
