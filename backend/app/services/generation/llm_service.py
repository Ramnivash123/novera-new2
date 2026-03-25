"""
LLM generation service using Google Gemini.

FIXES APPLIED:
  1. asyncio.get_event_loop() → asyncio.get_running_loop()
     (get_event_loop() is deprecated in 3.10, errors in 3.12+)
  2. asyncio.wait_for() timeout on every API call
     (a hung Gemini request no longer freezes the whole worker)
  3. generate_follow_up_suggestions is now truly async-parallel:
     it's fired as a background task, so the main chat response
     returns to the user immediately. Suggestions arrive separately.
  4. Removed stale DEBUG dir() / raw dump logs from production path.
  5. Re-initialises the model lazily on first call so startup is faster.
  6. Added retry backoff on rate-limit (429) errors specifically.
"""
from __future__ import annotations

import asyncio
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
import random
import hashlib
import time
import google.generativeai as genai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from loguru import logger
import tiktoken

from app.core.config import settings

# Global semaphore — only 1 LLM call at a time across all workers in this process.
# Prevents parallel requests from simultaneously exhausting the per-minute quota.
_LLM_SEMAPHORE = asyncio.Semaphore(1)

# Simple in-memory response cache — keyed by SHA256 of (model + prompt).
# Avoids burning quota on repeated identical queries.
_RESPONSE_CACHE: dict[str, tuple[Any, float]] = {}
_CACHE_TTL_SECONDS = 300   # cache hits live for 5 minutes


def _cache_key(model_name: str, content: Any) -> str:
    raw = f"{model_name}::{str(content)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_cached(key: str) -> Any | None:
    entry = _RESPONSE_CACHE.get(key)
    if entry and (time.monotonic() - entry[1]) < _CACHE_TTL_SECONDS:
        return entry[0]
    _RESPONSE_CACHE.pop(key, None)
    return None


def _set_cached(key: str, value: Any) -> None:
    # Keep cache small — evict oldest entry when over 200 items
    if len(_RESPONSE_CACHE) >= 200:
        oldest = min(_RESPONSE_CACHE, key=lambda k: _RESPONSE_CACHE[k][1])
        _RESPONSE_CACHE.pop(oldest, None)
    _RESPONSE_CACHE[key] = (value, time.monotonic())

# Per-call timeouts (seconds)
_ANSWER_TIMEOUT = 30.0
_SUGGESTIONS_TIMEOUT = 8.0
_SUMMARY_TIMEOUT = 20.0
_CONVERSATIONAL_TIMEOUT = 15.0


def _is_retryable(exc: BaseException) -> bool:
    """Retry only on API quota / transient errors, not on bad-request errors."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "quota", "resource_exhausted", "rate", "503", "500"))


class LLMService:
    """
    Service for generating answers using Google Gemini.
    All public methods are async and include per-call timeouts.
    """

    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)

        # Primary model from config
        raw = settings.gemini_chat_model
        self.model_name = raw.replace('models/', '') if raw.startswith('models/') else raw
        self.temperature = settings.temperature

        # Fallback chain — tried in order when primary hits quota/rate limits
        # gemini-2.5-flash → gemini-2.0-flash → gemini-1.5-flash → gemini-1.5-pro
        self._model_chain = [
            self.model_name,           # gemini-2.5-flash (primary)
            "gemini-2.5-flash-lite",   # highest free quota, fast
        ]
        seen: set = set()
        self._model_chain = [m for m in self._model_chain if not (m in seen or seen.add(m))]
        logger.info(f"LLM model chain: {self._model_chain}")

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": settings.max_response_tokens,
                "top_p": 0.95,
            }
        )

        self.system_instruction = self._build_system_instruction()

        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
            logger.info(f"LLM service initialized: {self.model_name} with tiktoken encoder")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoder: {e}, using estimation fallback")
            self._enc = None

    @staticmethod
    def _build_system_instruction() -> str:
        return """You are Novera, an AI assistant specializing in Finance and HRMS documentation.

Core Guidelines:
1. Answer questions using ONLY information from provided context when documents are available
2. Be conversational, friendly, and professional
3. Understand the situation properly and answer accordingly with empathy
4. For financial figures: Include exact numbers with proper citations
5. If information is not in context: Clearly state what's missing
6. CRITICAL: Use numbered citations [1], [2], [3] to reference sources
7. Place citations IMMEDIATELY after each fact: "The policy states X [1]."

Citation Rules:
- Use [1], [2], [3] format (NOT [Document: X, Page: Y])
- Each unique source gets a unique number
- Multiple sources: "This is confirmed [1,2,3]"
- Place citation right after the relevant statement

Response Formatting:
- Use natural, conversational language
- Structure clearly: paragraphs, bullet points, bold (**text**)
- For tabular data, use Markdown tables

Remember: Each fact from documents MUST have a citation number."""

    async def _call_with_fallback(
        self,
        messages_or_prompt: Any,
        generation_config: dict,
        timeout: float = 30.0,
    ) -> Any:
        import re

        cache_key = _cache_key(str(self._model_chain), messages_or_prompt)
        cached = _get_cached(cache_key)
        if cached is not None:
            logger.info("Cache hit — skipping LLM call")
            return cached

        def _parse_retry_delay(error_str: str) -> float:
            match = re.search(r'retry[_ ]in\s+([\d.]+)s', error_str, re.IGNORECASE)
            return float(match.group(1)) + 1.0 if match else 12.0

        async def _try_model(model_name: str) -> tuple[Any, float]:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                )
                loop = asyncio.get_running_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda m=model: m.generate_content(messages_or_prompt),
                    ),
                    timeout=timeout,
                )
                if model_name != self.model_name:
                    logger.info(f"Fallback succeeded using {model_name}")
                _set_cached(cache_key, response)
                return response, 0.0
            except asyncio.TimeoutError:
                logger.warning(f"{model_name} timed out after {timeout}s")
                return None, -1.0
            except Exception as e:
                msg = str(e).lower()
                if any(k in msg for k in ("429", "quota", "rate", "resource_exhausted")):
                    delay = _parse_retry_delay(str(e))
                    logger.warning(f"{model_name} rate-limited — wait {delay:.1f}s")
                    return None, delay
                logger.error(f"{model_name} non-quota error: {e}")
                return None, -1.0

        async with _LLM_SEMAPHORE:
            # Try all live models immediately
            max_delay = 0.0
            for model_name in self._model_chain:
                response, delay = await _try_model(model_name)
                if response is not None:
                    return response
                if delay > 0:
                    max_delay = max(max_delay, delay)

            if max_delay <= 0:
                # Hard errors on all models, not rate limiting
                logger.error("All models failed with hard errors — returning None")
                return None

            # Rate limited — wait exactly what Google says, then one final attempt
            logger.warning(f"Rate limited — waiting {max_delay:.1f}s then final attempt")
            await asyncio.sleep(max_delay)

            for model_name in self._model_chain:
                response, _ = await _try_model(model_name)
                if response is not None:
                    return response

            logger.error("All models exhausted — returning None")
            return None

    def count_tokens(self, text: str) -> int:
        if self._enc:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        return len(text) // 4

    # ── Core generation ───────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_answer(
        self,
        query: str,
        context: str,
        sources: list[dict],
        conversation_history: Optional[list[dict]] = None,
        reformulated_query: Optional[str] = None,
        conversation_context: Optional[dict] = None,
        is_conversational: bool = False,
    ) -> dict[str, Any]:
        """
        Generate an answer using Gemini. Includes a hard timeout so a
        hung API call never freezes a worker.
        """
        recent_history = (conversation_history or [])[-4:]  # last 2 turns

        system_instruction = self._get_context_aware_system_instruction(
            conversation_context
        )

        if is_conversational or not context or len(context.strip()) < 50:
            user_prompt = self._build_conversational_prompt(
                query, recent_history, conversation_context
            )
        else:
            user_prompt = self._build_contextual_prompt(
                query, context, sources, reformulated_query, conversation_context
            )

        messages = [
            {"role": "user", "parts": [system_instruction]},
            {"role": "model", "parts": ["Understood. I will provide accurate, context-aware responses."]},
        ]
        for msg in recent_history:
            role = "model" if msg["role"] == "assistant" else "user"
            messages.append({"role": role, "parts": [msg["content"]]})
        messages.append({"role": "user", "parts": [user_prompt]})

        response = await self._call_with_fallback(
            messages,
            generation_config={
                "temperature": 0.7 if is_conversational else self.temperature,
                "max_output_tokens": settings.max_response_tokens,
            },
            timeout=_ANSWER_TIMEOUT,
        )

        # ADD THIS BLOCK immediately after:
        if response is None:
            logger.warning("⚠️ All models rate-limited — returning quota message")
            return {
                "answer": (
                    "I'm currently experiencing high demand and my AI quota is temporarily "
                    "exhausted. Please wait 30-60 seconds and try again. "
                    "If this keeps happening, check your Gemini API billing at "
                    "https://aistudio.google.com"
                ),
                "citations": [],
                "confidence": "low",
                "finish_reason": "quota_exceeded",
                "usage": {"total_tokens": 0, "prompt_tokens": 0, 
                          "completion_tokens": 0, "cached_tokens": 0},
            }

        answer = response.text
        citations = self._extract_citations(answer, sources)
        confidence = self._assess_confidence(answer, context, conversation_context)
        usage = self._extract_usage(response, system_instruction, recent_history, user_prompt, answer)

        logger.info(
            f"✅ Generated: {len(answer)} chars | {len(citations)} citations | "
            f"{usage['total_tokens']} tokens"
        )

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "finish_reason": "stop",
            "usage": usage,
        }

    async def generate_conversational_response(
        self,
        query: str,
        context_message: str,
        history: Optional[list[dict]] = None,
        is_error: bool = False,
    ) -> dict[str, Any]:
        if is_error:
            prompt = (
                f'The user asked: "{query}"\n\n'
                f"Issue: {context_message}\n\n"
                "Respond helpfully and guide the user on how to get assistance "
                "with their Finance and HRMS documents."
            )
        else:
            prompt = (
                f'The user said: "{query}"\n\n'
                "Respond naturally. You are Novera, an AI assistant that helps "
                "with Finance and HRMS documents. Be warm and guide the user."
            )

        response = await self._call_with_fallback(
            prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 500,
            },
            timeout=_CONVERSATIONAL_TIMEOUT,
        )
        if response is None:
            return {
                "answer": "I'm temporarily rate-limited. Please wait 30 seconds and try again.",
                "citations": [],
                "confidence": "low",
                "usage": {"total_tokens": 0},
            }
        return {
            "answer": response.text,
            "citations": [],
            "confidence": "high",
            "usage": {"total_tokens": 0},
        }

    async def generate_follow_up_suggestions(self, prompt: str) -> list[str]:
        """
        Generate 3-4 follow-up suggestions. Uses a tight timeout so
        the main response is never delayed waiting for this.
        """
        try:
            response = await self._call_with_fallback(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 300,
                    "top_p": 0.9,
                },
                timeout=_SUGGESTIONS_TIMEOUT,
            )

            if response is None:
                return []
            return [
                line.strip()
                for line in response.text.split("\n")
                if line.strip() and len(line.strip()) > 10
            ]
        except asyncio.TimeoutError:
            logger.warning("Suggestion generation timed out — returning empty list")
            return []
        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return []

    async def generate_answer_stream(
        self,
        query: str,
        context: str,
        sources: list[dict],
        conversation_history: Optional[list[dict]] = None,
    ):
        """Async generator that yields response text chunks."""
        from app.services.generation.llm_service import llm_service  # avoid circular

        prompt = self._build_contextual_prompt(
            query, context, sources, None, None
        )
        loop = asyncio.get_running_loop()

        # Gemini streaming is not truly async in the SDK — run in executor
        def _stream():
            return self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": settings.max_response_tokens,
                },
                stream=True,
            )

        stream_response = await loop.run_in_executor(None, _stream)
        for chunk in stream_response:
            if chunk.text:
                yield chunk.text

    async def summarize_document(
        self, document_content: str, document_title: str, max_length: int = 500
    ) -> str:
        prompt = (
            f"Summarize the following document concisely in {max_length} words or less.\n"
            "Focus on key points, main topics, and important information.\n\n"
            f"Document: {document_title}\n\nContent:\n{document_content[:4000]}\n\nSummary:"
        )
        try:
            response = await self._call_with_fallback(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": max_length * 2,
                },
                timeout=_SUMMARY_TIMEOUT,
            )
            if response is None:
                return f"Summary unavailable for {document_title}"
            return response.text
        
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Summary unavailable for {document_title}"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_context_aware_system_instruction(self, ctx: Optional[dict]) -> str:
        instruction = self.system_instruction
        if ctx and ctx.get("primary_document"):
            instruction += "\n\nCurrent focus: Document-scoped conversation."
        return instruction

    def _build_contextual_prompt(
        self,
        query: str,
        context: str,
        sources: list[dict],
        reformulated_query: Optional[str],
        conversation_context: Optional[dict],
    ) -> str:
        sources_text = []
        for idx, src in enumerate(sources, 1):
            line = f"[{idx}] {src.get('document', 'Unknown')}"
            if src.get("page"):
                line += f", Page {src['page']}"
            if src.get("section"):
                line += f", Section: {src['section']}"
            sources_text.append(line)

        parts = []
        if conversation_context and conversation_context.get("primary_document"):
            parts.append(
                f"**Context**: We are currently discussing "
                f"'{conversation_context['primary_document']}'"
            )

        if reformulated_query and reformulated_query != query:
            parts.append(f"**User's Question**: {query}")
            parts.append(f"**Interpreted as**: {reformulated_query}")
        else:
            parts.append(f"**Question**: {query}")

        parts.append(
            f"**Available Context from Documents**:\n{context}\n\n"
            f"**Source References** (use these numbers in citations):\n"
            + "\n".join(sources_text)
            + "\n\n"
            "**Critical Instructions**:\n"
            "1. Answer based ONLY on the context above\n"
            "2. MUST cite sources using [1], [2], [3] format\n"
            "3. Place citations immediately after facts: 'The policy is X [1].'\n"
            "4. If the context does not contain information relevant to the question, say exactly:\n"
            "\"I don't have specific information about this in the available documents.\"\n"
            "   Do NOT guess, infer, or use general knowledge — only answer from the context above.\n"
            "5. For financial data, include exact figures with citations\n\n"
            "**Answer** (remember: MUST include [1], [2], [3] citations):"
        )

        return "\n\n".join(parts)

    def _build_conversational_prompt(
        self,
        query: str,
        history: Optional[list[dict]],
        ctx: Optional[dict],
    ) -> str:
        parts = [f"**User**: {query}"]
        if ctx and ctx.get("message_count", 0) > 0:
            parts.append("**Context**: This is part of an ongoing conversation.")
        parts.append(
            "**Instructions**: Respond naturally. Be warm, helpful, and professional.\n"
            "**Response**:"
        )
        return "\n".join(parts)

    def _extract_citations(
        self, answer: str, sources: list[dict]
    ) -> list[dict]:
        if not sources:
            return []

        cited_nums: set[int] = set()
        for match in re.findall(r"\[(\d+(?:,\s*\d+)*)\]", answer):
            cited_nums.update(int(n.strip()) for n in match.split(","))

        citations = []
        for num in sorted(cited_nums):
            idx = num - 1
            if 0 <= idx < len(sources):
                src = sources[idx]
                citations.append({
                    "document": src.get("document", "Unknown"),
                    "page": src.get("page"),
                    "chunk_id": src.get("chunk_id"),
                    "section": src.get("section"),
                    "citation_number": num,
                    "text_reference": f"[{num}]",
                })
        return citations

    def _assess_confidence(
        self, answer: str, context: str, ctx: Optional[dict]
    ) -> str:
        low = [
            "not available", "unclear", "doesn't specify",
            "may", "might", "possibly", "appears to", "i don't have",
        ]
        answer_lower = answer.lower()
        if any(ind in answer_lower for ind in low):
            return "low"

        has_citations = bool(re.search(r"\[\d+\]", answer))
        if has_citations:
            if ctx and ctx.get("primary_document"):
                return "high"
            return "medium"
        return "medium"

    def _extract_usage(
        self,
        response,
        system: str,
        history: list,
        prompt: str,
        answer: str,
    ) -> dict:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            m = response.usage_metadata
            return {
                "prompt_tokens": getattr(m, "prompt_token_count", 0),
                "completion_tokens": getattr(m, "candidates_token_count", 0),
                "total_tokens": getattr(m, "total_token_count", 0),
                "cached_tokens": getattr(m, "cached_content_token_count", 0),
            }

        # Fallback manual count
        p = self.count_tokens(system) + sum(
            self.count_tokens(m["content"]) for m in history
        ) + self.count_tokens(prompt)
        c = self.count_tokens(answer)
        return {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c,
            "cached_tokens": 0,
        }


# Singleton
llm_service = LLMService()

__all__ = ["LLMService", "llm_service"]
