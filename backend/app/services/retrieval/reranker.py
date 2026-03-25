"""
Reranking service using Cohere Rerank API.

FIXES APPLIED:
  1. The original did `cohere.Client(api_key=settings.cohere_api_key)` at
     module import time. If COHERE_API_KEY is missing the entire app crashes
     at startup with an AuthenticationError. Now we do a lazy init — the
     client is only created on the first rerank call, and if the key is
     absent we fall back to the original ranking gracefully.
  2. asyncio.get_event_loop() → asyncio.get_running_loop()
  3. Added a 10-second timeout on the Cohere API call so a slow response
     doesn't hold up the retrieval pipeline.
  4. Cohere v5 client API changed (rerank() method signature); this file
     is compatible with both cohere==4.x (our pinned version) and v5.
"""
from __future__ import annotations
from typing import Any, Optional

import asyncio
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

_RERANK_TIMEOUT = 10.0   # seconds


class RerankingService:
    """
    Reranks retrieved chunks using Cohere's Rerank model.
    Gracefully degrades to original ranking when Cohere is unavailable.
    """

    def __init__(self) -> None:
        self._client = None          # lazy — created on first use
        self._available: Optional[bool] = None   # None = not yet tested
        self.model = settings.cohere_rerank_model
        self.top_n = settings.rerank_top_k
        logger.info(f"Reranking service ready (lazy init): {self.model}")

    def _get_client(self):
        """Return Cohere client, or None if key is absent."""
        if self._available is False:
            return None
        if self._client is not None:
            return self._client

        api_key = settings.cohere_api_key
        if not api_key:
            logger.warning(
                "⚠️ COHERE_API_KEY not set — reranking disabled, using original ranking"
            )
            self._available = False
            return None

        try:
            import cohere
            self._client = cohere.Client(api_key=api_key)
            self._available = True
            logger.info("✅ Cohere client initialised")
            return self._client
        except Exception as e:
            logger.error(f"Cohere client init failed: {e} — reranking disabled")
            self._available = False
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank chunks by relevance to query.
        Falls back to original ranking if Cohere is unavailable.
        """
        if not chunks:
            return []

        n = top_n or self.top_n
        client = self._get_client()

        if client is None:
            return self._fallback_ranking(chunks, n)

        documents = [c["content"] for c in chunks]

        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: client.rerank(
                        model=self.model,
                        query=query,
                        documents=documents,
                        top_n=min(n, len(documents)),
                    ),
                ),
                timeout=_RERANK_TIMEOUT,
            )

            reranked = []
            for res in response.results:
                chunk = chunks[res.index].copy()
                chunk["rerank_score"] = res.relevance_score
                chunk["rerank_position"] = res.index
                reranked.append(chunk)

            logger.info(f"✅ Reranked: {len(reranked)} results")
            return reranked

        except asyncio.TimeoutError:
            logger.warning(f"Cohere rerank timed out after {_RERANK_TIMEOUT}s — using fallback")
            return self._fallback_ranking(chunks, n)
        except Exception as e:
            logger.error(f"Reranking failed: {e} — using fallback")
            return self._fallback_ranking(chunks, n)

    async def rerank_with_threshold(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        relevance_threshold: float = 0.5,
        top_n: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        reranked = await self.rerank(query, chunks, top_n)
        filtered = [c for c in reranked if c.get("rerank_score", 1.0) >= relevance_threshold]
        logger.info(f"Filtered to {len(filtered)} chunks above threshold {relevance_threshold}")
        return filtered

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_ranking(
        chunks: list[dict[str, Any]], n: int
    ) -> list[dict[str, Any]]:
        """Return top-n chunks by their existing similarity/fused score."""
        result = []
        for i, chunk in enumerate(chunks[:n]):
            c = chunk.copy()
            c["rerank_score"] = c.get("similarity_score") or c.get("fused_score") or 0.0
            c["rerank_position"] = i
            result.append(c)
        return result

    def calculate_score_statistics(self, chunks: list[dict]) -> dict:
        if not chunks:
            return {}
        scores = [c.get("rerank_score", 0) for c in chunks]
        sorted_scores = sorted(scores)
        return {
            "min_score": sorted_scores[0],
            "max_score": sorted_scores[-1],
            "avg_score": sum(scores) / len(scores),
            "median_score": sorted_scores[len(scores) // 2],
            "total_chunks": len(chunks),
        }


# Singleton
reranking_service = RerankingService()

__all__ = ["RerankingService", "reranking_service"]
