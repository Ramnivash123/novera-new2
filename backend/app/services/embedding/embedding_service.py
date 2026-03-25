"""
Embedding generation service using Google Gemini.

FIXES APPLIED:
  1. All asyncio.get_event_loop() → asyncio.get_running_loop()
     (deprecated in 3.10, raises RuntimeError in 3.12+)
  2. Per-call timeout on every Gemini embed_content call
  3. use_local_fallback is now instance-local (was already fine,
     kept for clarity)
  4. Removed the logger.info() that printed all method names on
     every startup (unnecessary noise in prod logs)
"""
from __future__ import annotations
import httpx
import asyncio
from typing import Any, Optional

import google.generativeai as genai
import numpy as np
from loguru import logger

from app.core.config import settings

# Try importing torch/sentence-transformers but don't fail if absent.
# On Render we skip installing them (saves 2 GB) and rely on Gemini.
try:
    import torch  # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

_EMBED_TIMEOUT = 30.0   # seconds per batch call


class EmbeddingService:
    """
    Embedding generation via Google Gemini API with optional local fallback.
    All async methods use asyncio.get_running_loop() for Python 3.12 safety.
    """

    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)

        model_name = settings.gemini_embedding_model
        # Normalise: embed_content wants 'models/text-embedding-004'
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        self.model_name = model_name

        self.dimensions = settings.gemini_embedding_dimensions
        self.batch_size = 100

        self.local_model = None
        self.use_local_fallback = False

        logger.info(f"Embedding service ready: {self.model_name} ({self.dimensions}D)")

    # ── Local model (lazy init, only if Gemini fails) ─────────────────────────

    def _init_local_model(self) -> None:
        if self.local_model is not None:
            return
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "Local embedding fallback requested but torch/sentence-transformers "
                "are not installed. Set GEMINI_API_KEY or install torch."
            )
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.warning(f"Loading local embedding model on {device} (Gemini fallback)...")
        self.local_model = SentenceTransformer("all-mpnet-base-v2", device=device)
        logger.info("✅ Local embedding model loaded")

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query (retrieval_query task type)."""
        text = f"Query: {query}"
        if self.use_local_fallback:
            return await self._embed_local(text)
        try:
            return await self._embed_gemini(text, task_type="retrieval_query")
        except Exception as e:
            logger.warning(f"Query embed failed → local fallback: {e}")
            self.use_local_fallback = True
            self._init_local_model()
            return await self._embed_local(text)

    async def generate_embedding(self, text: str) -> list[float]:
        """Embed a single document chunk."""
        if self.use_local_fallback:
            return await self._embed_local(text)
        try:
            return await self._embed_gemini(text)
        except Exception as e:
            if self._is_api_error(e):
                logger.warning(f"Embedding failed → local fallback: {e}")
                self.use_local_fallback = True
                self._init_local_model()
                return await self._embed_local(text)
            raise

    async def generate_embeddings_batch(
        self, texts: list[str], show_progress: bool = False
    ) -> list[list[float]]:
        """Embed a list of texts in batches."""
        if not texts:
            return []
        if self.use_local_fallback:
            return await self._batch_local(texts, show_progress)
        try:
            return await self._batch_gemini(texts, show_progress)
        except Exception as e:
            if self._is_api_error(e):
                logger.warning(f"Batch embed failed → local fallback: {e}")
                self.use_local_fallback = True
                self._init_local_model()
                return await self._batch_local(texts, show_progress)
            raise

    # ── Private Gemini helpers ────────────────────────────────────────────────

    async def _embed_gemini(
        self, text: str, task_type: str = "retrieval_document"
    ) -> list[float]:
        """Direct v1beta REST call — gemini-embedding-001 only exists on v1beta."""
        model_id = self.model_name.replace("models/", "")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models"
            f"/{model_id}:embedContent"
        )
        task_map = {
            "retrieval_document": "RETRIEVAL_DOCUMENT",
            "retrieval_query":    "RETRIEVAL_QUERY",
        }
        payload = {
            "model":    self.model_name,
            "content":  {"parts": [{"text": text}]},
            "task_type": task_map.get(task_type, "RETRIEVAL_DOCUMENT"),
        }
        async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
            response = await client.post(
                url,
                headers={"x-goog-api-key": settings.gemini_api_key},
                json=payload,
            )
            response.raise_for_status()
        return self._adjust_dims(response.json()["embedding"]["values"])

    async def _batch_gemini(
        self, texts: list[str], show_progress: bool
    ) -> list[list[float]]:
        """Batch embed via v1beta batchEmbedContents — gemini-embedding-001."""
        model_id = self.model_name.replace("models/", "")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models"
            f"/{model_id}:batchEmbedContents"
        )
        all_embeddings: list[list[float]] = []
        total = (len(texts) + self.batch_size - 1) // self.batch_size

        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_num = i // self.batch_size + 1
                if show_progress:
                    logger.info(
                        f"Embedding batch {batch_num}/{total} "
                        f"({len(batch)} texts) [v1beta REST]"
                    )
                payload = {
                    "requests": [
                        {
                            "model":    self.model_name,
                            "content":  {"parts": [{"text": t}]},
                            "task_type": "RETRIEVAL_DOCUMENT",
                        }
                        for t in batch
                    ]
                }
                response = await client.post(
                    url,
                    headers={"x-goog-api-key": settings.gemini_api_key},
                    json=payload,
                )
                response.raise_for_status()
                for emb in response.json()["embeddings"]:
                    all_embeddings.append(self._adjust_dims(emb["values"]))
                await asyncio.sleep(0.2)

        logger.info(f"✅ Generated {len(all_embeddings)} embeddings via v1beta REST")
        return all_embeddings

    # ── Private local helpers ─────────────────────────────────────────────────

    async def _embed_local(self, text: str) -> list[float]:
        if self.local_model is None:
            self._init_local_model()
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.local_model.encode(text, convert_to_numpy=True).tolist(),
        )
        return self._adjust_dims(embedding)

    async def _batch_local(
        self, texts: list[str], show_progress: bool
    ) -> list[list[float]]:
        if self.local_model is None:
            self._init_local_model()
        if show_progress:
            logger.info(f"Embedding {len(texts)} texts with local model...")
        loop = asyncio.get_running_loop()
        arr = await loop.run_in_executor(
            None,
            lambda: self.local_model.encode(
                texts, convert_to_numpy=True, show_progress_bar=show_progress
            ),
        )
        return [self._adjust_dims(e.tolist()) for e in arr]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _adjust_dims(self, embedding: list[float]) -> list[float]:
        n = len(embedding)
        if n == self.dimensions:
            return embedding
        if n < self.dimensions:
            return embedding + [0.0] * (self.dimensions - n)
        return embedding[: self.dimensions]

    @staticmethod
    def _is_api_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in ("quota", "rate limit", "429", "401", "403", "auth"))

    def enhance_text_for_embedding(
        self, text: str, context: Optional[dict[str, Any]] = None
    ) -> str:
        if not context:
            return text
        parts = []
        if context.get("document_title"):
            parts.append(f"Document: {context['document_title']}")
        if context.get("section"):
            parts.append(f"Section: {context['section']}")
        if context.get("page"):
            parts.append(f"Page: {context['page']}")
        if context.get("chunk_type") and context["chunk_type"] != "text":
            parts.append(f"Type: {context['chunk_type']}")
        if parts:
            return " | ".join(parts) + "\n\n" + text
        return text

    async def embed_chunks_with_context(
        self, chunks: list[dict], document_title: Optional[str] = None
    ) -> list[dict]:
        texts = [
            self.enhance_text_for_embedding(
                chunk["content"],
                {
                    "document_title": document_title,
                    "section": chunk.get("section_title"),
                    "page": (chunk.get("page_numbers") or [None])[0],
                    "chunk_type": chunk.get("chunk_type", "text"),
                },
            )
            for chunk in chunks
        ]
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embeddings = await self.generate_embeddings_batch(texts, show_progress=True)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb
        return chunks

    @staticmethod
    def cosine_similarity(v1: list[float], v2: list[float]) -> float:
        a, b = np.array(v1), np.array(v2)
        n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(a, b) / (n1 * n2))


# Singleton
embedding_service = EmbeddingService()

__all__ = ["EmbeddingService", "embedding_service"]
