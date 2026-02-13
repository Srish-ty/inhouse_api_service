from __future__ import annotations

import hashlib
import random
from typing import Iterable

from ..core.config import get_settings


class EmbeddingService:
    """Simple embedding adapter.

    If provider is "local", generate deterministic pseudo-embeddings.
    If provider is "openai", expects OPENAI_API_KEY and uses text-embedding-3-small.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._provider = settings.embedding_provider
        self._dim = settings.embedding_dim

    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        if self._provider == "openai":
            return await self._embed_openai(list(texts))
        return [self._embed_local(text) for text in texts]

    def _embed_local(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        random.seed(digest)
        return [random.uniform(-1, 1) for _ in range(self._dim)]

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("OpenAI client not installed.") from exc

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")

        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]