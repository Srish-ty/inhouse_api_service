from __future__ import annotations

import httpx
from typing import Iterable
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ..core.config import get_settings


class EmbeddingService:
    """Simple embedding adapter.

    If provider is "local", # generate deterministic pseudo-embeddings locally.
    generates embeddings using HuggingFace's all-MiniLM-L6-v2 model.
    If provider is "azure_openai", expects AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_EMBEDDING_MODEL, and AZURE_OPENAI_API_VERSION
    If provider is "openai", expects OPENAI_API_KEY and uses text-embedding-3-large.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._provider = settings.embedding_provider
        self._dim = settings.embedding_dim
        self._embedder: HuggingFaceEmbeddings | None = None

        if self._provider == "local":
            self._ensure_local_embedder()

    async def embed(self, texts: Iterable[str]) -> list[list[float]]:
        if self._provider == 'azure_openai':
            print(f"Using Azure OpenAI for embedding with model: {get_settings().azure_openai_embedding_model}")
            return await self._embed_azure_openai(list(texts))
        if self._provider == "openai":
            return await self._embed_openai(list(texts))
        print("Using local embedding provider (HuggingFace all-MiniLM-L6-v2).")
        return [self._embed_local(text) for text in texts]
    
    async def _embed_azure_openai(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()

        azure_endpoint = settings.azure_openai_endpoint
        azure_key = settings.azure_openai_key

        if not azure_endpoint or not azure_key:
            print("Error: Azure OpenAI credentials not found in settings")
            return [[0.0] * self._dim for _ in texts]
        deployment_name = settings.azure_openai_embedding_model
        url = f"{azure_endpoint}openai/deployments/{deployment_name}/embeddings?api-version={settings.azure_openai_api_version}"

        headers = {
            'api-key': azure_key,
            'Content-Type': 'application/json'
        }
        payload = {
            "input": texts
        }
        try:
            async with httpx.AsyncClient(timeout=50.0) as async_client:
                response = await async_client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                print(f"Received embedding response from Azure OpenAI: {data['data'][:1][0]['embedding'][:5]}...")  # Log only the first item for brevity
                return [item['embedding'] for item in data['data']]
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text if exc.response is not None else ""
            print(
                "HTTP status error during Azure OpenAI embedding: "
                f"status={status}, url={url}, response={body[:1000]}"
            )
            return self._embed_local_batch(texts)
        except httpx.RequestError as exc:
            print(
                "Network/request error during Azure OpenAI embedding: "
                f"type={type(exc).__name__}, detail={exc}, url={url}"
            )
            return self._embed_local_batch(texts)
        except httpx.HTTPError as exc:
            print(
                "Generic HTTP error during Azure OpenAI embedding: "
                f"type={type(exc).__name__}, detail={exc}, url={url}"
            )
            return self._embed_local_batch(texts)

    def _embed_local(self, text: str) -> list[float]:
        # digest = hashlib.sha256(text.encode("utf-8")).digest()
        # random.seed(digest)
        # return [random.uniform(-1, 1) for _ in range(self._dim)]
        self._ensure_local_embedder()
        if self._embedder is None:
            return [0.0] * self._dim
        return self._embedder.embed_query(text)

    def _embed_local_batch(self, texts: list[str]) -> list[list[float]]:
        print(
            "Falling back to local embeddings (sentence-transformers/all-MiniLM-L6-v2) "
            f"for {len(texts)} text(s)."
        )
        return [self._embed_local(text) for text in texts]

    def _ensure_local_embedder(self) -> None:
        if self._embedder is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._embedder = HuggingFaceEmbeddings(model_name=model_name)

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
            model="text-embedding-3-large",
            input=texts,
        )
        return [item.embedding for item in response.data]