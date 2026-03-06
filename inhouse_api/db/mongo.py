from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorCollection
from motor.motor_asyncio import AsyncIOMotorDatabase

from ..core.config import get_settings


def build_client() -> AsyncIOMotorClient:
    settings = get_settings()
    return AsyncIOMotorClient(settings.mongo_uri)


MONGO_CLIENT = build_client()


def get_database() -> AsyncIOMotorDatabase:
    settings = get_settings()
    return MONGO_CLIENT[settings.mongo_db]


def get_memory_collection() -> AsyncIOMotorCollection:
    settings = get_settings()
    return get_database()[settings.mongo_memory_collection]


def get_profile_collection() -> AsyncIOMotorCollection:
    settings = get_settings()
    return get_database()[settings.mongo_profile_collection]


async def ensure_indexes() -> None:
    """Ensure MongoDB indexes required by API flows exist."""
    memory_collection = get_memory_collection()
    await memory_collection.create_index(
        [
            ("app_name", 1),
            ("user_id", 1),
            ("session_id", 1),
            ("chunk_id", 1),
        ],
        unique=True,
        name="uniq_memory_chunk_per_session",
    )
    await memory_collection.create_index(
        [("app_name", 1), ("user_id", 1), ("session_id", 1)],
        name="idx_memory_session_scope",
    )


async def create_memory_vector_pipeline(
    *, query_embedding: list[float], app_name: str, user_id: str, top_k: int
) -> list[dict[str, Any]]:
    settings = get_settings()
    return [
        {
            "$vectorSearch": {
                "index": settings.vector_index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max(top_k * 5, 50),
                "limit": top_k,
                "filter": {"app_name": app_name, "user_id": user_id},
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
    ]


async def iter_cursor(cursor) -> AsyncGenerator[dict[str, Any], None]:
    async for document in cursor:
        yield document