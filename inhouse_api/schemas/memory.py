from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class MemoryEntrySchema(BaseModel):
    author: str | None = None
    content: dict[str, Any]
    timestamp: str | None = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResponse(BaseModel):
    memories: list[MemoryEntrySchema] = Field(default_factory=list)


class MemorySyncRequest(BaseModel):
    app_name: str
    user_id: str
    session_id: str
    prune_stale: bool = True


class MemorySyncResponse(BaseModel):
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    total_chunks: int = 0


class MemoryChunkSchema(BaseModel):
    app_name: str
    user_id: str
    session_id: str
    chunk_id: str
    text: str
    events: list[dict[str, Any]] = Field(default_factory=list)
    embedding: list[float]
    created_at: datetime