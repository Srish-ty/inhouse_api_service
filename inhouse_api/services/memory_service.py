from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import json
import re
import uuid

from pymongo.errors import OperationFailure

from ..core.config import get_settings
from ..db.mongo import create_memory_vector_pipeline
from ..db.mongo import get_memory_collection
from ..schemas.events import Content
from ..schemas.events import EventSchema
from ..schemas.memory import MemoryEntrySchema
from ..services.embedding_service import EmbeddingService


class MemoryService:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._embedding_service = EmbeddingService()

    async def ingest_session(
        self, *, app_name: str, user_id: str, session_id: str, events: list[EventSchema]
    ) -> int:
        structured_events: list[dict[str, object]] = []
        for event in events:
            if not event.content or not event.content.parts:
                continue
            text_parts = [
                part.text.replace("\n", " ")
                for part in event.content.parts
                if part.text
            ]
            if not text_parts:
                continue
            text = ".".join(text_parts)
            payload = {
                "event_id": event.id,
                "author": event.author,
                "timestamp": event.timestamp,
                "text": text,
            }
            structured_events.append(payload)

        if not structured_events:
            return 0

        chunks = _chunk_events(structured_events, max_chars=2500)
        embeddings = await self._embedding_service.embed([chunk["text"] for chunk in chunks])
        collection = get_memory_collection()
        now = datetime.utcnow()
        documents = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            documents.append(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk["text"],
                    "events": chunk["events"],
                    "embedding": embedding,
                    "created_at": now,
                }
            )
        if documents:
            await collection.insert_many(documents)
        return len(documents)

    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> list[MemoryEntrySchema]:
        embeddings = await self._embedding_service.embed([query])
        collection = get_memory_collection()
        try:
            pipeline = await create_memory_vector_pipeline(
                query_embedding=embeddings[0],
                app_name=app_name,
                user_id=user_id,
                top_k=self._settings.vector_top_k,
            )
            cursor = collection.aggregate(pipeline)
            results = await self._collect_results(cursor)
        except OperationFailure as exc:
            if "vectorSearch" not in str(exc):
                raise
            results = await self._fallback_text_search(
                collection=collection,
                app_name=app_name,
                user_id=user_id,
                query=query,
            )

        session_events_map: OrderedDict[str, list[list[dict[str, object]]]] = (
            OrderedDict()
        )
        for doc in results:
            session_id = str(doc.get("session_id"))
            events = doc.get("events") or []
            if session_id in session_events_map:
                session_events_map[session_id].append(events)
            else:
                session_events_map[session_id] = [events]

        memories: list[MemoryEntrySchema] = []
        for session_id, event_lists in session_events_map.items():
            for merged in _merge_event_lists(event_lists):
                sorted_events = sorted(merged, key=lambda e: e.get("timestamp", 0))
                seen: set[str] = set()
                for event in sorted_events:
                    dedupe_key = _event_dedupe_key(event)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    content = Content(parts=[{"text": event.get("text", "")}])
                    memories.append(
                        MemoryEntrySchema(
                            author=str(event.get("author", "")),
                            content=content.model_dump(),
                            timestamp=datetime.fromtimestamp(
                                float(event.get("timestamp", 0))
                            ).isoformat(),
                            custom_metadata={
                                "event_id": event.get("event_id"),
                                "session_id": session_id,
                            },
                        )
                    )
        return memories

    async def _collect_results(self, cursor) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        async for doc in cursor:
            score = doc.get("score")
            if score is not None and score < self._settings.vector_score_threshold:
                continue
            results.append(doc)
        return results

    async def _fallback_text_search(
        self, *, collection, app_name: str, user_id: str, query: str
    ) -> list[dict[str, object]]:
        escaped = re.escape(query)
        cursor = (
            collection.find(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "text": {"$regex": escaped, "$options": "i"},
                }
            )
            .sort("created_at", -1)
            .limit(self._settings.vector_top_k)
        )
        return [doc async for doc in cursor]


def _chunk_events(
    events: list[dict[str, object]], *, max_chars: int
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    current_events: list[dict[str, object]] = []
    current_lines: list[str] = []
    current_length = 0
    for event in events:
        line = json.dumps(
            {
                "author": event.get("author"),
                "timestamp": event.get("timestamp"),
                "text": event.get("text"),
            }
        )
        if current_length + len(line) + 1 > max_chars and current_lines:
            chunks.append({"text": "\n".join(current_lines), "events": current_events})
            current_lines = []
            current_events = []
            current_length = 0
        current_lines.append(line)
        current_events.append(event)
        current_length += len(line) + 1
    if current_lines:
        chunks.append({"text": "\n".join(current_lines), "events": current_events})
    return chunks


def _event_dedupe_key(event: dict[str, object]) -> str:
    if event.get("event_id"):
        return f"event:{event.get('event_id')}"
    return f"{event.get('timestamp')}::{event.get('author')}::{event.get('text')}"


def _merge_event_lists(
    event_lists: list[list[dict[str, object]]],
) -> list[list[dict[str, object]]]:
    merged: list[list[dict[str, object]]] = []
    event_lists = list(event_lists)
    while event_lists:
        current = event_lists.pop(0)
        current_ts = {event.get("timestamp") for event in current}
        merge_found = True

        while merge_found:
            merge_found = False
            remaining = []
            for other in event_lists:
                other_ts = {event.get("timestamp") for event in other}
                if current_ts & other_ts:
                    new_events = [
                        event
                        for event in other
                        if event.get("timestamp") not in current_ts
                    ]
                    current.extend(new_events)
                    current_ts.update(
                        event.get("timestamp") for event in new_events
                    )
                    merge_found = True
                else:
                    remaining.append(other)
            event_lists = remaining
        merged.append(current)
    return merged