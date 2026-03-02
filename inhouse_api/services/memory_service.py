from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import json
import math
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

        chunks = await _chunk_events_hybrid(
            structured_events,
            embedding_service=self._embedding_service,
            max_tokens=self._settings.memory_chunk_max_tokens,
            overlap_tokens=self._settings.memory_chunk_overlap_tokens,
            semantic_similarity_threshold=self._settings.memory_chunk_semantic_similarity_threshold,
            semantic_min_tokens=self._settings.memory_chunk_semantic_min_tokens,
        )
        embeddings = await self._embedding_service.embed([chunk["text"] for chunk in chunks])
        collection = get_memory_collection()
        now = datetime.utcnow()
        documents = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            documents.append(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": idx,
                    "token_count": chunk["token_count"],
                    "start_event_id": chunk.get("start_event_id"),
                    "end_event_id": chunk.get("end_event_id"),
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



async def _chunk_events_hybrid(
    events: list[dict[str, object]],
    *,
    embedding_service: EmbeddingService,
    max_tokens: int,
    overlap_tokens: int,
    semantic_similarity_threshold: float,
    semantic_min_tokens: int,
) -> list[dict[str, object]]:
    """Structural + semantic chunking.

    Structural rules:
    - Never split inside an event
    - Enforce max token budget per chunk

    Semantic rule:
    - If adjacent event embeddings have low similarity and current chunk has
      enough tokens, start a new chunk.
    """
    max_tokens = max(1, max_tokens)
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
    semantic_min_tokens = max(1, min(semantic_min_tokens, max_tokens))

    prepared_events = _prepare_events(events)
    if not prepared_events:
        return []

    event_texts = [str(item["event"].get("text", "")) for item in prepared_events]
    event_embeddings = await embedding_service.embed(event_texts)
 
    chunks_items: list[list[dict[str, object]]] = []
    current_items: list[dict[str, object]] = []
    current_tokens = 0

    for idx, item in enumerate(prepared_events):
        item_tokens = int(item["token_count"])

        # Hard structural split by max token budget.
        if current_items and current_tokens + item_tokens > max_tokens:
            chunks_items.append(current_items)
            current_items = []
            current_tokens = 0

        # Soft semantic split, but only if we already have enough context.
        if (
            current_items
            and current_tokens >= semantic_min_tokens
            and idx > 0
            and idx < len(event_embeddings)
        ):
            similarity = _cosine_similarity(
                event_embeddings[idx - 1],
                event_embeddings[idx],
            )
            if similarity < semantic_similarity_threshold:
                chunks_items.append(current_items)
                current_items = []
                current_tokens = 0

        current_items.append(item)
        current_tokens += item_tokens

    if current_items:
        chunks_items.append(current_items)

    if overlap_tokens > 0:
        chunks_items = _apply_overlap(chunks_items, overlap_tokens)

    return [_build_chunk(chunk_items) for chunk_items in chunks_items]


def _prepare_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    prepared_events: list[dict[str, object]] = []
    for event in events:
        line = json.dumps(
            {
                "author": event.get("author"),
                "timestamp": event.get("timestamp"),
                "text": event.get("text"),
            }
        )
        prepared_events.append(
            {
                "line": line,
                "event": event,
                "token_count": _estimate_tokens(line),
            }
        )
    return prepared_events


def _apply_overlap(
    chunks_items: list[list[dict[str, object]]], overlap_tokens: int
) -> list[list[dict[str, object]]]:
    if not chunks_items:
        return chunks_items

    with_overlap: list[list[dict[str, object]]] = [chunks_items[0]]
    for i in range(1, len(chunks_items)):
        prev_chunk = chunks_items[i - 1]
        current_chunk = list(chunks_items[i])

        overlap_items: list[dict[str, object]] = []
        token_sum = 0
        k = len(prev_chunk) - 1
        while k >= 0 and token_sum < overlap_tokens:
            overlap_items.append(prev_chunk[k])
            token_sum += int(prev_chunk[k]["token_count"])
            k -= 1
        overlap_items.reverse()

        existing_ids = {item["event"].get("event_id") for item in current_chunk}
        merged = [item for item in overlap_items if item["event"].get("event_id") not in existing_ids]
        merged.extend(current_chunk)
        with_overlap.append(merged)

    return with_overlap


def _build_chunk(items: list[dict[str, object]]) -> dict[str, object]:
    current_events = [dict(item["event"]) for item in items]
    current_lines = [str(item["line"]) for item in items]
    token_count = sum(int(item["token_count"]) for item in items)
    return {
        "text": "\n".join(current_lines),
        "events": current_events,
        "token_count": token_count,
        "start_event_id": current_events[0].get("event_id") if current_events else None,
        "end_event_id": current_events[-1].get("event_id") if current_events else None,
    }


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return dot / (norm_a * norm_b)


def _estimate_tokens(text: str) -> int:
    # Lightweight approximation for chunking decisions:
    # - ~4 chars/token heuristic
    # - with a floor using whitespace-based tokenization
    stripped = text.strip()
    if not stripped:
        return 1
    heuristic_tokens = math.ceil(len(stripped) / 4)
    whitespace_tokens = len(re.findall(r"\S+", stripped))
    return max(1, heuristic_tokens, whitespace_tokens)


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