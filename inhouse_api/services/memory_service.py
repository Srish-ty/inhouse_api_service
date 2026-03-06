from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import hashlib
import json
import math
import re

from pymongo import UpdateOne
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
        result = await self.sync_session_memory(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            events=events,
            prune_stale=True,
        )
        return result["inserted"]

    async def sync_session_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        events: list[EventSchema],
        prune_stale: bool = True,
    ) -> dict[str, int]:
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

        collection = get_memory_collection()

        if not structured_events:
            deleted = 0
            if prune_stale:
                delete_result = await collection.delete_many(
                    {
                        "app_name": app_name,
                        "user_id": user_id,
                        "session_id": session_id,
                    }
                )
                deleted = int(delete_result.deleted_count)
            return {
                "inserted": 0,
                "updated": 0,
                "deleted": deleted,
                "total_chunks": 0,
            }

        chunks = _chunk_events_hybrid(
            structured_events,
            max_tokens=self._settings.memory_chunk_max_tokens,
            overlap_tokens=self._settings.memory_chunk_overlap_tokens,
        )

        if not chunks:
            deleted = 0
            if prune_stale:
                delete_result = await collection.delete_many(
                    {
                        "app_name": app_name,
                        "user_id": user_id,
                        "session_id": session_id,
                    }
                )
                deleted = int(delete_result.deleted_count)
            return {
                "inserted": 0,
                "updated": 0,
                "deleted": deleted,
                "total_chunks": 0,
            }

        embeddings = await self._embedding_service.embed([chunk["text"] for chunk in chunks])
        now = datetime.utcnow()
        documents: list[dict[str, object]] = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            chunk_id = _deterministic_chunk_id(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                chunk_index=idx,
            )
            documents.append(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "token_count": chunk["token_count"],
                    "start_event_id": chunk.get("start_event_id"),
                    "end_event_id": chunk.get("end_event_id"),
                    "text": chunk["text"],
                    "events": chunk["events"],
                    "embedding": embedding,
                    "updated_at": now,
                }
            )

        operations = [
            UpdateOne(
                {
                    "app_name": doc["app_name"],
                    "user_id": doc["user_id"],
                    "session_id": doc["session_id"],
                    "chunk_id": doc["chunk_id"],
                },
                {
                    "$set": doc,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
            for doc in documents
        ]

        inserted = 0
        updated = 0
        if operations:
            bulk_result = await collection.bulk_write(operations, ordered=False)
            inserted = int(getattr(bulk_result, "upserted_count", 0))
            matched = int(getattr(bulk_result, "matched_count", 0))
            modified = int(getattr(bulk_result, "modified_count", 0))
            # matched includes modified and no-op updates.
            updated = max(matched, modified)

        deleted = 0
        if prune_stale:
            active_chunk_ids = [str(doc["chunk_id"]) for doc in documents]
            delete_result = await collection.delete_many(
                {
                    "app_name": app_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "chunk_id": {"$nin": active_chunk_ids},
                }
            )
            deleted = int(delete_result.deleted_count)

        return {
            "inserted": inserted,
            "updated": updated,
            "deleted": deleted,
            "total_chunks": len(documents),
        }

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



# hybrid chunking function
def _chunk_events_hybrid(
    events: list[dict[str, object]],
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, object]]:
    max_tokens = max(1, max_tokens)
    overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
    prepared_events = _prepare_events(
        events=events,
        max_tokens=max_tokens,
    )
    if not prepared_events:
        return []

    chunks_items: list[list[dict[str, object]]] = []
    current_items: list[dict[str, object]] = []
    current_tokens = 0

    for item in prepared_events:
        item_tokens = int(item["token_count"])

        if current_items and current_tokens + item_tokens > max_tokens:
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


def _prepare_events(
    events: list[dict[str, object]],
    *,
    max_tokens: int,
) -> list[dict[str, object]]:
    prepared_events: list[dict[str, object]] = []
    for source_idx, event in enumerate(events):
        event_text = str(event.get("text", "") or "")
        if not event_text.strip():
            continue

        fragments = _split_event_text_structurally(
            text=event_text,
            max_tokens=max_tokens,
        )
        if not fragments:
            fragments = [event_text]

        base_event_id = event.get("event_id")
        for fragment_idx, fragment in enumerate(fragments):
            text_fragment = fragment.strip()
            if not text_fragment:
                continue
            effective_event = dict(event)
            effective_event["text"] = text_fragment
            if len(fragments) > 1:
                effective_event["parent_event_id"] = base_event_id
                if base_event_id:
                    effective_event["event_id"] = (
                        f"{base_event_id}::seg:{fragment_idx}"
                    )
                else:
                    effective_event["event_id"] = (
                        f"event:{source_idx}::seg:{fragment_idx}"
                    )
                effective_event["event_fragment_index"] = fragment_idx

            line = json.dumps(
                {
                    "author": effective_event.get("author"),
                    "timestamp": effective_event.get("timestamp"),
                    "text": effective_event.get("text"),
                }
            )
            prepared_events.append(
                {
                    "line": line,
                    "event": effective_event,
                    "token_count": _estimate_tokens(text_fragment),
                }
            )
    return prepared_events


def _split_event_text_structurally(
    *,
    text: str,
    max_tokens: int,
) -> list[str]:
    stripped = text.strip()
    print(f"Attempting to structurally split event text: '{stripped[:1000]}' with estimated tokens: {_estimate_tokens(stripped)} and max_tokens: {max_tokens}")
    if not stripped:
        return []
    if _estimate_tokens(stripped) <= max_tokens:
        return [stripped]

    split_with_llama = _split_text_with_llamaindex(
        text=stripped,
        max_tokens=max_tokens,
    )
    if split_with_llama:
        print(
            f"Successfully split event text into {len(split_with_llama)} chunk(s) using LlamaIndex-based semantic splitting."
        )
        return split_with_llama
    else:
        print(
            "LlamaIndex-based splitting failed or produced no chunks, "
            "falling back to heuristic token-based splitting."
        )

    return _split_text_by_token_budget(stripped, max_tokens)


_LLAMA_SEMANTIC_EMBED_MODEL = None


def _get_llama_semantic_embed_model():
    global _LLAMA_SEMANTIC_EMBED_MODEL
    if _LLAMA_SEMANTIC_EMBED_MODEL is None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        _LLAMA_SEMANTIC_EMBED_MODEL = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _LLAMA_SEMANTIC_EMBED_MODEL


def _split_text_with_llamaindex(
    *,
    text: str,
    max_tokens: int,
) -> list[str]:
    try:
        from llama_index.core import Document
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.node_parser import TokenTextSplitter
    except Exception:
        return []

    chunks: list[str] = []
    try:
        semantic_splitter = SemanticSplitterNodeParser.from_defaults(
            embed_model=_get_llama_semantic_embed_model(),
        )
        nodes = semantic_splitter.get_nodes_from_documents([Document(text=text)])
        for node in nodes:
            content = str(getattr(node, "text", "") or "").strip()
            if content:
                chunks.append(content)
    except Exception:
        chunks = []

    if not chunks:
        chunks = SentenceSplitter(
            chunk_size=max_tokens,
            chunk_overlap=0,
        ).split_text(text)

    if not chunks:
        return []

    token_splitter = TokenTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=0,
    )
    bounded: list[str] = []
    for chunk in chunks:
        text_chunk = str(chunk).strip()
        if not text_chunk:
            continue
        if _estimate_tokens(text_chunk) <= max_tokens:
            bounded.append(text_chunk)
            continue
        bounded.extend(
            part.strip()
            for part in token_splitter.split_text(text_chunk)
            if part and part.strip()
        )
    return bounded


def _split_text_by_token_budget(text: str, max_tokens: int) -> list[str]:
    words = re.findall(r"\S+\s*", text)
    if not words:
        return [text]

    chunks: list[str] = []
    current_words: list[str] = []
    current_tokens = 0
    for word in words:
        token_est = _estimate_tokens(word)
        if token_est > max_tokens:
            if current_words:
                chunks.append("".join(current_words).strip())
                current_words = []
                current_tokens = 0
            chunks.extend(_split_text_by_char_budget(word, max_tokens))
            continue
        if current_words and current_tokens + token_est > max_tokens:
            chunks.append("".join(current_words).strip())
            current_words = []
            current_tokens = 0
        current_words.append(word)
        current_tokens += token_est
    if current_words:
        chunks.append("".join(current_words).strip())
    return [chunk for chunk in chunks if chunk]


def _split_text_by_char_budget(text: str, max_tokens: int) -> list[str]:
    max_chars = max(1, max_tokens * 4)
    parts = [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars)]
    return [part for part in parts if part]


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

        existing_keys = {_event_dedupe_key(item["event"]) for item in current_chunk}
        merged = [
            item
            for item in overlap_items
            if _event_dedupe_key(item["event"]) not in existing_keys
        ]
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


def _deterministic_chunk_id(
    *, app_name: str, user_id: str, session_id: str, chunk_index: int
) -> str:
    raw = f"{app_name}|{user_id}|{session_id}|{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _merge_event_lists(
    event_lists: list[list[dict[str, object]]],
) -> list[list[dict[str, object]]]:
    merged: list[list[dict[str, object]]] = []
    event_lists = list(event_lists)
    while event_lists:
        current = event_lists.pop(0)
        current_keys = {_event_dedupe_key(event) for event in current}
        merge_found = True

        while merge_found:
            merge_found = False
            remaining = []
            for other in event_lists:
                other_keys = {_event_dedupe_key(event) for event in other}
                if current_keys & other_keys:
                    new_events = [
                        event
                        for event in other
                        if _event_dedupe_key(event) not in current_keys
                    ]
                    current.extend(new_events)
                    current_keys.update(
                        _event_dedupe_key(event) for event in new_events
                    )
                    merge_found = True
                else:
                    remaining.append(other)
            event_lists = remaining
        merged.append(current)
    return merged
