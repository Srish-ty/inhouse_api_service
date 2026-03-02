# Inhouse ADK API

Standalone FastAPI service that mirrors `VertexAiSessionService` and
`VertexAiRagMemoryService` semantics using Postgres + MongoDB Atlas Vector Search.
If you run against a local MongoDB instance, the memory search endpoint falls
back to a simple text match instead of Atlas vector search.

## Features
- Session lifecycle endpoints (create/get/list/delete)
- Append events with BaseSessionService semantics
- Long-term memory ingestion + vector search
- MongoDB user profile storage

## Local Setup

1. Create Postgres tables:

```bash
psql "$POSTGRES_DSN" -f inhouse_api/migrations/001_init.sql
```

2. Configure environment variables (copy `.env.example`).

3. Install dependencies (FastAPI, SQLAlchemy, asyncpg, motor, pydantic-settings).

4. Run the API:

```bash
uvicorn inhouse_api.main:app --reload --host 0.0.0.0 --port 5001
```

## Environment Variables

See `.env.example` for a full list. Key ones:
- `POSTGRES_DSN` - async SQLAlchemy DSN
- `MONGO_URI` / `MONGO_DB`
- `MONGO_VECTOR_INDEX` - Atlas Vector Search index name
- `EMBEDDING_PROVIDER` (`local`, `openai`, or `azure_openai`)
- `EMBEDDING_DIM` - must match your embedding model output dimension
- `MEMORY_CHUNK_MAX_TOKENS` / `MEMORY_CHUNK_OVERLAP_TOKENS` - token-based chunking knobs
- `MEMORY_CHUNK_SEMANTIC_SIMILARITY_THRESHOLD` / `MEMORY_CHUNK_SEMANTIC_MIN_TOKENS` - hybrid chunking controls

> Note: `$vectorSearch` is only available in MongoDB Atlas. For local MongoDB,
> searches will use a case-insensitive regex match on the stored text instead.

## API Endpoints

### Sessions
- `POST   /v1/sessions`
- `GET    /v1/sessions/{session_id}` (query params: `app_name`, `user_id`)
- `GET    /v1/sessions?app_name=&user_id=`
- `DELETE /v1/sessions/{session_id}` (query params: `app_name`, `user_id`)
- `POST   /v1/sessions/{session_id}/events` (query params: `app_name`, `user_id`)

### Memory
- `POST /v1/memory/ingest-sess`
- `GET  /v1/memory/search?app_name=&user_id=&query=`

### Health
- `GET /health`

## MongoDB Collections

### `memory_chunks`
Document schema:

```json
{
  "app_name": "my-app",
  "user_id": "user-123",
  "session_id": "sess-456",
  "chunk_id": "chunk-uuid",
  "chunk_index": 0,
  "token_count": 512,
  "start_event_id": "evt-001",
  "end_event_id": "evt-004",
  "text": "{json-lines transcript}",
  "events": [{"author": "user", "timestamp": 123, "text": "hi"}],
  "embedding": [0.1, 0.2],
  "created_at": "2026-02-05T12:00:00Z"
}
```

Vector index (Atlas Vector Search):

```json
{
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
    {"type": "filter", "path": "app_name"},
    {"type": "filter", "path": "user_id"}
  ]
}
```

## Memory ingestion chunking behavior

`/v1/memory/ingest-sess` supports two modes:

- `token`: event-boundary-aware token chunking with overlap
- `hybrid` (default): structural + semantic chunking

Hybrid mode does:

- preserves event boundaries (never splits inside an event payload)
- enforces max token budget (`MEMORY_CHUNK_MAX_TOKENS`)
- computes adjacent event semantic similarity using event embeddings
- if similarity drops below `MEMORY_CHUNK_SEMANTIC_SIMILARITY_THRESHOLD` and
  chunk already has at least `MEMORY_CHUNK_SEMANTIC_MIN_TOKENS`, starts a new chunk
- applies overlap (`MEMORY_CHUNK_OVERLAP_TOKENS`) using whole events

This improves topical cohesion vs fixed-size only chunking.

> Note: current token counting is an approximation (`max(chars/4, whitespace tokens)`),
> not model-native tokenizer counting.

## Choosing vector dimension

`EMBEDDING_DIM` should exactly match your embedder output:

- `text-embedding-ada-002`: **1536**
- `text-embedding-3-small`: **1536**
- `text-embedding-3-large`: **3072**
- `all-MiniLM-L6-v2` (local): **384**

If dimensions mismatch Atlas index configuration, `$vectorSearch` will fail.

### `user_profiles`
Minimal schema:

```json
{
  "user_id": "user-123",
  "app_scopes": ["my-app"],
  "profile_data": {"name": "Ada"},
  "updated_at": "2026-02-05T12:00:00Z"
}
```