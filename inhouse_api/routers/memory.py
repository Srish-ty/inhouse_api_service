from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.postgres import get_db_session
from ..schemas.memory import MemorySearchResponse
from ..schemas.memory import MemorySyncRequest
from ..schemas.memory import MemorySyncResponse
from ..services.memory_service import MemoryService
from ..services.session_service import SessionService


router = APIRouter(prefix="/v1/memory")


@router.post("/sync-session", response_model=MemorySyncResponse)
async def sync_session_memory(
    payload: MemorySyncRequest,
    db: AsyncSession = Depends(get_db_session),
) -> MemorySyncResponse:
    session_service = SessionService(db)
    session = await session_service.get_session(
        app_name=payload.app_name,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    memory_service = MemoryService()
    result = await memory_service.sync_session_memory(
        app_name=payload.app_name,
        user_id=payload.user_id,
        session_id=payload.session_id,
        events=session.events,
        prune_stale=payload.prune_stale,
    )
    return MemorySyncResponse(**result)


@router.get("/search", response_model=MemorySearchResponse)
async def search_memory(
    app_name: str,
    user_id: str,
    query: str,
) -> MemorySearchResponse:
    memory_service = MemoryService()
    memories = await memory_service.search_memory(
        app_name=app_name,
        user_id=user_id,
        query=query,
    )
    return MemorySearchResponse(memories=memories)