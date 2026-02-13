from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.postgres import get_db_session
from ..schemas.events import EventSchema
from ..services.session_service import SessionService


router = APIRouter(prefix="/v1/sessions")


@router.post("/{session_id}/events", response_model=EventSchema)
async def append_event(
    session_id: str,
    app_name: str,
    user_id: str,
    payload: EventSchema,
    db: AsyncSession = Depends(get_db_session),
) -> EventSchema:
    service = SessionService(db)
    session = await service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return await service.append_event(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        event=payload,
    )