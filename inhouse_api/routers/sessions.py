from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.postgres import get_db_session
from ..schemas.sessions import SessionCreateRequest
from ..schemas.sessions import SessionListResponse
from ..schemas.sessions import SessionResponse
from ..services.session_service import SessionService


router = APIRouter(prefix="/v1/sessions")


@router.post("", response_model=SessionResponse)
async def create_session(
    payload: SessionCreateRequest,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    service = SessionService(db)
    return await service.create_session(
        app_name=payload.app_name,
        user_id=payload.user_id,
        state=payload.state,
        session_id=payload.session_id,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    app_name: str,
    user_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    service = SessionService(db)
    session = await service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    app_name: str,
    user_id: str | None = None,
    db: AsyncSession = Depends(get_db_session),
) -> SessionListResponse:
    service = SessionService(db)
    sessions = await service.list_sessions(app_name=app_name, user_id=user_id)
    return SessionListResponse(sessions=sessions)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    app_name: str,
    user_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    service = SessionService(db)
    await service.delete_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    return {"status": "deleted"}