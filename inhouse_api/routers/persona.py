from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.postgres import get_db_session
from ..schemas.persona import PersonaResponse
from ..schemas.persona import PersonaUpdateFromSessionRequest
from ..services.persona_service import PersonaService
from ..services.session_service import SessionService


router = APIRouter(prefix="/v1/persona")


@router.get("", response_model=PersonaResponse)
async def get_persona(
    app_name: str,
    user_id: str,
) -> PersonaResponse:
    service = PersonaService()
    persona = await service.get_persona(app_name=app_name, user_id=user_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


@router.post("/update-from-session", response_model=PersonaResponse)
async def update_persona_from_session(
    payload: PersonaUpdateFromSessionRequest,
    db: AsyncSession = Depends(get_db_session),
) -> PersonaResponse:
    session_service = SessionService(db)
    session = await session_service.get_session(
        app_name=payload.app_name,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    persona_service = PersonaService()
    return await persona_service.upsert_from_session_events(
        app_name=payload.app_name,
        user_id=payload.user_id,
        session_id=payload.session_id,
        events=session.events,
    )
