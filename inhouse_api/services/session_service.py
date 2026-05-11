from __future__ import annotations

import base64
from datetime import datetime
import secrets
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_settings
from ..models.session import SessionRecord
from ..models.session_event import SessionEventRecord
from ..schemas.events import EventSchema
from ..schemas.sessions import SessionResponse


class SessionService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._settings = get_settings()

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> SessionResponse:
        session_id = session_id or self._generate_encoded_id()
        now = datetime.utcnow()
        record = SessionRecord(
            session_id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=state or {},
            created_at=now,
            last_update_time=now,
            deleted=False,
        )
        self._db.add(record)
        await self._db.commit()
        await self._db.refresh(record)
        return SessionResponse(
            session_id=record.session_id,
            app_name=record.app_name,
            user_id=record.user_id,
            state=record.state or {},
            created_at=record.created_at,
            last_update_time=record.last_update_time,
            events=[],
        )

    async def get_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> SessionResponse | None:
        stmt = select(SessionRecord).where(
            SessionRecord.session_id == session_id,
            SessionRecord.app_name == app_name,
            SessionRecord.user_id == user_id,
            SessionRecord.deleted.is_(False),
        )
        record = (await self._db.execute(stmt)).scalars().first()
        if not record:
            return None
        events = await self._get_events(session_id=session_id)
        return SessionResponse(
            session_id=record.session_id,
            app_name=record.app_name,
            user_id=record.user_id,
            state=record.state or {},
            created_at=record.created_at,
            last_update_time=record.last_update_time,
            events=events,
        )

    async def list_sessions(
        self, *, app_name: str, user_id: str | None = None
    ) -> list[SessionResponse]:
        stmt = select(SessionRecord).where(
            SessionRecord.app_name == app_name,
            SessionRecord.deleted.is_(False),
        )
        if user_id:
            stmt = stmt.where(SessionRecord.user_id == user_id)
        stmt = stmt.order_by(SessionRecord.last_update_time.desc())
        records = (await self._db.execute(stmt)).scalars().all()
        return [
            SessionResponse(
                session_id=record.session_id,
                app_name=record.app_name,
                user_id=record.user_id,
                state=record.state or {},
                created_at=record.created_at,
                last_update_time=record.last_update_time,
                events=[],
            )
            for record in records
        ]

    async def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        stmt = select(SessionRecord).where(
            SessionRecord.session_id == session_id,
            SessionRecord.app_name == app_name,
            SessionRecord.user_id == user_id,
        )
        record = (await self._db.execute(stmt)).scalars().first()
        if not record:
            return
        record.deleted = True
        record.last_update_time = datetime.utcnow()
        await self._db.commit()

    async def append_event(
        self, *, app_name: str, user_id: str, session_id: str, event: EventSchema
    ) -> EventSchema:
        self._ensure_event_ids(event)
        if event.partial:
            return event

        event = self._trim_temp_delta_state(event)
        await self._update_session_state(
            session_id=session_id, app_name=app_name, user_id=user_id, event=event
        )
        record = SessionEventRecord(
            event_id=event.id,
            session_id=session_id,
            app_name=app_name,
            user_id=user_id,
            author=event.author,
            timestamp=event.timestamp_dt,
            content_json=event.content.model_dump() if event.content else None,
            actions_json=event.actions.model_dump() if event.actions else None,
            metadata_json=event.metadata,
            error_code=event.error_code,
            error_message=event.error_message,
            invocation_id=event.invocation_id,
        )
        self._db.add(record)
        await self._touch_session(session_id=session_id, app_name=app_name, user_id=user_id)
        await self._db.commit()
        return event

    def _ensure_event_ids(self, event: EventSchema) -> None:
        if not event.id:
            event.id = self._generate_encoded_id()
        if not event.invocation_id:
            event.invocation_id = self._generate_encoded_id()

    @staticmethod
    def _generate_encoded_id() -> str:
        # 32 random bytes encoded with URL-safe base64 produces a 44-char ID
        # (including one trailing "="), matching current production-like shape.
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")

    async def _get_events(self, *, session_id: str) -> list[EventSchema]:
        stmt = (
            select(SessionEventRecord)
            .where(SessionEventRecord.session_id == session_id)
            .order_by(SessionEventRecord.timestamp.asc())
        )
        records = (await self._db.execute(stmt)).scalars().all()
        events: list[EventSchema] = []
        for record in records:
            events.append(
                EventSchema(
                    id=record.event_id,
                    invocation_id=record.invocation_id,
                    author=record.author,
                    timestamp=record.timestamp.timestamp(),
                    content=record.content_json,
                    actions=record.actions_json,
                    metadata=record.metadata_json,
                    error_code=record.error_code,
                    error_message=record.error_message,
                )
            )
        return events

    def _trim_temp_delta_state(self, event: EventSchema) -> EventSchema:
        if not event.actions or not event.actions.state_delta:
            return event
        prefix = self._settings.temp_state_prefix
        event.actions.state_delta = {
            key: value
            for key, value in event.actions.state_delta.items()
            if not key.startswith(prefix)
        }
        return event

    async def _update_session_state(
        self,
        *,
        session_id: str,
        app_name: str,
        user_id: str,
        event: EventSchema,
    ) -> None:
        if not event.actions or not event.actions.state_delta:
            return
        stmt = select(SessionRecord).where(
            SessionRecord.session_id == session_id,
            SessionRecord.app_name == app_name,
            SessionRecord.user_id == user_id,
        )
        record = (await self._db.execute(stmt)).scalars().first()
        if not record:
            return
        state = record.state or {}
        for key, value in event.actions.state_delta.items():
            if key.startswith(self._settings.temp_state_prefix):
                continue
            state[key] = value
        record.state = state
        record.last_update_time = datetime.utcnow()
        await self._db.commit()

    async def _touch_session(
        self, *, session_id: str, app_name: str, user_id: str
    ) -> None:
        stmt = select(SessionRecord).where(
            SessionRecord.session_id == session_id,
            SessionRecord.app_name == app_name,
            SessionRecord.user_id == user_id,
        )
        record = (await self._db.execute(stmt)).scalars().first()
        if not record:
            return
        record.last_update_time = datetime.utcnow()
