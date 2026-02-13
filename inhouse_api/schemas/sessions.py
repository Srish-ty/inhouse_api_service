from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from .events import EventSchema


class SessionCreateRequest(BaseModel):
    app_name: str
    user_id: str
    state: dict[str, Any] | None = None
    session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    app_name: str
    user_id: str
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    last_update_time: datetime | None = None
    events: list[EventSchema] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse] = Field(default_factory=list)