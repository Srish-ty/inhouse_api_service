from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from .base import Base


class SessionEventRecord(Base):
    __tablename__ = "session_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    app_name: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    author: Mapped[str] = mapped_column(String(128))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    content_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    actions_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    invocation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)