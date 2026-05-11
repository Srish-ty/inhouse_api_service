from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ContentPart(BaseModel):
    text: str | None = None
    function_call: dict[str, Any] | None = None
    function_response: dict[str, Any] | None = None
    code_execution_result: dict[str, Any] | None = None


class Content(BaseModel):
    parts: list[ContentPart] = Field(default_factory=list)


class EventActionsSchema(BaseModel):
    skip_summarization: bool | None = None
    state_delta: dict[str, Any] = Field(default_factory=dict)
    artifact_delta: dict[str, int] = Field(default_factory=dict)
    transfer_to_agent: str | None = None
    escalate: bool | None = None
    requested_auth_configs: dict[str, Any] = Field(default_factory=dict)
    requested_tool_confirmations: dict[str, Any] = Field(default_factory=dict)
    compaction: dict[str, Any] | None = None
    end_of_agent: bool | None = None
    agent_state: dict[str, Any] | None = None
    rewind_before_invocation_id: str | None = None


class EventSchema(BaseModel):
    id: str | None = None
    invocation_id: str | None = None
    author: str
    timestamp: float
    content: Content | None = None
    actions: EventActionsSchema | None = None
    metadata: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    partial: bool = False

    @property
    def timestamp_dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)
