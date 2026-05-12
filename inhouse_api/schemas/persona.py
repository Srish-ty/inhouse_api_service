from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel
from pydantic import Field


class PersonaAttributes(BaseModel):
    job: str | None = None
    job_guess: str | None = None
    designation: str | None = None
    designation_guess: str | None = None
    work_field: list[str] = Field(default_factory=list)
    work_field_guess: list[str] = Field(default_factory=list)
    manager_name: str | None = None
    location: str | None = None

    language_preference: str | None = None
    interaction_pattern: list[str] = Field(default_factory=list)

    age: str | None = None
    experience: str | None = None
    hobbies: list[str] = Field(default_factory=list)
    user_personality: list[str] = Field(default_factory=list)
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    curiosity_topics: list[str] = Field(default_factory=list)


class PersonaResponse(BaseModel):
    user_id: str
    last_session: str | None = None
    persona: PersonaAttributes = Field(default_factory=PersonaAttributes)
    last_updated_at: datetime | None = None


class PersonaUpdateFromSessionRequest(BaseModel):
    app_name: str
    user_id: str
    session_id: str
