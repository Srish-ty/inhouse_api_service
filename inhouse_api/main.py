from __future__ import annotations

from fastapi import FastAPI

from .core.config import get_settings
from .routers import events
from .routers import health
from .routers import memory
from .routers import sessions


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Inhouse ADK API", debug=settings.app_env != "prod")
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(events.router)
    app.include_router(memory.router)
    return app


app = create_app()