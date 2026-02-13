from __future__ import annotations

from datetime import datetime
from typing import Any

from ..db.mongo import get_profile_collection


class UserProfileService:
    async def upsert_profile(
        self,
        *,
        user_id: str,
        profile_data: dict[str, Any],
        app_scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        collection = get_profile_collection()
        payload = {
            "user_id": user_id,
            "profile_data": profile_data,
            "app_scopes": app_scopes or [],
            "updated_at": datetime.utcnow(),
        }
        await collection.update_one(
            {"user_id": user_id}, {"$set": payload}, upsert=True
        )
        return payload

    async def get_profile(self, *, user_id: str) -> dict[str, Any] | None:
        collection = get_profile_collection()
        return await collection.find_one({"user_id": user_id})