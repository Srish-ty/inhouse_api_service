from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import re

import httpx

from ..core.config import get_settings
from ..db.mongo import get_persona_collection
from ..schemas.events import EventSchema
from ..schemas.persona import PersonaAttributes
from ..schemas.persona import PersonaResponse


class PersonaService:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def get_persona(self, *, app_name: str, user_id: str) -> PersonaResponse | None:
        collection = get_persona_collection()
        doc = await collection.find_one({"app_name": app_name, "user_id": user_id})
        if not doc:
            return None
        return PersonaResponse(
            user_id=str(doc.get("user_id", user_id)),
            last_session=doc.get("last_session"),
            persona=PersonaAttributes(**(doc.get("persona") or {})),
            last_updated_at=doc.get("last_updated_at"),
        )

    async def upsert_from_session_events(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        events: list[EventSchema],
    ) -> PersonaResponse:
        collection = get_persona_collection()
        existing = await collection.find_one({"app_name": app_name, "user_id": user_id})
        existing_persona = PersonaAttributes(**(existing or {}).get("persona", {}))

        user_texts = _extract_user_texts(events)
        heuristic_persona = _build_behavior_persona(user_texts)
        llm_persona = await self._build_persona_with_llm(
            existing=existing_persona,
            session_id=session_id,
            user_texts=user_texts,
        )

        merged_persona = _merge_persona(
            existing=existing_persona,
            updates=llm_persona if llm_persona is not None else heuristic_persona,
        )

        now = datetime.utcnow()
        payload = {
            "app_name": app_name,
            "user_id": user_id,
            "last_session": session_id,
            "persona": merged_persona.model_dump(),
            "last_updated_at": now,
        }
        await collection.update_one(
            {"app_name": app_name, "user_id": user_id},
            {"$set": payload, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

        return PersonaResponse(
            user_id=user_id,
            last_session=session_id,
            persona=merged_persona,
            last_updated_at=now,
        )

    async def _build_persona_with_llm(
        self,
        *,
        existing: PersonaAttributes,
        session_id: str,
        user_texts: list[str],
    ) -> PersonaAttributes | None:
        endpoint = self._settings.azure_openai_endpoint
        key = self._settings.azure_openai_key
        model = self._settings.azure_openai_chat_model
        version = self._settings.azure_openai_api_version
        if not endpoint or not key or not model or not user_texts:
            return None

        url = f"{endpoint}openai/deployments/{model}/chat/completions?api-version={version}"
        headers = {"api-key": key, "Content-Type": "application/json"}
        
        prompt = ( # need to add job roles here
            "You are a persona summarizer.\n"
            "Given prior persona and latest user utterances, return ONLY JSON with keys: "
            "role, job, age, experience, location, hobbies, user_personality, likes, dislikes, preferences, curiosity_topics.\n"
            "Rules: do not hallucinate facts; if unknown use null for strings and [] for lists;"
            " focus on behavior and preferences.\n"
            f"session_id={session_id}\n"
            f"existing_persona={json.dumps(existing.model_dump(), ensure_ascii=False)}\n"
            f"latest_user_messages={json.dumps(user_texts[-40:], ensure_ascii=False)}"
        )

        payload = {
            "messages": [
                {"role": "system", "content": "Output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = _safe_parse_json_object(str(content))
            if not parsed:
                return None
            return PersonaAttributes(**parsed)
        except Exception:
            return None


def _extract_user_texts(events: list[EventSchema]) -> list[str]:
    texts: list[str] = []
    for event in events:
        author = (event.author or "").lower()
        if "agent" in author or "system" in author or "tool" in author:
            continue
        if not event.content or not event.content.parts:
            continue
        parts = [part.text.strip() for part in event.content.parts if part.text and part.text.strip()]
        if not parts:
            continue
        texts.append(" ".join(parts))
    return texts


def _build_behavior_persona(user_texts: list[str]) -> PersonaAttributes:
    if not user_texts:
        return PersonaAttributes()
    combined = "\n".join(user_texts)
    likes = _extract_phrase_matches(combined, [r"\bi like ([^\.\n]+)", r"\bi love ([^\.\n]+)"])
    dislikes = _extract_phrase_matches(combined, [r"\bi dislike ([^\.\n]+)", r"\bi hate ([^\.\n]+)", r"\bi don't like ([^\.\n]+)"])
    preferences = _extract_phrase_matches(combined, [r"\bi prefer ([^\.\n]+)", r"\bi usually ([^\.\n]+)"])

    curiosity_topics = _top_keywords(user_texts)
    question_count = sum(text.count("?") for text in user_texts)
    avg_len = sum(len(t.split()) for t in user_texts) / max(1, len(user_texts))

    personality: list[str] = []
    if question_count >= max(2, len(user_texts) // 3):
        personality.append("inquisitive")
    if avg_len > 20:
        personality.append("detail-oriented")
    if any(word in combined.lower() for word in ["plan", "strategy", "approach", "design"]):
        personality.append("structured-thinker")
    if any(word in combined.lower() for word in ["optimize", "performance", "cost", "efficient"]):
        personality.append("optimization-focused")

    return PersonaAttributes(
        user_personality=_dedupe_keep_order(personality),
        likes=_dedupe_keep_order(likes),
        dislikes=_dedupe_keep_order(dislikes),
        preferences=_dedupe_keep_order(preferences),
        curiosity_topics=curiosity_topics,
    )


def _merge_persona(*, existing: PersonaAttributes, updates: PersonaAttributes) -> PersonaAttributes:
    return PersonaAttributes(
        role=updates.role or existing.role,
        job=updates.job or existing.job,
        age=updates.age or existing.age,
        experience=updates.experience or existing.experience,
        location=updates.location or existing.location,
        hobbies=_merge_lists(existing.hobbies, updates.hobbies),
        user_personality=_merge_lists(existing.user_personality, updates.user_personality),
        likes=_merge_lists(existing.likes, updates.likes),
        dislikes=_merge_lists(existing.dislikes, updates.dislikes),
        preferences=_merge_lists(existing.preferences, updates.preferences),
        curiosity_topics=_merge_lists(existing.curiosity_topics, updates.curiosity_topics),
    )


def _merge_lists(old: list[str], new: list[str]) -> list[str]:
    return _dedupe_keep_order([*old, *new])


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _extract_phrase_matches(text: str, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            cleaned = re.sub(r"\s+", " ", match).strip(" .,!?:;\"'")
            if cleaned:
                found.append(cleaned)
    return found


def _top_keywords(texts: list[str], limit: int = 8) -> list[str]:
    stopwords = {
        "the", "is", "a", "an", "to", "and", "or", "for", "of", "in", "on", "at",
        "i", "we", "you", "it", "this", "that", "with", "about", "from", "my", "me",
        "do", "does", "did", "can", "could", "should", "would", "how", "what", "why",
    }
    words: list[str] = []
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower()):
            if token in stopwords:
                continue
            words.append(token)
    return [word for word, _ in Counter(words).most_common(limit)]


def _safe_parse_json_object(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
