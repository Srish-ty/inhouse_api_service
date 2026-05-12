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


JOB_GUESS_OPTIONS = [
    "Senior Relationship Manager",
    "Relationship Manager",
    "Deputy Manager",
    "Associate Manager",
    "Assistant Manager",
    "Senior Executive",
    "Executive",
    "Manager",
    "Senior Manager",
    "Senior Associate",
    "Sales Trainee",
    "Intern",
    "Management Trainee",
    "Assistant Vice President",
    "Vice President",
    "Senior Vice President",
    "President",
    "Managing Director",
    "Software Development Engineer",
    "AI Engineer",
    "Data Scientist",
    "Data Engineer",
    "Product Manager",
    "Business Analyst",
    "Quality Engineer",
    "Application Support Engineer",
]

DESIGNATION_GUESS_OPTIONS = [
    "Frontline - Sales",
    "Branch Sales Manager - DSA & DST",
    "Branch Sales Manager - DST",
    "Branch Sales Manager - DSA",
    "Assistant Branch Sales Manager",
    "Cluster Sales Manager",
    "Branch Manager",
    "Branch Manager - Micro Loans",
    "Assistant Branch Manager - Micro Loans",
    "Branch Credit Manager",
    "Cluster Credit Manager",
    "CPU Credit Manager",
    "Credit Processing Associate",
    "Branch Operations Manager",
    "Branch Operations Support",
    "Cluster Operations Manager",
    "Field Collections",
    "Field Recoveries",
    "Branch Collections Manager",
    "Branch Collections Manager - Unsecured",
    "Area Collections Manager - Unsecured",
    "Branch Recovery Manager",
    "Branch Technical Manager",
    "Cluster Technical Manager",
    "Business Legal Manager",
    "Software Development Engineer 1",
    "Software Developer",
    "AI Engineer",
    "Data Scientist",
    "Data Engineer",
    "Product Manager",
    "Quality Engineer",
    "Application Support Engineer",
    "Business Analyst",
    "Intern",
    "Intern - NAPS",
]

WORK_FIELD_GUESS_OPTIONS = [
    "Sales",
    "Business Intelligence Unit", 
    "Mortgages",
    "Field Collections",
    "Branch Operations",
    "Collections",
    "Field Recoveries",
    "Engineering",
    "AI Engineering",
    "Data Platform Engineering",
    "Quality Engineering",
    "Application Support",
    "Unsecured",
    "Fraud Control Unit",
    "Branch Underwriting",
    "Technical",
    "Product",
    "Digital Product Management",
    "Credit",
    "Credit Risk Analytics",
    "Business HR",
    "Talent Acquisition",
    "Learning & Development",
    "Administration",
    "Audit",
    "Payments & Insurance Operations",
    "Collections Legal",
]


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
        summary_context = _extract_persona_context(events)
        heuristic_persona = _build_behavior_persona(user_texts)
        llm_persona = await self._build_persona_with_llm(
            existing=existing_persona,
            session_id=session_id,
            user_texts=user_texts,
            summary_context=summary_context,
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
        summary_context: list[str],
    ) -> PersonaAttributes | None:
        endpoint = self._settings.azure_openai_endpoint
        key = self._settings.azure_openai_key
        model = self._settings.azure_openai_chat_model
        version = self._settings.azure_openai_api_version
        if not endpoint or not key or not model or not (user_texts or summary_context):
            return None

        url = f"{endpoint}openai/deployments/{model}/chat/completions?api-version={version}"
        headers = {"api-key": key, "Content-Type": "application/json"}

        prompt = _build_persona_prompt(
            existing=existing,
            session_id=session_id,
            user_texts=user_texts,
            summary_context=summary_context,
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


def _extract_persona_context(events: list[EventSchema], limit: int = 20) -> list[str]:
    """Extract summary/profile snippets that can ground stable persona fields."""
    snippets: list[str] = []
    seen: set[str] = set()
    for event in events:
        candidates: list[str] = []
        if event.content and event.content.parts:
            candidates.extend(
                part.text.strip()
                for part in event.content.parts
                if part.text and part.text.strip()
            )
        if event.metadata:
            candidates.append(json.dumps(event.metadata, ensure_ascii=False, default=str))

        for candidate in candidates:
            cleaned = _clean_context_text(candidate)
            if not cleaned or not _looks_like_persona_context(cleaned):
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            snippets.append(cleaned[:2500])
            if len(snippets) >= limit:
                return snippets
    return snippets


def _clean_context_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_persona_context(text: str) -> bool:
    lowered = text.lower()
    keywords = [
        "user_summary",
        "designation",
        "job name",
        "function is",
        "based out of",
        "manager name",
        "employee details",
        "you can access the following usecases",
        "usecase",
    ]
    return any(keyword in lowered for keyword in keywords)


def _build_persona_prompt(
    *,
    existing: PersonaAttributes,
    session_id: str,
    user_texts: list[str],
    summary_context: list[str],
) -> str:
    return (
        "You are a strict persona extraction engine for an internal employee assistant.\n"
        "Return ONLY one valid JSON object. Do not include markdown or explanation.\n\n"
        "Required JSON keys exactly:\n"
        "job, job_guess, designation, designation_guess, work_field, "
        "work_field_guess, manager_name, location, language_preference, "
        "interaction_pattern, age, experience, hobbies, user_personality, "
        "likes, dislikes, preferences, curiosity_topics.\n\n"
        "Field meanings:\n"
        "- job: exact job name/title from explicit evidence only. If no exact evidence exists, return null.\n"
        "- job_guess: best guess selected from job_guess_options when job is not explicitly known. Use messages, usecases, and context clues.\n"
        "- designation: exact business/HR designation from explicit evidence only. If no exact evidence exists, return null.\n"
        "- designation_guess: best guess selected from designation_guess_options when designation is not explicitly known.\n"
        "- work_field: exact function/work field from explicit evidence only. If no exact evidence exists, return [].\n"
        "- work_field_guess: best one or more guesses selected from work_field_guess_options using query behavior, usecases, and context clues.\n"
        "- manager_name: manager name only if explicitly present.\n"
        "- location: branch/base/location only if explicitly present.\n"
        "- language_preference: explicit preferred language only; never infer from name or location.\n"
        "- interaction_pattern: normalized behavior labels from user messages. Allowed examples: summary_checker, breakdown_seeker, performance_tracker, lead_status_checker, contest_tracker, manager_help_seeker, learning_or_hr_query.\n"
        "- age, experience, hobbies, user_personality, likes, dislikes: fill only with direct evidence.\n"
        "- preferences: user-stated or strongly repeated response/task preferences, such as concise summaries, detailed breakdowns, metric-focused responses.\n"
        "- curiosity_topics: recurring business topics/usecases, such as BCI, sales_incentive, UNNATI, Lead Up Report, Lead Status Report, PIRAMAL TWENTY-20 Contest, Piramal Legends Contest, Power-Hitters Contest, Productivity Insights, MFI Task Planner, s2s_incentive.\n\n"
        "Extraction rules:\n"
        "- Use summary_context as the source of truth for job, designation, work_field, manager_name, and location.\n"
        "- Use latest_user_messages as the source of truth for interaction_pattern, preferences, curiosity_topics, and language_preference.\n"
        "- Use existing_persona only to preserve stable prior values when new evidence is absent.\n"
        "- Prefer exact wording from summary_context for job, designation, manager_name, and location.\n"
        "- Exact fields are evidence-only: never fill job, designation, work_field, manager_name, or location from a guess.\n"
        "- Guess fields are allowed to be predictive for demo/evaluation: choose only from the provided option lists.\n"
        "- If no option is reasonably supported for a guess field, use null for job_guess/designation_guess and [] for work_field_guess.\n"
        "- Do not hallucinate factual fields. Unknown string fields must be null. Unknown list fields must be [].\n\n"
        f"job_guess_options={json.dumps(JOB_GUESS_OPTIONS, ensure_ascii=False)}\n"
        f"designation_guess_options={json.dumps(DESIGNATION_GUESS_OPTIONS, ensure_ascii=False)}\n"
        f"work_field_guess_options={json.dumps(WORK_FIELD_GUESS_OPTIONS, ensure_ascii=False)}\n"
        f"session_id={session_id}\n"
        f"existing_persona={json.dumps(existing.model_dump(), ensure_ascii=False)}\n"
        f"summary_context={json.dumps(summary_context[-20:], ensure_ascii=False)}\n"
        f"latest_user_messages={json.dumps(user_texts[-40:], ensure_ascii=False)}"
    )


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
        curiosity_topics=_dedupe_keep_order(
            [*curiosity_topics, *_infer_curiosity_topics(user_texts)]
        ),
        interaction_pattern=_infer_interaction_patterns(user_texts),
        job_guess=_infer_job_guess(user_texts),
        designation_guess=_infer_designation_guess(user_texts),
        work_field_guess=_infer_work_field_guess(user_texts),
    )


def _merge_persona(*, existing: PersonaAttributes, updates: PersonaAttributes) -> PersonaAttributes:
    return PersonaAttributes(
        job=updates.job or existing.job,
        job_guess=updates.job_guess or existing.job_guess,
        designation=updates.designation or existing.designation,
        designation_guess=updates.designation_guess or existing.designation_guess,
        work_field=_merge_lists(existing.work_field, updates.work_field),
        work_field_guess=_merge_lists(
            existing.work_field_guess, updates.work_field_guess
        ),
        manager_name=updates.manager_name or existing.manager_name,
        language_preference=updates.language_preference or existing.language_preference,
        interaction_pattern=_merge_lists(
            existing.interaction_pattern, updates.interaction_pattern
        ),
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


def _infer_interaction_patterns(user_texts: list[str]) -> list[str]:
    combined = "\n".join(user_texts).lower()
    patterns: list[str] = []
    if any(word in combined for word in ["summary", "snapshot", "overview"]):
        patterns.append("summary_checker")
    if any(word in combined for word in ["breakdown", "details", "detail"]):
        patterns.append("breakdown_seeker")
    if any(word in combined for word in ["performance", "bci", "unnati", "incentive"]):
        patterns.append("performance_tracker")
    if any(word in combined for word in ["lead", "leads"]):
        patterns.append("lead_status_checker")
    if any(word in combined for word in ["contest", "leaderboard", "twenty-20", "legends"]):
        patterns.append("contest_tracker")
    if "manager" in combined:
        patterns.append("manager_help_seeker")
    if any(word in combined for word in ["learn", "training", "hr", "policy"]):
        patterns.append("learning_or_hr_query")
    return patterns


def _infer_curiosity_topics(user_texts: list[str]) -> list[str]:
    topic_patterns = [
        ("BCI", [r"\bbci\b"]),
        ("sales_incentive", [r"\bincentive\b", r"\bincentives\b"]),
        ("UNNATI", [r"\bunnati\b"]),
        ("Lead Up Report", [r"\blead up\b", r"\blead update\b", r"\bleads?\b"]),
        ("Lead Status Report", [r"\blead status\b"]),
        ("PIRAMAL TWENTY-20 Contest", [r"\btwenty-20\b", r"\bt20\b"]),
        ("Piramal Legends Contest", [r"\blegends?\b"]),
        ("Power-Hitters Contest", [r"\bpower-hitters?\b", r"\bpower hitters?\b"]),
        ("Productivity Insights", [r"\bproductivity\b"]),
        ("MFI Task Planner", [r"\bmfi\b", r"\btask planner\b", r"\bplan of the day\b"]),
        ("s2s_incentive", [r"\bs2s\b"]),
    ]
    combined = "\n".join(user_texts).lower()
    topics: list[str] = []
    for topic, patterns in topic_patterns:
        if any(re.search(pattern, combined) for pattern in patterns):
            topics.append(topic)
    return topics


def _infer_job_guess(user_texts: list[str]) -> str | None:
    combined = "\n".join(user_texts).lower()
    if _has_any(combined, ["developer", "software", "code", "api", "backend"]):
        return "Software Developer"
    if _has_any(combined, ["ai", "llm", "model", "prompt", "persona"]):
        return "AI Developer"
    if _has_any(combined, ["data", "analytics", "dashboard"]):
        return "Data Scientist"
    if _has_any(combined, ["incentive", "lead", "unnati", "bci", "sales"]):
        return "Relationship Manager"
    return None


def _infer_designation_guess(user_texts: list[str]) -> str | None:
    combined = "\n".join(user_texts).lower()
    if _has_any(combined, ["developer", "software", "code", "api", "backend"]):
        return "Software Developer"
    if _has_any(combined, ["ai", "llm", "model", "prompt", "persona"]):
        return "AI Developer"
    if _has_any(combined, ["data", "analytics", "dashboard"]):
        return "Data Scientist"
    if _has_any(combined, ["incentive", "lead", "unnati", "bci", "sales"]):
        return "Frontline - Sales"
    return None


def _infer_work_field_guess(user_texts: list[str]) -> list[str]:
    combined = "\n".join(user_texts).lower()
    guesses: list[str] = []
    if _has_any(combined, ["ai", "llm", "model", "prompt", "persona"]):
        guesses.append("AI Engineering")
    if _has_any(combined, ["developer", "software", "code", "api", "backend"]):
        guesses.append("Engineering")
    if _has_any(combined, ["data", "analytics", "dashboard"]):
        guesses.append("Data Platform Engineering")
    if _has_any(combined, ["incentive", "lead", "unnati", "bci", "sales"]):
        guesses.append("Sales")
    if _has_any(combined, ["collection", "collections", "recovery"]):
        guesses.append("Collections")
    if _has_any(combined, ["credit", "underwriting"]):
        guesses.append("Credit")
    if _has_any(combined, ["operation", "operations"]):
        guesses.append("Branch Operations")
    return _dedupe_keep_order(guesses)


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


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
