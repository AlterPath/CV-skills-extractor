"""Оценка уровня владения компетенцией (0–3) по тому, где и как она упомянута.

Правила простые и прозрачные, у каждой оценки есть причина:
- 1 (Базовый): навык указан в навыках, образовании, сертификатах или дополнительной информации;
- 2 (Продвинутый): навык применялся в опыте работы или проектах либо рядом есть пример применения;
- 3 (Экспертный): к применению добавляются признаки экспертизы: руководство, архитектура, наставничество, призовые места;
- упоминания в цели и шапке резюме не считаются: это пожелание, а не подтверждение;
- одного стажа для повышения уровня мало: навык из списка, не подтверждённый опытом, остаётся базовым,
  и в отчёте видно, что его стоит уточнить на собеседовании.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import Mention
from .text import NON_EVIDENCE_SECTIONS, Unit

SECTION_LEVEL = {"skills": 1, "education": 1, "certificates": 1, "extra": 1, "experience": 2, "projects": 2}
SECTION_NAMES = {"skills": "навыки", "education": "образование", "certificates": "сертификаты и достижения",
                 "extra": "дополнительно", "experience": "опыт работы", "projects": "проекты"}
EXPERT_RE = re.compile(
    r"руковод|тимлид|\bлид\b|team ?lead|tech ?lead|архитект|ментор|наставн|эксперт|senior|ведущ|преподава"
    r"|(?:топ|top)[- ]?\d+\s*%|призер|призёр|победител|winner", re.I)
USE_RE = re.compile(
    r"разработ|внедр|построи|постро|участ|применя|использ|реализ|настро|администр|провел|провёл|обучил|"
    r"developed|built|implemented|deployed|designed", re.I)


@dataclass(frozen=True)
class LevelEstimate:
    level: int
    reason: str
    quotes: tuple[str, ...]


def estimate_level(mentions: list[Mention], units: list[Unit]) -> LevelEstimate:
    by_index = {u.index: u for u in units}
    best_level, best_reason = 0, "не найдено в резюме"
    quotes: list[str] = []
    for m in mentions:
        if m.section in NON_EVIDENCE_SECTIONS:
            continue
        unit_text = by_index[m.unit].text
        level = SECTION_LEVEL.get(m.section, 1)
        reason = f"{SECTION_NAMES.get(m.section, m.section)}: «{m.text}»"
        if m.semantic:
            level, reason = 1, f"косвенно, похожая формулировка ({m.score:.2f}): «{_short(unit_text)}»"
        elif level == 1 and USE_RE.search(unit_text):
            level, reason = 2, reason + ", с примером применения"
        if not m.semantic and EXPERT_RE.search(unit_text):
            level, reason = min(3, level + 1), reason + ", с признаком экспертизы"
        if _short(unit_text) not in quotes:
            quotes.append(_short(unit_text))
        if level > best_level:
            best_level, best_reason = level, reason
    return LevelEstimate(best_level, best_reason, tuple(quotes[:3]))


def _short(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
