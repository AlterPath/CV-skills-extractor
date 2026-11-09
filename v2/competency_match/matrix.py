"""Матрица компетенций Альянса: роли, компетенции и требуемые уровни."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LEVEL_NAMES = {1: "Базовый", 2: "Продвинутый", 3: "Экспертный"}


@dataclass(frozen=True)
class Competency:
    id: str
    group: str
    name: str
    industries: tuple[str, ...] = ()


@dataclass(frozen=True)
class Role:
    code: str
    name: str
    description: str
    # id компетенции -> требуемый уровень 1–3, None значит «для роли не требуется»
    required: dict[str, int | None]


@dataclass(frozen=True)
class Matrix:
    competencies: dict[str, Competency]
    roles: dict[str, Role]
    level_names: dict[int, str]
    industries: tuple[str, ...]

    def level_name(self, level: int | None) -> str:
        if not level:
            return "—"
        return self.level_names.get(level, str(level))


def load_matrix(path: str | Path) -> Matrix:
    """Читает alliance_matrix.json. Одна и та же компетенция в разных ролях должна называться одинаково."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    level_names = {int(k): v for k, v in data.get("level_mapping", {}).items()} or DEFAULT_LEVEL_NAMES
    competencies: dict[str, Competency] = {}
    roles: dict[str, Role] = {}
    for code, raw in data["roles"].items():
        required: dict[str, int | None] = {}
        for item in raw["competencies"]:
            comp = Competency(
                id=item["id"],
                group=item["group"],
                name=" ".join(item["name"].split()),
                industries=tuple(item.get("relevant_industries") or ()),
            )
            known = competencies.setdefault(comp.id, comp)
            if known.name != comp.name:
                raise ValueError(f"Компетенция {comp.id} в ролях названа по-разному: {known.name!r} и {comp.name!r}")
            level = item.get("required_level_code")
            if level is not None and level not in level_names:
                raise ValueError(f"{code}/{comp.id}: неизвестный уровень {level!r}")
            required[comp.id] = level
        roles[code] = Role(code=code, name=raw["name"], description=raw.get("description", ""), required=required)
    return Matrix(competencies, roles, level_names, tuple(data.get("industries", ())))
