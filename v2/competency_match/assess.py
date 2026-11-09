"""Сравнение резюме с ролями матрицы: что подтверждено, что ниже требуемого уровня, чего не хватает."""
from __future__ import annotations

from dataclasses import dataclass, field

from .extract import AliasMatcher, Mention, find_mentions, load_aliases, semantic_mentions
from .levels import estimate_level
from .matrix import Matrix
from .text import Unit, experience_years, split_units, tokenize

MET, BELOW, MISSING, OPTIONAL = "подтверждено", "ниже требуемого", "не найдено", "не требуется"


@dataclass
class CompetencyResult:
    id: str
    name: str
    group: str
    required: int | None
    level: int
    status: str
    reason: str
    quotes: list[str]
    industry_relevant: bool


@dataclass
class RoleResult:
    code: str
    name: str
    score: float          # покрытие: сумма min(уровень, требуемый) / сумма требуемых уровней
    met: int
    below: int
    missing: int
    items: list[CompetencyResult]


@dataclass
class Analysis:
    source: str
    experience_years: float
    industry: str | None
    roles: list[RoleResult]                     # по убыванию покрытия
    target_role: str | None = None              # роль из цели резюме («Цель: Data Scientist»)
    extra_skills: list[str] = field(default_factory=list)   # найдены теггером, в словаре их нет


def detect_target_role(units: list[Unit], matrix: Matrix) -> str | None:
    """Ищет в шапке и цели резюме название роли: «Аналитик данных (Data scientist, ml engineer)» даёт три варианта."""
    text = " ".join(" ".join(t.norm for t in tokenize(u.text)) for u in units if u.section in ("header", "objective"))
    for code, role in matrix.roles.items():
        head, _, tail = role.name.partition("(")
        variants = [head] + tail.rstrip(")").split(",")
        for variant in variants:
            norms = " ".join(t.norm for t in tokenize(variant))
            if norms and f" {norms} " in f" {text} ":
                return code
    return None


def analyze(text: str, matrix: Matrix, *, aliases: dict[str, list[str]] | None = None, roles: list[str] | None = None,
            industry: str | None = None, semantic: bool = True, tagger=None, source: str = "") -> Analysis:
    aliases = aliases if aliases is not None else load_aliases()
    if industry and industry not in matrix.industries:
        raise ValueError(f"Неизвестная отрасль {industry!r}, есть: {', '.join(matrix.industries)}")
    units = split_units(text)
    years = experience_years(units)
    mentions = find_mentions(units, AliasMatcher(aliases))
    if semantic:
        mentions += semantic_mentions(units, matrix, aliases, skip={m.competency_id for m in mentions})
    by_comp: dict[str, list[Mention]] = {}
    for m in mentions:
        by_comp.setdefault(m.competency_id, []).append(m)
    estimates = {cid: estimate_level(by_comp.get(cid, []), units) for cid in matrix.competencies}

    results = []
    for code in roles or list(matrix.roles):
        role = matrix.roles[code]
        items, total, got = [], 0, 0
        for cid, required in role.required.items():
            comp, est = matrix.competencies[cid], estimates[cid]
            if required is None:
                status = OPTIONAL
            else:
                total += required
                got += min(est.level, required)
                status = MET if est.level >= required else BELOW if est.level else MISSING
            items.append(CompetencyResult(cid, comp.name, comp.group, required, est.level, status, est.reason,
                                          list(est.quotes), bool(industry and industry in comp.industries)))
        counts = {s: sum(i.status == s for i in items) for s in (MET, BELOW, MISSING)}
        results.append(RoleResult(code, role.name, got / total if total else 0.0,
                                  counts[MET], counts[BELOW], counts[MISSING], items))
    results.sort(key=lambda r: r.score, reverse=True)

    extra = tagger.extra_skills(units, mentions) if tagger is not None else []
    target = detect_target_role(units, matrix)
    return Analysis(source, years, industry, results, target if target in {r.code for r in results} else None, extra)
