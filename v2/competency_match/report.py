"""Отчёт по анализу резюме в Markdown и JSON."""
from __future__ import annotations

import json
from dataclasses import asdict

from .assess import BELOW, MET, MISSING, OPTIONAL, Analysis, RoleResult
from .matrix import Matrix


def to_json(analysis: Analysis) -> str:
    return json.dumps(asdict(analysis), ensure_ascii=False, indent=2)


def to_markdown(analysis: Analysis, matrix: Matrix, detailed_roles: int = 1) -> str:
    lines = ["# Соответствие резюме матрице компетенций", ""]
    facts = [f"Резюме: `{analysis.source}`." if analysis.source else "",
             f"Стаж по разделу «Опыт»: {analysis.experience_years:g} г."]
    target = next((r for r in analysis.roles if r.code == analysis.target_role), None)
    if target:
        facts.append(f"Желаемая роль по цели резюме: {target.name}.")
    if analysis.industry:
        facts.append(f"Отрасль: {analysis.industry}, её компетенции отмечены ★.")
    lines += [" ".join(f for f in facts if f), "", "## Рейтинг ролей", "",
              "| Роль | Покрытие | Подтверждено | Ниже уровня | Не найдено |", "|---|---|---|---|---|"]
    for r in analysis.roles:
        mark = " ← цель резюме" if r is target else ""
        lines.append(f"| {r.name} (`{r.code}`){mark} | {r.score:.0%} | {r.met} | {r.below} | {r.missing} |")
    lines += ["", "Покрытие: сумма min(оценка, требуемый уровень) по требуемым компетенциям, делённая на сумму "
              "требуемых уровней. У ролей с низкими требованиями покрытие выше, поэтому первой в отчёте идёт роль "
              "из цели резюме, если она указана.", ""]
    shown = ([target] if target else []) + [r for r in analysis.roles if r is not target]
    for role in shown[:max(detailed_roles, 1 + bool(target))]:
        lines += _role_section(role, matrix)
    if analysis.extra_skills:
        lines += ["## Навыки вне словаря", "",
                  "Их нашёл BiLSTM-теггер, в словаре компетенций таких ключевых слов нет. Стоит проверить вручную.", "",
                  ", ".join(analysis.extra_skills), ""]
    return "\n".join(lines)


def _role_section(role: RoleResult, matrix: Matrix) -> list[str]:
    lines = [f"## {role.name}: покрытие {role.score:.0%}", ""]
    groups = [(MET, "Подтверждено"), (BELOW, "Ниже требуемого уровня"), (MISSING, "Не найдено, стоит подтянуть")]
    for status, title in groups:
        items = [i for i in role.items if i.status == status]
        if not items:
            continue
        items.sort(key=lambda i: (not i.industry_relevant, -(i.required or 0), i.id))
        lines += [f"### {title} ({len(items)})", ""]
        if status == BELOW:
            lines += ["Навык упомянут, но уровень по резюме ниже требуемого: стоит уточнить на собеседовании.", ""]
        if status == MISSING:
            lines += ["| Компетенция | Требуется |", "|---|---|"]
            lines += [f"| {_name(i)} | {matrix.level_name(i.required)} |" for i in items]
        else:
            lines += ["| Компетенция | Требуется | Оценка | Почему |", "|---|---|---|---|"]
            lines += [f"| {_name(i)} | {matrix.level_name(i.required)} | {matrix.level_name(i.level)} | {_cell(i.reason)} |"
                      for i in items]
        lines.append("")
    bonus = [i for i in role.items if i.status == OPTIONAL and i.level > 0]
    if bonus:
        lines += [f"### Есть в резюме, но для роли не обязательно ({len(bonus)})", "",
                  ", ".join(f"{_name(i)} ({matrix.level_name(i.level)})" for i in bonus), ""]
    return lines


def _name(item) -> str:
    return ("★ " if item.industry_relevant else "") + f"{item.id} {item.name}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")
