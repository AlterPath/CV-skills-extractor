"""Поиск упоминаний компетенций в резюме: словарь ключевых слов и сходство формулировок."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .matrix import Matrix
from .text import NON_EVIDENCE_SECTIONS, Token, Unit, tokenize

ALIASES_PATH = Path(__file__).with_name("aliases.json")


@dataclass(frozen=True)
class Mention:
    competency_id: str
    alias: str        # ключевое слово из словаря
    text: str         # как написано в резюме
    unit: int         # индекс фрагмента
    section: str
    start: int
    end: int
    semantic: bool = False   # найдено по сходству формулировки, а не по ключевому слову
    score: float = 1.0


def load_aliases(path: str | Path = ALIASES_PATH) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_acronym(alias: str) -> bool:
    # короткие аббревиатуры (ML, ИИ, SQL, GPU) сравниваются с учётом регистра
    return len(alias) <= 4 and alias.isupper()


class AliasMatcher:
    """Ищет ключевые слова компетенций как последовательности основ слов, самое длинное совпадение побеждает."""

    def __init__(self, aliases: dict[str, list[str]]):
        self.index: dict[str, list[tuple[tuple[str, ...], str, str, bool]]] = {}
        for comp_id, words in aliases.items():
            for alias in words:
                norms = tuple(t.norm for t in tokenize(alias))
                if norms:
                    self.index.setdefault(norms[0], []).append((norms, comp_id, alias, _is_acronym(alias)))

    def find(self, tokens: list[Token]) -> list[tuple[int, int, str, str]]:
        """Список (начальный токен, конечный токен, id компетенции, ключевое слово)."""
        found, i = [], 0
        while i < len(tokens):
            best_len, best = 0, []
            for norms, comp_id, alias, exact in self.index.get(tokens[i].norm, ()):
                n = len(norms)
                window = tokens[i:i + n]
                if len(window) < n or tuple(t.norm for t in window) != norms:
                    continue
                if exact and " ".join(t.text for t in window) != alias:
                    continue
                if n > best_len:
                    best_len, best = n, [(comp_id, alias)]
                elif n == best_len and comp_id not in {c for c, _ in best}:
                    best.append((comp_id, alias))
            for comp_id, alias in best:
                found.append((i, i + best_len, comp_id, alias))
            i += best_len or 1
        return found


def find_mentions(units: list[Unit], matcher: AliasMatcher) -> list[Mention]:
    mentions = []
    for unit in units:
        tokens = tokenize(unit.text, unit.start)
        for a, b, comp_id, alias in matcher.find(tokens):
            start, end = tokens[a].start, tokens[b - 1].end
            text = unit.text[start - unit.start:end - unit.start]
            mentions.append(Mention(comp_id, alias, text, unit.index, unit.section, start, end))
    return mentions


def semantic_mentions(units: list[Unit], matrix: Matrix, aliases: dict[str, list[str]],
                      skip: set[str], threshold: float = 0.35) -> list[Mention]:
    """Косвенные упоминания: фрагмент резюме похож на название компетенции (TF-IDF по символьным n-граммам).

    Проверяются только компетенции, которых нет среди найденных по ключевым словам.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    candidates = [u for u in units if u.section not in NON_EVIDENCE_SECTIONS]
    comp_ids = [c for c in matrix.competencies if c not in skip]
    if not candidates or not comp_ids:
        return []
    profiles = [matrix.competencies[c].name.lower() for c in comp_ids]
    texts = [u.text.lower() for u in candidates]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vectorizer.fit(profiles + texts + [a.lower() for c in comp_ids for a in aliases.get(c, [])])
    sim = cosine_similarity(vectorizer.transform(profiles), vectorizer.transform(texts))
    mentions = []
    for row, comp_id in enumerate(comp_ids):
        col = int(sim[row].argmax())
        if sim[row, col] >= threshold:
            unit = candidates[col]
            mentions.append(Mention(comp_id, matrix.competencies[comp_id].name, unit.text, unit.index, unit.section,
                                    unit.start, unit.start + len(unit.text), semantic=True, score=float(sim[row, col])))
    return mentions
