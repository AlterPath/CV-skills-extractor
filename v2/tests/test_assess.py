from pathlib import Path

import pytest

from competency_match import analyze, load_matrix
from competency_match.assess import BELOW, MET, MISSING, OPTIONAL
from competency_match.extract import AliasMatcher, find_mentions
from competency_match.levels import estimate_level
from competency_match.text import split_units

MATRIX = Path(__file__).resolve().parents[2] / "data" / "alliance_matrix.json"


@pytest.fixture(scope="module")
def matrix():
    return load_matrix(MATRIX)


def mentions_for(text, aliases):
    units = split_units(text)
    return units, find_mentions(units, AliasMatcher(aliases))


def test_matrix_shape(matrix):
    assert set(matrix.roles) == {"DS_ML", "DE", "TA_AI", "MGR_AI"}
    assert len(matrix.competencies) == 34
    assert all(len(r.required) == 34 for r in matrix.roles.values())


def test_inflected_phrase_found():
    units, ms = mentions_for("Навыки:\nПроекты по машинному обучению", {"A01": ["машинное обучение"]})
    assert [(m.competency_id, m.text) for m in ms] == [("A01", "машинному обучению")]


def test_acronym_is_case_sensitive():
    _, ms = mentions_for("Навыки:\nhtml, ml, ML", {"A01": ["ML"]})
    assert [m.text for m in ms] == ["ML"]


def test_longest_alias_wins():
    _, ms = mentions_for("Навыки:\nApache Spark", {"D04": ["Spark", "Apache Spark"]})
    assert [m.text for m in ms] == ["Apache Spark"]


@pytest.mark.parametrize("text, level", [
    ("Навыки:\nPython", 1),
    ("Опыт работы:\nПисал сервисы на Python", 2),
    ("Навыки:\nPython, использовал в учебных проектах", 2),
    ("Опыт работы:\nРуководил разработкой на Python", 3),
    ("Цель: стать разработчиком на Python", 0),
])
def test_level_rules(text, level):
    units, ms = mentions_for(text, {"C07": ["Python"]})
    assert estimate_level(ms, units).level == level


def test_statuses_and_score(matrix):
    text = "Опыт работы:\nРазработка моделей машинного обучения на Python, SQL-запросы к PostgreSQL."
    result = analyze(text, matrix, roles=["DS_ML"], semantic=False).roles[0]
    by_id = {i.id: i for i in result.items}
    assert by_id["A01"].status == MET and by_id["A01"].level == 2
    assert by_id["D01"].status == MET
    assert by_id["A04"].status == MISSING
    assert by_id["C04"].status == OPTIONAL
    assert 0 < result.score < 1


def test_below_when_only_listed(matrix):
    result = analyze("Навыки:\nPython", matrix, roles=["DS_ML"], semantic=False).roles[0]
    assert next(i for i in result.items if i.id == "C07").status == BELOW


def test_industry_marks_competencies(matrix):
    result = analyze("Навыки:\nPython", matrix, roles=["DS_ML"], industry="Нефтегазовая", semantic=False).roles[0]
    marked = {i.id for i in result.items if i.industry_relevant}
    assert "A02" in marked and "C07" not in marked


def test_target_role_from_objective(matrix):
    text = "ФИО: Иванов Иван Цель: позиция Data Scientist\nНавыки:\nPython"
    assert analyze(text, matrix, semantic=False).target_role == "DS_ML"
    assert analyze("Навыки:\nPython", matrix, semantic=False).target_role is None


def test_unknown_industry_rejected(matrix):
    with pytest.raises(ValueError):
        analyze("Навыки:\nPython", matrix, industry="Космос")
