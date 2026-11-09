from datetime import date

from competency_match.text import experience_years, section_of_heading, split_units, stem, tokenize


def test_tokens_keep_technical_names():
    texts = [t.text for t in tokenize("Знаю C++, scikit-learn, CI/CD и Node.js.")]
    assert texts == ["Знаю", "C++", "scikit-learn", "CI/CD", "и", "Node.js"]


def test_cyrillic_hyphen_split_matches_space():
    assert [t.norm for t in tokenize("промпт-инжиниринг")] == [t.norm for t in tokenize("Промпт- инжиниринг")]


def test_stem_merges_case_forms():
    assert stem("машинного") == stem("машинное") == stem("машинному")
    assert stem("обучения") == stem("обучение")
    assert stem("нейронных") == stem("нейронные") and stem("сетей") == stem("сети")


def test_headings_detected_without_false_hits():
    assert section_of_heading("Опыт (пример):") == "experience"
    assert section_of_heading("Ключевые навыки:") == "skills"
    assert section_of_heading("Машинное обучение: Scikit-learn") is None
    assert section_of_heading("Аналитик данных (ООО «ТехноСистемы», 2022–2024):") is None


def test_split_units_keeps_dotted_numbers_and_sections():
    units = split_units("Навыки:\nPython (06.02).\nОпыт работы:\nРазработка моделей. Участие в проектах.")
    assert [(u.section, u.text) for u in units] == [
        ("skills", "Python (06.02)."), ("experience", "Разработка моделей."), ("experience", "Участие в проектах.")]


def test_experience_years_merge_overlaps():
    units = split_units("Опыт работы:\nКомпания А, 2018–2021\nКомпания Б, 2020 — н.в.")
    assert experience_years(units, today=date(2024, 5, 1)) == 6
