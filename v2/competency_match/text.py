"""Чтение резюме, разбиение на разделы и фрагменты, токенизация с лёгким стеммингом."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9+#./\-]*")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
# окончания для грубого стемминга: «машинного обучения» и «машинное обучение» дают одну основу
ENDINGS = sorted(
    {"иями", "ями", "ами", "иях", "ях", "ах", "ией", "ием", "ем", "ом", "ого", "его", "ому", "ему", "ыми", "ими",
     "ым", "им", "ых", "их", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой", "ей", "ою", "ею", "ую", "юю",
     "ия", "ья", "ию", "ью", "ье", "ии", "ов", "ев", "ам", "ям", "а", "я", "о", "е", "ы", "и", "у", "ю", "ь", "й"},
    key=len, reverse=True)

# раздел резюме -> шаблон заголовка (сравнивается с текстом до двоеточия целиком)
SECTION_PATTERNS = {
    "objective": r"цель|о себе|желаемая должность|objective|summary|about( me)?",
    "skills": r"(ключевые |профессиональные |технические |hard )?навыки|стек( технологий)?|технологии|skills|tech stack",
    "experience": r"опыт( работы)?|места работы|work experience|experience|employment",
    "projects": r"проекты|pet-проекты|projects",
    "education": r"образование|программа|специализация|курсы|дополнительное образование|education|courses",
    "certificates": r"сертификаты|сертификация|достижения|награды|публикации|certificates|awards|publications",
    "extra": r"дополнительно|прочее|языки|знание профессиональных стандартов|additional|languages",
}
# разделы, упоминания в которых не считаются подтверждением навыка
NON_EVIDENCE_SECTIONS = {"header", "objective"}
YEAR_RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*(?:[–—-]|по)\s*((?:19|20)\d{2}|н\.?\s*в\.?|наст\w*|сейчас|present|now|current)", re.I)


@dataclass(frozen=True)
class Token:
    text: str   # как в тексте
    norm: str   # нижний регистр, «ё» -> «е», основа слова для кириллицы
    start: int
    end: int


@dataclass(frozen=True)
class Unit:
    """Фрагмент резюме (строка или её часть) с разделом, к которому он относится."""
    index: int
    text: str
    section: str
    start: int


def stem(word: str) -> str:
    word = word.lower().replace("ё", "е")
    if CYRILLIC_RE.search(word) and len(word) > 3:
        for ending in ENDINGS:
            if word.endswith(ending) and len(word) - len(ending) >= 3:
                return word[: -len(ending)]
    return word


def tokenize(text: str, offset: int = 0) -> list[Token]:
    """Токены с позициями. «C++», «scikit-learn», «CI/CD» остаются целыми, кириллица делится по дефису."""
    tokens = []
    for match in TOKEN_RE.finditer(text):
        raw = match.group().rstrip(".-/")
        if not raw:
            continue
        if "-" in raw and CYRILLIC_RE.search(raw):
            parts = [(m.group(), m.start()) for m in re.finditer(r"[^-]+", raw)]
        else:
            parts = [(raw, 0)]
        for part, shift in parts:
            start = offset + match.start() + shift
            tokens.append(Token(part, stem(part), start, start + len(part)))
    return tokens


def section_of_heading(line: str) -> str | None:
    """Если строка выглядит как заголовок раздела («Опыт (пример):»), возвращает раздел."""
    head = line.split(":", 1)[0] if ":" in line else line
    head = re.sub(r"\(.*?\)", "", head).strip().lower().replace("ё", "е")
    if not head or len(head) > 40:
        return None
    for section, pattern in SECTION_PATTERNS.items():
        if re.fullmatch(pattern, head):
            return section
    return None


def split_units(text: str) -> list[Unit]:
    """Делит резюме на фрагменты: строки, длинные строки ещё и по предложениям."""
    units: list[Unit] = []
    section = "header"
    pos = 0
    for line in text.splitlines(keepends=True):
        line_start, pos = pos, pos + len(line)
        stripped = line.strip()
        if not stripped:
            continue
        heading = section_of_heading(stripped)
        if heading:
            section = heading
            rest = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if not rest:
                continue
        offset = line_start + line.index(stripped[0])
        # граница предложения: знак препинания и пробел; «06.02», «Node.js» не режутся
        bounds = [m.span() for m in re.finditer(r"(?<=[.;!?])\s+", stripped)]
        starts = [0] + [end for _, end in bounds]
        ends = [start for start, _ in bounds] + [len(stripped)]
        for start, end in zip(starts, ends):
            piece = stripped[start:end].strip()
            if len(piece) > 1:
                units.append(Unit(len(units), piece, section, offset + start))
    return units


def experience_years(units: list[Unit], today: date | None = None) -> float:
    """Суммарный стаж по диапазонам лет в разделе «Опыт» (пересечения склеиваются)."""
    current = (today or date.today()).year
    spans = []
    for unit in units:
        if unit.section != "experience":
            continue
        for start, end in YEAR_RANGE_RE.findall(unit.text):
            end_year = int(end) if end[:2] in ("19", "20") else current
            if end_year >= int(start):
                spans.append((int(start), end_year))
    total, last_end = 0, None
    for start, end in sorted(spans):
        if last_end is not None and start < last_end:
            start = last_end
        if end > start:
            total += end - start
        last_end = max(end, last_end or end)
    return float(total)


def read_resume(path: str | Path) -> str:
    """Текст резюме из TXT, DOCX или PDF."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        import docx  # python-docx
        document = docx.Document(str(path))
        lines = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                lines.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(lines)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Для PDF нужен пакет pypdf: pip install pypdf") from exc
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось определить кодировку файла {path}")
