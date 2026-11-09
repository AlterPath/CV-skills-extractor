"""BIO-теггер навыков: BiLSTM поверх эмбеддингов слов, символьной свёртки и признаков написания.

Что исправлено по сравнению с v1 (`model_resume.py`):
- метка `O` больше не совпадает с индексом паддинга. Отступы в метках заполняются `-100` и игнорируются в лоссе,
  а `O` учитывается. Поэтому лосс не бывает NaN, даже если в батче одни `O`;
- обучение идёт на тысячах предложений с отложенной выборкой и метрикой F1 по сущностям, а не на двух предложениях;
- навыки, которых модель не видела, проверяются отдельно: часть ключевых слов в обучение не попадает;
- предсказания используются: навыки, которых нет в словаре, попадают в отчёт.

Разметка «серебряная»: предложения собираются из шаблонов и ключевых слов словаря.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .text import CYRILLIC_RE, NON_EVIDENCE_SECTIONS, Token, Unit, tokenize

LABELS = ("O", "B-SKILL", "I-SKILL")
O, B, I = 0, 1, 2
IGNORE = -100          # метка для отступов, лосс её пропускает
PAD, UNK = 0, 1        # индексы в словарях слов и символов
MAX_CHARS = 20
# знаки препинания идут в теггер отдельными токенами: так видна граница между «SciPy),» и «SQL»
PUNCT_RE = re.compile(r"[,;:()«»\"“”|—]")

TEMPLATES = [
    "Опыт работы с {s}.", "Опыт работы с {s} и {s}.", "Навыки: {s}, {s}, {s}.", "Ключевые навыки: {s}, {s}.",
    "Стек: {s}, {s}, {s}, {s}.", "Разработка моделей с использованием {s}.", "Использовал {s} для анализа данных клиентов.",
    "Построил пайплайн на {s} и {s}.", "Внедрил {s} в продакшн.", "Настроил {s} для команды аналитиков.",
    "Участвовал в проекте по {s}.", "Знание {s} на уровне уверенного пользователя.", "Прошла курс «{s}» на Stepik.",
    "Сертификат по {s} (2023).", "Писал скрипты на {s}.", "Администрирование {s} и {s}.",
    "Отвечала за {s} в отделе аналитики.", "Применял {s} в задачах прогнозирования спроса.", "Изучаю {s}.",
    "Руководил командой из пяти человек, стек {s} и {s}.", "Владею {s}, {s}.",
    "Технологии: {s} / {s} / {s}.", "Перевёл отчётность с Excel на {s}.", "Ускорил расчёты в 3 раза за счёт {s}.",
    # структура настоящих резюме: перечни по категориям, скобки, образование
    "Инструменты: {s} ({s}, {s}), {s}.", "Базы данных: {s}, {s}.", "Языки программирования: {s} ({s}), {s}.",
    "Администрирование {s} ({s}, {s}).", "Участие в проектах по {s} ({s}).", "Специализация: {s} и {s}.",
    "Разработка моделей {s} (улучшение точности на 25%).", "{s} — опыт 2 года.", "Курс «{s}» (Яндекс Практикум, 2024).",
    "Публикации на Kaggle (призовые места в конкурсах по {s}).", "Сертификат «Специалист по {s}» (2024).",
    "Аналитик данных (ООО «Альфа», 2021–2023): {s}, {s}.",
    "Experience with {s} and {s}.", "Skills: {s}, {s}, {s}.", "Built data pipelines with {s}.",
    "Used {s} to deploy models.", "Tech stack: {s}, {s}.", "Familiar with {s}.",
]
NAMES = ["Иванов Алексей Петрович", "Смирнова Анна Сергеевна", "Петров Дмитрий Олегович", "Кузнецова Мария Игоревна"]
UNIVERSITIES = ["НИУ ВШЭ", "МГТУ им. Н. Э. Баумана", "Университет ИТМО", "МФТИ", "СПбГУ", "РТУ МИРЭА", "МГУ"]
NEUTRAL = [
    "Ответственный, быстро обучаюсь, люблю работать в команде.", "Английский язык: B2.", "Готов к командировкам.",
    "Работал в ООО «Вектор», Москва, 2019–2021.", "Компания Ozon, отдел аналитики.", "Образование: МГУ, факультет ВМК.",
    "Увлекаюсь шахматами и бегом.", "Подготовка еженедельных отчётов для руководства.", "Coordinated meetings with stakeholders.",
    "Опыт работы в банке, 3 года.", "Компания Yandex, Санкт-Петербург.", "Наставник в школе программирования для детей.",
    "Победитель городской олимпиады по математике.", "Водительские права категории B.",
    "ФИО: {name}.", "Цель: Трудоустройство на позицию аналитика с фокусом на развитие.", "Образование: {uni}.",
    "{uni} / {uni}", "Программа: Прикладная математика и информатика (01.03.02).", "Аккредитация: до 2027 г.",
    "Специализация: прикладная информатика.", "«Специалист по информационным системам» (Приказ Минтруда № 896н).",
    "Профессиональный стандарт 06.015 «Специалист по информационным системам».", "«Системный аналитик» (06.02).",
    "Английский: C1 (чтение технической документации).", "Улучшение точности прогноза на 15%.",
    "Аналитик (ООО «Бета», 2020–2022):", "Телефон: +7 900 000-00-00.", "(соответствует профстандартам).",
]


def entity_spans(labels: list[int]) -> set[tuple[int, int]]:
    """Сущности (начало, конец) из BIO-меток; `I` без `B` тоже открывает сущность."""
    spans, start = set(), None
    for i, label in enumerate(list(labels) + [O]):
        if label == B or (label == I and start is None) or label == O:
            if start is not None:
                spans.add((start, i))
                start = None
            if label in (B, I):
                start = i
    return spans


def tagger_tokens(text: str) -> list[Token]:
    """Слова (как в `tokenize`) и знаки препинания между ними."""
    words = tokenize(text)
    inside = [(t.start, t.end) for t in words]
    punct = [Token(m.group(), m.group(), m.start(), m.end()) for m in PUNCT_RE.finditer(text)
             if not any(s <= m.start() < e for s, e in inside)]
    return sorted(words + punct, key=lambda t: t.start)


def make_sentence(template: str, pool: list[str], rng: random.Random) -> tuple[list[str], list[int]]:
    """Подставляет навыки в шаблон и размечает токены по позициям подстановок."""
    template = template.replace("{name}", rng.choice(NAMES))
    while "{uni}" in template:
        template = template.replace("{uni}", rng.choice(UNIVERSITIES), 1)
    text, spans, rest = "", [], template
    while "{s}" in rest:
        head, rest = rest.split("{s}", 1)
        text += head
        skill = rng.choice(pool)
        if skill.isascii() and rng.random() < 0.15:
            skill = skill.lower()
        spans.append((len(text), len(text) + len(skill)))
        text += skill
    text += rest
    tokens = tagger_tokens(text)
    labels = []
    for tok in tokens:
        inside = [s for s in spans if s[0] <= tok.start < s[1]]
        labels.append(O if not inside else B if tok.start == inside[0][0] else I)
    return [t.text for t in tokens], labels


def build_corpus(pool: list[str], n: int, rng: random.Random, neutral_share: float = 0.25):
    corpus = []
    for _ in range(n):
        template = rng.choice(NEUTRAL) if rng.random() < neutral_share else rng.choice(TEMPLATES)
        corpus.append(make_sentence(template, pool, rng))
    return corpus


def split_aliases(aliases: dict[str, list[str]], unseen_share: float, seed: int) -> tuple[list[str], list[str]]:
    """Делит ключевые слова на «видимые» при обучении и «новые» для проверки обобщения."""
    unique = sorted({a for words in aliases.values() for a in words}, key=str.lower)
    rng = random.Random(seed)
    rng.shuffle(unique)
    cut = int(len(unique) * (1 - unseen_share))
    return unique[:cut], unique[cut:]


def token_features(token: str) -> list[float]:
    return [float(bool(token) and token[0].isascii() and token[0].isalpha()), float(bool(CYRILLIC_RE.search(token))),
            float(any(c.isupper() for c in token)), float(any(not c.isalpha() for c in token))]


@dataclass
class Vocab:
    words: dict[str, int]
    chars: dict[str, int]

    @classmethod
    def build(cls, corpus) -> "Vocab":
        words, chars = {"<pad>": PAD, "<unk>": UNK}, {"<pad>": PAD, "<unk>": UNK}
        for tokens, _ in corpus:
            for tok in tokens:
                words.setdefault(tok.lower(), len(words))
                for ch in tok[:MAX_CHARS]:
                    chars.setdefault(ch, len(chars))
        return cls(words, chars)


class SkillTagger(nn.Module):
    def __init__(self, n_words: int, n_chars: int, word_dim: int = 64, char_dim: int = 24, char_filters: int = 48,
                 hidden: int = 64, n_features: int = 4, dropout: float = 0.3):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=PAD)
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=PAD)
        self.char_cnn = nn.Conv1d(char_dim, char_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(word_dim + char_filters + n_features, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, len(LABELS))

    def forward(self, words, chars, feats, lengths):
        batch, steps, n_chars = chars.shape
        c = self.char_emb(chars.view(batch * steps, n_chars)).transpose(1, 2)
        c = torch.relu(self.char_cnn(c)).max(dim=2).values.view(batch, steps, -1)
        x = self.dropout(torch.cat([self.word_emb(words), c, feats], dim=-1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=steps)
        return self.out(self.dropout(out))


def encode_batch(batch, vocab: Vocab, word_dropout: float = 0.0, rng: random.Random | None = None):
    """Тензоры батча. Метки отступов = IGNORE, поэтому `O` (индекс 0) не путается с паддингом."""
    steps = max(len(tokens) for tokens, _ in batch)
    words = torch.full((len(batch), steps), PAD, dtype=torch.long)
    chars = torch.full((len(batch), steps, MAX_CHARS), PAD, dtype=torch.long)
    feats = torch.zeros((len(batch), steps, 4))
    labels = torch.full((len(batch), steps), IGNORE, dtype=torch.long)
    lengths = torch.tensor([len(tokens) for tokens, _ in batch])
    for row, (tokens, tags) in enumerate(batch):
        for col, tok in enumerate(tokens):
            wid = vocab.words.get(tok.lower(), UNK)
            if rng is not None and word_dropout and rng.random() < (word_dropout * (2 if tags[col] != O else 1)):
                wid = UNK
            words[row, col] = wid
            for k, ch in enumerate(tok[:MAX_CHARS]):
                chars[row, col, k] = vocab.chars.get(ch, UNK)
            feats[row, col] = torch.tensor(token_features(tok))
            labels[row, col] = tags[col]
    return words, chars, feats, lengths, labels


def evaluate(model: SkillTagger, corpus, vocab: Vocab, batch_size: int = 64) -> dict[str, float]:
    """Точность, полнота и F1 по сущностям (точное совпадение границ)."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i + batch_size]
            words, chars, feats, lengths, _ = encode_batch(batch, vocab)
            pred = model(words, chars, feats, lengths).argmax(-1)
            for row, (tokens, tags) in enumerate(batch):
                gold = entity_spans(tags)
                got = entity_spans(pred[row, :len(tokens)].tolist())
                tp, fp, fn = tp + len(gold & got), fp + len(got - gold), fn + len(gold - got)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def train(train_set, val_set, vocab: Vocab, *, epochs: int = 12, batch_size: int = 32, lr: float = 3e-3,
          word_dropout: float = 0.15, seed: int = 13, log=print) -> SkillTagger:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SkillTagger(len(vocab.words), len(vocab.chars))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    best_f1, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_set)))
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), batch_size):
            batch = [train_set[j] for j in order[i:i + batch_size]]
            words, chars, feats, lengths, labels = encode_batch(batch, vocab, word_dropout, rng)
            logits = model(words, chars, feats, lengths)
            loss = criterion(logits.view(-1, len(LABELS)), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(batch)
        metrics = evaluate(model, val_set, vocab)
        log(f"эпоха {epoch:2d}: лосс {total / len(train_set):.4f}, F1 на валидации {metrics['f1']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


class TaggerRunner:
    """Обученный теггер для анализа резюме."""

    def __init__(self, model: SkillTagger, vocab: Vocab):
        self.model, self.vocab = model.eval(), vocab

    def spans(self, text: str) -> list[tuple[int, int, str]]:
        """Навыки в тексте: (начало, конец, фрагмент) в символах."""
        tokens = tagger_tokens(text)
        if not tokens:
            return []
        words, chars, feats, lengths, _ = encode_batch([([t.text for t in tokens], [O] * len(tokens))], self.vocab)
        with torch.no_grad():
            pred = self.model(words, chars, feats, lengths).argmax(-1)[0].tolist()
        result = []
        for a, b in sorted(entity_spans(pred)):
            if all(PUNCT_RE.fullmatch(t.text) for t in tokens[a:b]):
                continue
            start, end = tokens[a].start, tokens[b - 1].end
            result.append((start, end, text[start:end]))
        return result

    def fragments(self, units: list[Unit]) -> list[tuple[int, int, str]]:
        """Навыки по всему резюме, кроме шапки и цели, без подписей полей и кодов."""
        result = []
        for unit in units:
            if unit.section in NON_EVIDENCE_SECTIONS:
                continue
            for start, end, fragment in self.spans(unit.text):
                if unit.text[end:].lstrip().startswith(":") or sum(c.isdigit() for c in fragment) > 1:
                    continue        # подпись поля («Специализация:») или код («06.02»)
                result.append((start + unit.start, end + unit.start, fragment))
        return result

    def extra_skills(self, units: list[Unit], mentions) -> list[str]:
        """Навыки, которые нашёл теггер, но не словарь."""
        known = [(m.start, m.end) for m in mentions if not m.semantic]
        found: dict[str, str] = {}
        for start, end, fragment in self.fragments(units):
            if not any(s < end and start < e for s, e in known):
                found.setdefault(fragment.lower(), fragment)
        return sorted(found.values(), key=str.lower)

    def save(self, path: str | Path) -> None:
        torch.save({"state": self.model.state_dict(), "words": self.vocab.words, "chars": self.vocab.chars}, path)

    @classmethod
    def load(cls, path: str | Path) -> "TaggerRunner":
        data = torch.load(path, map_location="cpu", weights_only=True)
        vocab = Vocab(data["words"], data["chars"])
        model = SkillTagger(len(vocab.words), len(vocab.chars))
        model.load_state_dict(data["state"])
        return cls(model, vocab)
