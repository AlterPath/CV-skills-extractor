# CV Skills Extractor

[English version](README.en.md)

Итоговый проект курса Samsung Innovation Campus (трек «Искусственный интеллект») и элитных групп, 2025. Система извлекает навыки из текста резюме и сопоставляет их с матрицей компетенций IT-ролей, чтобы показать, насколько кандидат подходит под роль.

## Как работает

1. **Матрица компетенций** (`alliance_matrix.json`) содержит роли, компетенции и уровни. Навыки роли разбиваются на кластеры: эмбеддинги `sentence-transformers` (`all-MiniLM-L6-v2`), затем `MiniBatchKMeans`, визуализация через UMAP.
2. **Разбор резюме**: текст из DOCX или TXT, предобработка через spaCy (`ru_core_news_md`).
3. **Извлечение навыков**:
   - сопоставление с навыками матрицы по эмбеддингам;
   - собственный BIO-теггер навыков на PyTorch (BiLSTM, `model_resume.py`).
4. **Сравнение с ролью**: таблица совпадений навыков кандидата с требованиями роли и список недостающих навыков. Для уточнения пар «навык из резюме – навык матрицы» используется cross-encoder.

В версии 2026 года в ноутбуке добавлены эксперименты с NER на natasha.

## Структура

| Путь | Что внутри |
|---|---|
| `model.ipynb` | Весь пайплайн: кластеризация матрицы, разбор резюме, сопоставление, теггер |
| `model_resume.py` | BiLSTM-теггер навыков (разметка BIO) |
| `data/` | Матрица компетенций, кластеры навыков, пример вакансии и тестовое резюме (выдуманное) |
| `models/` | Эмбеддинги навыков и обученная модель кластеризации |

## Запуск

```bash
pip install sentence-transformers scikit-learn spacy umap-learn torch pandas python-docx natasha
python -m spacy download ru_core_news_md
jupyter notebook model.ipynb
```

## Стек

Python, sentence-transformers, scikit-learn, spaCy, natasha, PyTorch, UMAP, pandas.

## Лицензия

[MIT](LICENSE)
