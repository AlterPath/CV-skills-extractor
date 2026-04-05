# CV Skills Extractor

[Русская версия](README.md)

A capstone project for Samsung Innovation Campus (Artificial Intelligence track) and the elite study groups, 2025. The system extracts skills from resume text and matches them against a competency matrix of IT roles to show how well a candidate fits a role.

## How it works

1. **Competency matrix** (`alliance_matrix.json`) holds roles, competencies and levels. A role's skills are clustered: `sentence-transformers` embeddings (`all-MiniLM-L6-v2`), then `MiniBatchKMeans`, visualized with UMAP.
2. **Resume parsing**: text from DOCX or TXT, preprocessing with spaCy (`ru_core_news_md`).
3. **Skill extraction**:
   - embedding-based matching with the matrix skills;
   - a custom PyTorch BIO skill tagger (BiLSTM, `model_resume.py`).
4. **Role comparison**: a table matching the candidate's skills to the role requirements and a list of missing skills. A cross-encoder refines "resume skill – matrix skill" pairs.

The 2026 version of the notebook adds NER experiments with natasha.

## Layout

| Path | Contents |
|---|---|
| `model.ipynb` | The whole pipeline: matrix clustering, resume parsing, matching, the tagger |
| `model_resume.py` | The BiLSTM skill tagger (BIO tagging) |
| `data/` | The competency matrix, skill clusters, a sample vacancy and a test resume (fictional) |
| `models/` | Skill embeddings and the fitted clustering model |

## Quick start

```bash
pip install sentence-transformers scikit-learn spacy umap-learn torch pandas python-docx natasha
python -m spacy download ru_core_news_md
jupyter notebook model.ipynb
```

Notebook text is in Russian.

## Tech stack

Python, sentence-transformers, scikit-learn, spaCy, natasha, PyTorch, UMAP, pandas.

## License

[MIT](LICENSE)
