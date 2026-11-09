"""Обучает BiLSTM-теггер навыков, проверяет его и сохраняет модель и метрики.

    python scripts/train_tagger.py

Модель: models/skill_tagger.pt, метрики: results/tagger_metrics.json.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

from competency_match.extract import AliasMatcher, find_mentions, load_aliases  # noqa: E402
from competency_match.tagger import TaggerRunner, Vocab, build_corpus, evaluate, split_aliases, train  # noqa: E402
from competency_match.text import read_resume, split_units, tokenize  # noqa: E402

SEED = 7


def norm(text: str) -> str:
    return " ".join(t.norm for t in tokenize(text))


def set_metrics(found: set[str], gold: set[str]) -> dict:
    tp = len(found & gold)
    p = tp / len(found) if found else 0.0
    r = tp / len(gold) if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "missed": sorted(gold - found), "extra": sorted(found - gold)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    aliases = load_aliases()
    seen, unseen = split_aliases(aliases, unseen_share=0.25, seed=SEED)
    rng = random.Random(SEED)
    train_set = build_corpus(seen, 4000, rng)
    val_set = build_corpus(seen, 600, rng)
    test_seen = build_corpus(seen, 600, rng)
    test_unseen = build_corpus(unseen, 600, rng, neutral_share=0.0)
    print(f"ключевых слов: {len(seen)} в обучении, {len(unseen)} отложено; предложений: {len(train_set)} в обучении")

    vocab = Vocab.build(train_set)
    model = train(train_set, val_set, vocab)
    runner = TaggerRunner(model, vocab)

    gold_data = json.loads((V2 / "data" / "gold_sample_resume.json").read_text(encoding="utf-8"))
    gold = {norm(s) for s in gold_data["skills"]}
    units = split_units(read_resume(V2.parent / "data" / "sample_resume.txt"))
    by_dict = {norm(m.text) for m in find_mentions(units, AliasMatcher(aliases))}
    by_tagger = {norm(fragment) for _, _, fragment in runner.fragments(units)}

    metrics = {
        "setup": {"seen_aliases": len(seen), "unseen_aliases": len(unseen), "train_sentences": len(train_set),
                  "seed": SEED},
        "synthetic_seen_skills": evaluate(model, test_seen, vocab),
        "synthetic_unseen_skills": evaluate(model, test_unseen, vocab),
        "sample_resume_dictionary": set_metrics(by_dict, gold),
        "sample_resume_tagger": set_metrics(by_tagger, gold),
        "sample_resume_union": set_metrics(by_dict | by_tagger, gold),
    }
    (V2 / "models").mkdir(exist_ok=True)
    runner.save(V2 / "models" / "skill_tagger.pt")
    (V2 / "results").mkdir(exist_ok=True)
    (V2 / "results" / "tagger_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                                                         encoding="utf-8")
    for name, value in metrics.items():
        if name != "setup":
            print(f"{name}: P={value['precision']:.3f} R={value['recall']:.3f} F1={value['f1']:.3f}")


if __name__ == "__main__":
    main()
