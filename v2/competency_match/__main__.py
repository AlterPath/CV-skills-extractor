"""Командная строка: python -m competency_match resume.docx [--role DS_ML] [--industry Нефтегазовая]."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assess import analyze
from .matrix import load_matrix
from .report import to_json, to_markdown
from .text import read_resume

V2 = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = V2.parent / "data" / "alliance_matrix.json"
DEFAULT_TAGGER = V2 / "models" / "skill_tagger.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m competency_match",
                                     description="Сравнивает резюме с ролями матрицы компетенций Альянса.")
    parser.add_argument("resume", type=Path, help="резюме: .txt, .docx или .pdf")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX, help="alliance_matrix.json")
    parser.add_argument("--role", action="append", help="код роли (DS_ML, DE, TA_AI, MGR_AI), можно несколько")
    parser.add_argument("--industry", help="отрасль из матрицы, её компетенции будут отмечены")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("--details", type=int, default=1, help="сколько ролей расписать подробно")
    parser.add_argument("--tagger", type=Path, default=DEFAULT_TAGGER, help="обученный теггер (scripts/train_tagger.py)")
    parser.add_argument("--no-semantic", action="store_true", help="не искать косвенные упоминания")
    parser.add_argument("--out", type=Path, help="куда сохранить отчёт")
    args = parser.parse_args(argv)

    matrix = load_matrix(args.matrix)
    unknown = [r for r in args.role or [] if r not in matrix.roles]
    if unknown:
        parser.error(f"неизвестные роли: {', '.join(unknown)}; есть: {', '.join(matrix.roles)}")
    tagger = None
    if args.tagger and args.tagger.exists():
        try:
            from .tagger import TaggerRunner
        except ImportError:
            print("PyTorch не установлен, отчёт без теггера", file=sys.stderr)
        else:
            tagger = TaggerRunner.load(args.tagger)
    analysis = analyze(read_resume(args.resume), matrix, roles=args.role, industry=args.industry,
                       semantic=not args.no_semantic, tagger=tagger, source=args.resume.name)
    text = to_json(analysis) if args.format == "json" else to_markdown(analysis, matrix, args.details)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
