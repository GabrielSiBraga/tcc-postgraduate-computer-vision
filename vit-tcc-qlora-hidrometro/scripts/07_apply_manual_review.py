#!/usr/bin/env python3
"""Aplica correções do CSV de revisão manual nos labels validados e regenera SFT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.autolabel.label_studio import apply_manual_review_csv
from hidrometro.config import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aplicar label_studio_manual_review.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=resolve_path("reports") / "label_studio_manual_review.csv",
        help="CSV de revisão manual",
    )
    parser.add_argument(
        "--validated-dir",
        type=Path,
        default=None,
        help="Raiz de data/autolabel/validated (default: paths.yaml)",
    )
    parser.add_argument(
        "--export-sft",
        action="store_true",
        help="Regenerar data/sft/*.jsonl após aplicar correções",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV não encontrado: {args.csv}")

    stats = apply_manual_review_csv(args.csv, args.validated_dir)
    print(f"Labels atualizados: {stats['updated']} | sem alteração/ausentes: {stats['skipped']}")

    if args.export_sft:
        import subprocess

        cmd = [sys.executable, str(ROOT / "scripts" / "03_export_label_studio.py"), "--mode", "export", "--split", "all"]
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
