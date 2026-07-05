#!/usr/bin/env python3
"""Treina adaptador LoRA sobre Florence-2-large."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.config import load_yaml, resolve_path
from hidrometro.training.train_qlora import train_qlora


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino QLoRA Florence-2")
    parser.add_argument("--config", default="qlora.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    paths = load_yaml("paths.yaml")
    sft_root = resolve_path(paths["output"]["sft"])
    train_jsonl = sft_root / "train.jsonl"
    val_jsonl = sft_root / "val.jsonl"

    if not train_jsonl.exists():
        raise SystemExit(
            f"Dataset SFT ausente: {train_jsonl}. Execute scripts/03_export_label_studio.py --mode export"
        )

    val_path = val_jsonl if val_jsonl.exists() else None
    out_dir = train_qlora(
        train_jsonl=train_jsonl,
        val_jsonl=val_path,
        config_name=args.config,
        num_epochs=args.epochs,
    )
    print(f"Adaptador LoRA salvo em: {out_dir}")


if __name__ == "__main__":
    main()
