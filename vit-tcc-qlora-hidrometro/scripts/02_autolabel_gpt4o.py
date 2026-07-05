#!/usr/bin/env python3
"""Auto-labelling em batch via GPT-4o sobre crops CLAHE."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.autolabel.openai_labeler import OpenAILabeler
from hidrometro.config import ensure_dir, load_yaml, resolve_path


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Auto-labelling GPT-4o")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Interrompe no primeiro erro (padrão: continua e registra falhas)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocessa imagens mesmo se o JSON já existir",
    )
    args = parser.parse_args()

    paths = load_yaml("paths.yaml")
    crops_root = resolve_path(paths["output"]["crops"])
    raw_root = ensure_dir(resolve_path(paths["output"]["autolabel_raw"]))
    reports_dir = ensure_dir(resolve_path("reports"))

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    labeler = OpenAILabeler()
    failures: list[dict[str, str]] = []
    success = 0
    skipped = 0

    for split in splits:
        split_dir = crops_root / split
        if not split_dir.exists():
            print(f"Split ausente: {split_dir}")
            continue
        images = sorted(split_dir.glob("*.jpg"))
        if args.limit:
            images = images[: args.limit]

        for image_path in tqdm(images, desc=f"autolabel {split}"):
            out_path = raw_root / split / f"{image_path.stem}.json"
            if out_path.exists() and not args.force:
                skipped += 1
                continue
            try:
                labeler.label_and_save(image_path, out_path)
                success += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "split": split,
                        "image": str(image_path),
                        "error": str(exc),
                    }
                )
                if args.fail_fast:
                    raise
                tqdm.write(f"Falha em {image_path.name}: {exc}")

    report_path = reports_dir / "autolabel_failures.csv"
    if failures:
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "image", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"Falhas registradas em: {report_path}")

    print(f"Auto-labels salvos em: {raw_root}")
    print(f"Sucesso: {success} | Pulados: {skipped} | Falhas: {len(failures)}")


if __name__ == "__main__":
    main()
