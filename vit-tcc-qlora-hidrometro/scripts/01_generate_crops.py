#!/usr/bin/env python3
"""Gera crops do hidrômetro completo com CLAHE para todo o dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.config import ensure_dir, load_yaml, resolve_path
from hidrometro.data.coco import iter_coco_samples, list_all_splits
from hidrometro.detection.predictor import load_predictor
from hidrometro.preprocessing.clahe import apply_clahe
from hidrometro.preprocessing.crop import ExpansionRatios, crop_meter
from hidrometro.preprocessing.io import imread_unicode, imwrite_unicode


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerar crops equalizados do hidrômetro")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Limitar amostras por split")
    args = parser.parse_args()

    paths = load_yaml("paths.yaml")
    prep = load_yaml("preprocessing.yaml")
    min_coverage = float(prep["crop"]["min_coverage"])
    ratios = ExpansionRatios.from_config()
    crops_root = ensure_dir(resolve_path(paths["output"]["crops"]))
    reports_dir = ensure_dir(resolve_path(paths["output"].get("reports", "reports")))

    splits = list_all_splits() if args.split == "all" else [args.split]
    predictor = load_predictor()

    summary_rows: list[dict] = []

    for split in splits:
        split_out = ensure_dir(crops_root / split)
        samples = [s for s in iter_coco_samples(split) if s.image_path.exists()]
        if args.limit:
            samples = samples[: args.limit]

        for sample in tqdm(samples, desc=f"split={split}"):
            image = imread_unicode(sample.image_path)
            row = {
                "image_id": sample.image_id,
                "split": split,
                "source_path": str(sample.image_path),
                "status": "ok",
            }
            if image is None:
                row["status"] = "read_error"
                summary_rows.append(row)
                continue

            detection = predictor.predict(image)
            if detection is None:
                row["status"] = "no_detection"
                summary_rows.append(row)
                continue

            crop, bbox_full, coverage = crop_meter(image, detection.bbox, ratios)
            if coverage < min_coverage:
                row["status"] = "low_coverage"

            crop_clahe = apply_clahe(crop)
            stem = f"{sample.image_id:06d}"
            crop_path = split_out / f"{stem}.jpg"
            meta_path = split_out / f"{stem}.meta.json"

            imwrite_unicode(crop_path, crop_clahe)
            meta = {
                "image_id": sample.image_id,
                "source_path": str(sample.image_path),
                "file_name": sample.file_name,
                "split": split,
                "bbox_display": list(detection.bbox),
                "bbox_full_meter": list(bbox_full),
                "detection_score": detection.score,
                "expansion_ratios": {
                    "left": ratios.left,
                    "right": ratios.right,
                    "top": ratios.top,
                    "bottom": ratios.bottom,
                },
                "coverage_ratio": coverage,
                "crop_path": str(crop_path.relative_to(ROOT)),
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            row.update({"coverage_ratio": coverage, "crop_path": str(crop_path)})
            summary_rows.append(row)

    report_path = reports_dir / "crop_generation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    ok = sum(1 for r in summary_rows if r.get("status") == "ok")
    print(f"Relatório: {report_path}")
    print(f"Crops gerados com sucesso: {ok}/{len(summary_rows)}")


if __name__ == "__main__":
    main()
