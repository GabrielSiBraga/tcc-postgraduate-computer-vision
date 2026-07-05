#!/usr/bin/env python3
"""Avaliação: acurácia, precisão, CER e F1 no conjunto de teste."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.autolabel.label_studio import load_meter_label
from hidrometro.config import ensure_dir, load_yaml, resolve_path
from hidrometro.evaluation.benchmark import adapter_size_mb, benchmark_throughput
from hidrometro.evaluation.metrics import (
    compute_metrics,
    metrics_summary_table,
    run_vlm_evaluation,
)
from hidrometro.pipeline.hybrid import HybridPipeline
from hidrometro.preprocessing.io import imread_unicode


def load_ground_truth(split: str) -> dict[str, dict]:
    paths = load_yaml("paths.yaml")
    validated = resolve_path(paths["output"]["autolabel_validated"]) / split
    raw = resolve_path(paths["output"]["autolabel_raw"]) / split
    source = validated if validated.exists() and any(validated.glob("*.json")) else raw
    return {
        label_path.stem: load_meter_label(label_path)
        for label_path in source.glob("*.json")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliar pipeline híbrido")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="0 = todas as amostras")
    args = parser.parse_args()

    paths = load_yaml("paths.yaml")
    reports_dir = ensure_dir(resolve_path("reports"))
    gt = load_ground_truth(args.split)
    crops_dir = resolve_path(paths["output"]["crops"]) / args.split

    crop_paths = sorted(p for p in crops_dir.glob("*.jpg") if p.stem in gt)
    if args.limit > 0:
        crop_paths = crop_paths[: args.limit]

    pipeline = HybridPipeline()
    rows = run_vlm_evaluation(pipeline.vlm, crop_paths, gt)
    metrics = compute_metrics(rows)

    report: dict = {
        "split": args.split,
        **metrics,
        "summary_table": metrics_summary_table(metrics),
        "adapter_size_mb": adapter_size_mb(resolve_path(paths["output"]["lora_adapter"])),
    }

    if crops_dir.exists() and crop_paths:
        def _run(path: Path) -> None:
            image = imread_unicode(path)
            if image is not None:
                pipeline.vlm.predict(image)

        bench = benchmark_throughput(_run, crop_paths[: min(len(crop_paths), 20)])
        report["vram_peak_mb"] = bench.vram_peak_mb
        report["throughput_ips"] = bench.throughput_ips

    out_path = reports_dir / f"evaluation_{args.split}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Resultados (percentuais) ===")
    for row in metrics_summary_table(metrics):
        print(f"  {row['métrica']}: {row['valor']}")
    print(f"\nRelatório completo: {out_path}")


if __name__ == "__main__":
    main()
