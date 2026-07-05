#!/usr/bin/env python3
"""Gera grid visual para calibrar ratios de expansão do hidrômetro completo."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.config import ensure_dir, load_yaml, project_root, resolve_path
from hidrometro.data.coco import iter_coco_samples
from hidrometro.detection.predictor import load_predictor
from hidrometro.preprocessing.clahe import apply_clahe
from hidrometro.preprocessing.crop import crop_meter, draw_bboxes
from hidrometro.preprocessing.io import imread_unicode, imwrite_unicode


def build_grid(tiles: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not tiles:
        raise ValueError("Nenhum tile para montar grid.")
    target_h = max(t.shape[0] for t in tiles)
    target_w = max(t.shape[1] for t in tiles)
    normalized = []
    for tile in tiles:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        h, w = tile.shape[:2]
        canvas[:h, :w] = tile
        normalized.append(canvas)
    rows = []
    for i in range(0, len(normalized), cols):
        row_tiles = normalized[i : i + cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros_like(normalized[0]))
        rows.append(cv2.hconcat(row_tiles))
    return cv2.vconcat(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrar crop expand_to_full_meter")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prep = load_yaml("preprocessing.yaml")
    num_samples = args.num_samples or prep["calibration"]["num_samples"]
    output_dir = ensure_dir(resolve_path(prep["calibration"]["output_dir"]))

    random.seed(args.seed)
    samples = list(iter_coco_samples("train"))
    random.shuffle(samples)
    samples = [s for s in samples if s.image_path.exists()][:num_samples]

    predictor = load_predictor()
    tiles: list[np.ndarray] = []

    for sample in samples:
        image = imread_unicode(sample.image_path)
        if image is None:
            continue
        detection = predictor.predict(image)
        if detection is None:
            continue
        crop, bbox_full, coverage = crop_meter(image, detection.bbox)
        clahe_crop = apply_clahe(crop)
        overlay = draw_bboxes(image, detection.bbox, bbox_full)
        thumb_overlay = cv2.resize(overlay, (640, 480))
        thumb_crop = cv2.resize(clahe_crop, (640, 480))
        combined = cv2.vconcat([thumb_overlay, thumb_crop])
        cv2.putText(
            combined,
            f"id={sample.image_id} cov={coverage:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        tiles.append(combined)

    if not tiles:
        raise SystemExit("Nenhuma amostra válida encontrada. Verifique dataset e pesos Detectron2.")

    grid = build_grid(tiles)
    out_path = output_dir / "calibration_grid.jpg"
    imwrite_unicode(out_path, grid)
    print(f"Grid salvo em: {out_path}")
    print(f"Amostras processadas: {len(tiles)}")


if __name__ == "__main__":
    main()
