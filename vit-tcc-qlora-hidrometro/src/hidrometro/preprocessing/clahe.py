"""CLAHE sobre o crop completo do hidrômetro."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from hidrometro.config import load_yaml


def apply_clahe(image_bgr: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or load_yaml("preprocessing.yaml")["clahe"]
    clip_limit = float(cfg.get("clip_limit", 2.0))
    tile = cfg.get("tile_grid_size", [8, 8])
    tile_grid_size = (int(tile[0]), int(tile[1]))

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l_channel)
    merged = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
