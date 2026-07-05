"""Expansão assimétrica do bbox display para crop do hidrômetro completo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hidrometro.config import load_yaml


@dataclass
class ExpansionRatios:
    left: float = 1.5
    right: float = 1.5
    top: float = 1.0
    bottom: float = 2.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "ExpansionRatios":
        cfg = config or load_yaml("preprocessing.yaml")["crop"]["ratios"]
        return cls(
            left=float(cfg["left"]),
            right=float(cfg["right"]),
            top=float(cfg["top"]),
            bottom=float(cfg["bottom"]),
        )


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1_i = max(0, int(round(x1)))
    y1_i = max(0, int(round(y1)))
    x2_i = min(width, int(round(x2)))
    y2_i = min(height, int(round(y2)))
    if x2_i <= x1_i:
        x2_i = min(width, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(height, y1_i + 1)
    return x1_i, y1_i, x2_i, y2_i


def expand_to_full_meter(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    ratios: ExpansionRatios | None = None,
) -> tuple[int, int, int, int]:
    ratios = ratios or ExpansionRatios.from_config()
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    new_x1 = x1 - ratios.left * width
    new_x2 = x2 + ratios.right * width
    new_y1 = y1 - ratios.top * height
    new_y2 = y2 + ratios.bottom * height
    return clamp_bbox(new_x1, new_y1, new_x2, new_y2, image_shape)


def coverage_ratio(
    bbox_display: tuple[int, int, int, int],
    bbox_full: tuple[int, int, int, int],
) -> float:
    dx1, dy1, dx2, dy2 = bbox_display
    fx1, fy1, fx2, fy2 = bbox_full
    display_area = max(1, (dx2 - dx1) * (dy2 - dy1))
    ix1 = max(dx1, fx1)
    iy1 = max(dy1, fy1)
    ix2 = min(dx2, fx2)
    iy2 = min(dy2, fy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / display_area


def crop_meter(
    image: np.ndarray,
    bbox_display: tuple[int, int, int, int],
    ratios: ExpansionRatios | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    bbox_full = expand_to_full_meter(bbox_display, image.shape, ratios)
    x1, y1, x2, y2 = bbox_full
    crop = image[y1:y2, x1:x2].copy()
    ratio = coverage_ratio(bbox_display, bbox_full)
    return crop, bbox_full, ratio


def draw_bboxes(
    image: np.ndarray,
    bbox_display: tuple[int, int, int, int],
    bbox_full: tuple[int, int, int, int],
) -> np.ndarray:
    import cv2

    overlay = image.copy()
    dx1, dy1, dx2, dy2 = bbox_display
    fx1, fy1, fx2, fy2 = bbox_full
    cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0, 255, 0), 3)
    cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
    cv2.putText(
        overlay,
        "full_meter",
        (fx1, max(20, fy1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        overlay,
        "display",
        (dx1, max(20, dy1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    return overlay
