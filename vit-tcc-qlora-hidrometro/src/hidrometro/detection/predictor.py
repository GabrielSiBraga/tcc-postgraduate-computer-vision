"""Predictor Detectron2 congelado para detecção do visor (classe display)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hidrometro.config import load_yaml
from hidrometro.detection.config import build_detectron2_cfg, register_dataset_metadata
from hidrometro.preprocessing.io import imread_unicode


@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]
    score: float
    class_id: int
    class_name: str


class DisplayPredictor:
    """Wrapper fino sobre DefaultPredictor filtrando a classe display."""

    def __init__(self) -> None:
        setup_paths = load_yaml("paths.yaml")
        self.display_class_id = int(setup_paths["detectron2"]["display_class_id"])
        self.score_thresh = float(setup_paths["detectron2"]["score_thresh"])

        register_dataset_metadata()
        self.cfg = build_detectron2_cfg()

        from detectron2.engine import DefaultPredictor

        self.predictor = DefaultPredictor(self.cfg)
        self.class_names = load_yaml("detectron2.yaml")["model"]["thing_classes"]

    def predict(self, image: np.ndarray) -> DetectionResult | None:
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            return None

        mask = instances.pred_classes == self.display_class_id
        if not mask.any():
            return None

        filtered = instances[mask]
        scores = filtered.scores.numpy()
        best_idx = int(scores.argmax())
        if scores[best_idx] < self.score_thresh:
            return None

        box = filtered.pred_boxes[best_idx].tensor.numpy()[0]
        x1, y1, x2, y2 = map(int, box.tolist())
        class_id = int(filtered.pred_classes[best_idx])
        return DetectionResult(
            bbox=(x1, y1, x2, y2),
            score=float(scores[best_idx]),
            class_id=class_id,
            class_name=self.class_names[class_id],
        )

    def predict_path(self, image_path: str) -> tuple[np.ndarray, DetectionResult | None]:
        image = imread_unicode(image_path)
        if image is None:
            raise FileNotFoundError(f"Não foi possível ler a imagem: {image_path}")
        return image, self.predict(image)


def load_predictor() -> DisplayPredictor:
    return DisplayPredictor()
