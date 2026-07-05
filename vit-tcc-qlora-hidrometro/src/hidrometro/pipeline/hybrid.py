"""Pipeline híbrido Detectron2 → crop completo → CLAHE → Florence-2."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from hidrometro.autolabel.reading import LeituraStructured
from hidrometro.detection.predictor import DisplayPredictor, load_predictor
from hidrometro.preprocessing.clahe import apply_clahe
from hidrometro.preprocessing.crop import ExpansionRatios, crop_meter, draw_bboxes
from hidrometro.vlm.inference import VLMInference, sanitize_json


@dataclass
class PipelineResult:
    leitura: LeituraStructured
    fabricante: str
    estado: str
    latency_ms: float
    debug: dict[str, Any] = field(default_factory=dict)


class HybridPipeline:
    def __init__(
        self,
        predictor: DisplayPredictor | None = None,
        vlm: VLMInference | None = None,
        ratios: ExpansionRatios | None = None,
    ) -> None:
        self.predictor = predictor or load_predictor()
        self.vlm = vlm or VLMInference()
        self.ratios = ratios or ExpansionRatios.from_config()

    def run(self, image_bgr: np.ndarray) -> PipelineResult:
        start = time.perf_counter()

        detection = self.predictor.predict(image_bgr)
        if detection is None:
            raise ValueError("Nenhum visor (display) detectado na imagem.")

        crop, bbox_full, coverage = crop_meter(image_bgr, detection.bbox, self.ratios)
        crop_clahe = apply_clahe(crop)
        reading = self.vlm.predict(crop_clahe)
        payload = sanitize_json(reading)

        overlay = draw_bboxes(image_bgr, detection.bbox, bbox_full)
        _, crop_buffer = cv2.imencode(".jpg", crop_clahe)
        _, overlay_buffer = cv2.imencode(".jpg", overlay)

        leitura_dict = payload["leitura"]
        leitura = LeituraStructured(
            inteiro=int(leitura_dict["inteiro"]),
            decimal=int(leitura_dict["decimal"]),
            completo=str(leitura_dict["completo"]),
        )

        latency_ms = (time.perf_counter() - start) * 1000
        return PipelineResult(
            leitura=leitura,
            fabricante=payload["fabricante"],
            estado=payload["estado"],
            latency_ms=latency_ms,
            debug={
                "bbox_display": list(detection.bbox),
                "bbox_full_meter": list(bbox_full),
                "detection_score": detection.score,
                "coverage_ratio": coverage,
                "crop_base64": base64.b64encode(crop_buffer).decode("ascii"),
                "overlay_base64": base64.b64encode(overlay_buffer).decode("ascii"),
            },
        )

    def to_response_dict(self, result: PipelineResult) -> dict[str, Any]:
        return {
            "leitura": result.leitura.to_dict(),
            "fabricante": result.fabricante,
            "estado": result.estado,
            "latency_ms": round(result.latency_ms, 2),
            "debug": result.debug,
        }
