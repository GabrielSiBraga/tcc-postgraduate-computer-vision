"""Configuração e carregamento do Detectron2 congelado."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from hidrometro.config import load_yaml, project_root, resolve_path


def setup_detectron2_path() -> Path:
    paths = load_yaml("paths.yaml")
    repo = resolve_path(paths["detectron2"]["repo"])
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def build_detectron2_cfg():
    setup_detectron2_path()

    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    paths = load_yaml("paths.yaml")
    d2_cfg = load_yaml("detectron2.yaml")
    root = project_root()

    detectron2_root = resolve_path(paths["detectron2"]["repo"])
    weights = resolve_path(paths["detectron2"]["weights"])
    config_base = paths["detectron2"]["config_base"]
    device = paths["detectron2"].get("device", "cuda")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_base))
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = d2_cfg["model"]["num_classes"]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = d2_cfg["model"]["score_thresh_test"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = d2_cfg["model"]["batch_size_per_image"]
    cfg.DATALOADER.NUM_WORKERS = d2_cfg["training_reference"]["num_workers"]
    cfg.MODEL.DEVICE = device if _cuda_available() else "cpu"
    cfg.OUTPUT_DIR = str(root / "artifacts" / "detectron2_cache")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def register_dataset_metadata() -> None:
    setup_detectron2_path()
    from detectron2.data import MetadataCatalog

    d2_cfg = load_yaml("detectron2.yaml")
    MetadataCatalog.get("hidrometro_inference").set(
        thing_classes=d2_cfg["model"]["thing_classes"]
    )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
