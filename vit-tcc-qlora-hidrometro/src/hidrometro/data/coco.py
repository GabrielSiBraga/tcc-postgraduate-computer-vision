"""Utilitários para percorrer o dataset COCO do mask-rcnn."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from hidrometro.config import load_yaml, project_root, resolve_path


@dataclass
class CocoSample:
    image_id: int
    file_name: str
    image_path: Path
    split: str


def split_dirs() -> dict[str, str]:
    paths = load_yaml("paths.yaml")
    return paths["dataset"]["splits"]


def dataset_root() -> Path:
    paths = load_yaml("paths.yaml")
    return resolve_path(paths["dataset"]["root"])


def iter_coco_samples(split_key: str = "train") -> Iterator[CocoSample]:
    paths = load_yaml("paths.yaml")
    split_name = paths["dataset"]["splits"][split_key]
    root = dataset_root()
    ann_path = root / split_name / paths["dataset"]["annotation_file"]
    images_dir = root / split_name

    with ann_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    for image in payload["images"]:
        image_path = images_dir / image["file_name"]
        yield CocoSample(
            image_id=int(image["id"]),
            file_name=image["file_name"],
            image_path=image_path,
            split=split_key,
        )


def list_all_splits() -> list[str]:
    return list(split_dirs().keys())
