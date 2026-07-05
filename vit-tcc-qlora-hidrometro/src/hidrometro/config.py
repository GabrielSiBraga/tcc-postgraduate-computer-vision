"""Utilitários compartilhados do projeto hidrômetro VLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict[str, Any]:
    path = project_root() / "configs" / name
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str | Path, base: Path | None = None) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    root = base or project_root()
    return (root / path).resolve()


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
