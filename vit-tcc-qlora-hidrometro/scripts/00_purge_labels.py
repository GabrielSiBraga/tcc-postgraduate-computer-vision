#!/usr/bin/env python3
"""Remove labels legados antes da regeneração com schema estruturado."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.config import ensure_dir, load_yaml, project_root, resolve_path

PURGE_PATHS = [
    "data/autolabel/raw_json",
    "data/autolabel/validated",
    "data/sft",
    "data/label_studio/import",
    "reports/autolabel_failures.csv",
]


def collect_targets() -> list[Path]:
    targets: list[Path] = []
    for relative in PURGE_PATHS:
        path = resolve_path(relative)
        if path.exists():
            targets.append(path)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Purge labels legados")
    parser.add_argument("--dry-run", action="store_true", help="Lista arquivos sem apagar")
    parser.add_argument("--confirm", action="store_true", help="Executa remoção")
    args = parser.parse_args()

    if not args.dry_run and not args.confirm:
        raise SystemExit("Use --dry-run para preview ou --confirm para executar.")

    targets = collect_targets()
    if not targets:
        print("Nenhum artefato legado encontrado.")
        return

    removed_files = 0
    removed_dirs = 0
    for target in targets:
        if target.is_file():
            count = 1
            print(f"{'[dry-run] ' if args.dry_run else ''}remove file: {target}")
            if args.confirm:
                target.unlink(missing_ok=True)
            removed_files += count
        elif target.is_dir():
            files = list(target.rglob("*"))
            file_count = sum(1 for p in files if p.is_file())
            print(
                f"{'[dry-run] ' if args.dry_run else ''}remove dir: {target} "
                f"({file_count} arquivos)"
            )
            if args.confirm:
                shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
            removed_files += file_count
            removed_dirs += 1

    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "targets": [str(p) for p in targets],
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "dry_run": args.dry_run,
    }
    reports_dir = ensure_dir(resolve_path("reports"))
    manifest_path = reports_dir / "purge_manifest.json"
    if args.confirm:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Manifest salvo em: {manifest_path}")
    print(f"Total arquivos afetados: {removed_files}")


if __name__ == "__main__":
    main()
