#!/usr/bin/env python3
"""Import/export Label Studio e conversão para SFT jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.autolabel.label_studio import (
    build_sft_record,
    convert_studio_export,
    extract_raw_from_studio_export,
    label_studio_task,
    local_files_image_url,
    regenerate_label_studio_from_validated,
    write_jsonl,
)
from hidrometro.config import ensure_dir, load_yaml, project_root, resolve_path


def import_tasks(split: str) -> None:
    paths = load_yaml("paths.yaml")
    crops_base = resolve_path(paths["output"]["crops"])
    crops_root = crops_base / split
    # DOCUMENT_ROOT deve ser o pai de crops/ (ex: data/) — storage aponta para crops/
    document_root = crops_base.parent
    raw_root = resolve_path(paths["output"]["autolabel_raw"]) / split
    validated_root = resolve_path(paths["output"]["autolabel_validated"]) / split
    out_dir = ensure_dir(project_root() / "data" / "label_studio" / "import")
    tasks = []

    for crop_path in sorted(crops_root.glob("*.jpg")):
        label_path = raw_root / f"{crop_path.stem}.json"
        if not label_path.exists():
            label_path = validated_root / f"{crop_path.stem}.json"
        if not label_path.exists():
            continue
        prelabel = json.loads(label_path.read_text(encoding="utf-8"))
        image_url = local_files_image_url(crop_path, document_root)
        tasks.append(label_studio_task(crop_path, prelabel, image_url))

    out_file = out_dir / f"{split}_tasks.json"
    out_file.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Import Label Studio: {out_file} ({len(tasks)} tasks)")


def export_sft(split: str, validated_dir: Path | None = None) -> None:
    paths = load_yaml("paths.yaml")
    crops_root = resolve_path(paths["output"]["crops"]) / split
    validated_root = validated_dir or resolve_path(paths["output"]["autolabel_validated"]) / split
    sft_root = ensure_dir(resolve_path(paths["output"]["sft"]))

    records = []
    source = validated_root if validated_root.exists() else resolve_path(paths["output"]["autolabel_raw"]) / split

    for label_path in sorted(source.glob("*.json")):
        crop_path = crops_root / f"{label_path.stem}.jpg"
        if not crop_path.exists():
            continue
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        records.append(build_sft_record(crop_path, payload))

    out_path = sft_root / f"{split}.jsonl"
    write_jsonl(records, out_path)
    print(f"SFT exportado: {out_path} ({len(records)} amostras)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Label Studio import/export")
    parser.add_argument(
        "--mode",
        choices=["import", "convert", "export", "sync", "extract-raw"],
        required=True,
        help=(
            "import: tasks JSON a partir de raw_json; convert: LS export → validated; "
            "export: validated → SFT; sync: raw→import + validated→export; "
            "extract-raw: export LS original (campo data) → raw_json"
        ),
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument(
        "--export-file",
        type=Path,
        help="JSON exportado pelo Label Studio (convert ou extract-raw)",
    )
    parser.add_argument("--validated-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "convert":
        if args.split == "all":
            raise SystemExit("Use --split train|val|test com --mode convert (um export por projeto).")
        if not args.export_file or not args.export_file.exists():
            raise SystemExit("Informe --export-file com o JSON exportado do Label Studio.")
        convert_studio_export(args.export_file, args.split, args.validated_dir)
        return

    if args.mode == "extract-raw":
        if args.split == "all":
            raise SystemExit("Use --split train|val|test com --mode extract-raw.")
        if not args.export_file or not args.export_file.exists():
            raise SystemExit("Informe --export-file com o export LS original (campo data = prelabels).")
        extract_raw_from_studio_export(args.export_file, args.split)
        return

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        if args.mode == "import":
            import_tasks(split)
        elif args.mode == "sync":
            count, import_path, export_path = regenerate_label_studio_from_validated(split)
            print(f"Sync Label Studio [{split}]: {count} tasks")
            print(f"  import: {import_path}")
            print(f"  export: {export_path}")
        else:
            export_sft(split, args.validated_dir)


if __name__ == "__main__":
    main()
