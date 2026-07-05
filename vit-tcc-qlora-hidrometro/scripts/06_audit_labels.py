#!/usr/bin/env python3
"""Audita inconsistências entre inteiro e completo nos labels SFT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.config import ensure_dir, load_yaml, resolve_path


def check_record(row: dict) -> tuple[bool, str]:
    resp = json.loads(row["response"])
    leitura = resp["leitura"]
    inteiro = leitura["inteiro"]
    completo = leitura["completo"]
    parte_inteira = completo.split(",")[0].lstrip("0") or "0"
    ok = str(inteiro) == parte_inteira
    return ok, f"inteiro={inteiro} vs completo={completo}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar labels SFT")
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    args = parser.parse_args()

    sft_dir = resolve_path(load_yaml("paths.yaml")["output"]["sft"])
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    issues: list[dict] = []
    total = 0
    for split in splits:
        path = sft_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            ok, detail = check_record(row)
            if not ok:
                issues.append({"split": split, "image": row["image"], "detail": detail})

    report = {
        "total_records": total,
        "inconsistent": len(issues),
        "consistency_rate": (total - len(issues)) / total if total else 0.0,
        "issues": issues,
    }

    out = ensure_dir(resolve_path("reports")) / "label_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    pct = report["consistency_rate"] * 100
    print(f"Registros: {total} | inconsistentes: {len(issues)} | consistência: {pct:.1f}%")
    print(f"Relatório: {out}")
    for item in issues[:10]:
        print(f"  [{item['split']}] {Path(item['image']).name}: {item['detail']}")


if __name__ == "__main__":
    main()
