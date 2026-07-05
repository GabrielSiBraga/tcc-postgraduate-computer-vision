#!/usr/bin/env python3
"""Audita divergências nos exports JSON do Label Studio."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hidrometro.autolabel.label_studio import audit_studio_export
from hidrometro.config import ensure_dir, project_root, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar exports Label Studio")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "val", "test"],
        help="Split a auditar (default: all)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=project_root() / "data" / "label_studio" / "export",
        help="Diretório com train.json, val.json, test.json",
    )
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    reports: list[dict] = []
    manual_rows: list[dict] = []

    for split in splits:
        export_path = args.export_dir / f"{split}.json"
        if not export_path.exists():
            print(f"Aviso: {export_path} não encontrado, pulando.")
            continue
        report = audit_studio_export(export_path, split)
        reports.append(report)
        for issue in report["issues"]:
            if any(
                tag.startswith("array_multi_valor:")
                or tag == "inconsistent_leitura"
                for tag in issue["issues"]
            ):
                manual_rows.append(
                    {
                        "split": issue["split"],
                        "stem": issue.get("stem", ""),
                        "task_id": issue.get("task_id", ""),
                        "issues": ";".join(issue["issues"]),
                        "completo": issue.get("completo", ""),
                        "inteiro": issue.get("inteiro", ""),
                        "decimal": issue.get("decimal", ""),
                        "completo_values": issue.get("completo_values", ""),
                        "fabricante_values": issue.get("fabricante_values", ""),
                    }
                )

    total_tasks = sum(r["total_tasks"] for r in reports)
    total_issues = sum(r["tasks_with_issues"] for r in reports)
    total_multi = sum(r["array_multi_valor"] for r in reports)
    total_inconsistent = sum(r["inconsistent_leitura"] for r in reports)

    summary = {
        "total_tasks": total_tasks,
        "tasks_with_issues": total_issues,
        "array_multi_valor": total_multi,
        "inconsistent_leitura": total_inconsistent,
        "splits": reports,
    }

    reports_dir = ensure_dir(resolve_path("reports"))
    json_path = reports_dir / "label_studio_export_audit.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = reports_dir / "label_studio_export_issues.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "stem",
                "task_id",
                "issues",
                "completo",
                "inteiro",
                "decimal",
                "completo_values",
                "fabricante_values",
            ],
        )
        writer.writeheader()
        writer.writerows(manual_rows)

    print(f"Tasks: {total_tasks} | com issues: {total_issues}")
    print(f"array_multi_valor: {total_multi} | inconsistent_leitura: {total_inconsistent}")
    print(f"Relatório JSON: {json_path}")
    print(f"Issues do export (CSV): {csv_path} ({len(manual_rows)} tasks)")

    for report in reports:
        print(
            f"  [{report['split']}] {report['total_tasks']} tasks, "
            f"{report['array_multi_valor']} multi-valor, "
            f"{report['inconsistent_leitura']} inconsistentes"
        )


if __name__ == "__main__":
    main()
