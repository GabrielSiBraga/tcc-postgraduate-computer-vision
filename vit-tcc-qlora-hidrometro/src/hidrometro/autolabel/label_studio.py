"""Conversão entre crops, auto-labels e formato SFT/Label Studio."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from hidrometro.autolabel.reading import LeituraStructured, parse_leitura
from hidrometro.config import ensure_dir, load_yaml, project_root, resolve_path


def iter_crop_files(crops_dir: Path) -> Iterator[Path]:
    for path in sorted(crops_dir.glob("*.jpg")):
        yield path


def load_meter_label(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    leitura = parse_leitura(payload)
    return {
        "leitura": leitura.to_dict(),
        "fabricante": str(payload.get("fabricante", "")).strip(),
        "estado": str(payload.get("estado", "normal")).strip(),
    }


def build_sft_record(image_path: Path, label: dict[str, Any]) -> dict[str, str]:
    cfg = load_yaml("autolabel.yaml")
    response = {
        "leitura": label["leitura"],
        "fabricante": label["fabricante"],
        "estado": label["estado"],
    }
    return {
        "image": str(image_path),
        "prompt": cfg["sft_prompt"],
        "response": json.dumps(response, ensure_ascii=False),
    }


def write_jsonl(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def local_files_image_url(crop_path: Path, document_root: Path) -> str:
    """URL para /data/local-files/ (path relativo ao DOCUMENT_ROOT, ex: data/)."""
    relative = crop_path.resolve().relative_to(document_root.resolve())
    return f"/data/local-files/?d={relative.as_posix()}"


def label_studio_task(image_path: Path, prelabel: dict[str, Any], image_url: str) -> dict:
    leitura = prelabel.get("leitura", {})
    if isinstance(leitura, dict):
        inteiro = int(leitura.get("inteiro", 0))
        decimal = int(leitura.get("decimal", 0))
        completo = str(leitura.get("completo", ""))
    else:
        parsed = parse_leitura({"leitura": leitura})
        inteiro, decimal, completo = parsed.inteiro, parsed.decimal, parsed.completo

    return {
        "data": {
            "image": image_url,
            # Label Studio Number tag exige strings em task.data
            "inteiro": str(inteiro),
            "decimal": str(decimal),
            "completo": completo,
            "fabricante": prelabel.get("fabricante", ""),
            "estado": prelabel.get("estado", "normal"),
        },
        "predictions": [
            {
                "result": [
                    {
                        "from_name": "inteiro",
                        "to_name": "image",
                        "type": "number",
                        "value": {"number": inteiro},
                    },
                    {
                        "from_name": "decimal",
                        "to_name": "image",
                        "type": "number",
                        "value": {"number": decimal},
                    },
                    {
                        "from_name": "completo",
                        "to_name": "image",
                        "type": "textarea",
                        "value": {"text": [completo]},
                    },
                    {
                        "from_name": "fabricante",
                        "to_name": "image",
                        "type": "textarea",
                        "value": {"text": [prelabel.get("fabricante", "")]},
                    },
                    {
                        "from_name": "estado",
                        "to_name": "image",
                        "type": "choices",
                        "value": {"choices": [prelabel.get("estado", "normal")]},
                    },
                ]
            }
        ],
    }


FIELDS = ("inteiro", "decimal", "completo", "fabricante", "estado")


def _last_list_value(items: list[Any] | None, default: str = "") -> str:
    """Retorna o último valor de um array Label Studio (correção mais recente)."""
    if not items:
        return default
    return str(items[-1]).strip()


def inteiro_from_completo(completo: str) -> int | None:
    """Extrai a parte inteira do visor (antes da vírgula)."""
    parts = completo.replace(".", ",").split(",")
    if not parts or not parts[0].strip():
        return None
    try:
        return int(parts[0].lstrip("0") or "0")
    except ValueError:
        return None


def inteiro_matches_completo(inteiro: int, completo: str) -> bool:
    """Verifica se inteiro (roletes pretos) bate com a parte inteira do completo literal."""
    parsed = inteiro_from_completo(completo)
    return parsed is not None and parsed == inteiro


def parse_list_field(raw: str) -> list[str]:
    """Parse coluna CSV/Label Studio com formato ['a', 'b']."""
    text = raw.strip()
    if not text:
        return []
    try:
        value = json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        import ast

        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _extract_annotation_values(result: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in result:
        name = item.get("from_name")
        value = item.get("value", {})
        if name == "inteiro":
            values["inteiro"] = value.get("number", 0)
        elif name == "decimal":
            values["decimal"] = value.get("number", 0)
        elif name == "completo":
            values["completo"] = value.get("text") or [""]
        elif name == "fabricante":
            values["fabricante"] = value.get("text") or [""]
        elif name == "estado":
            values["estado"] = value.get("choices") or ["normal"]
    return values


def audit_studio_export_row(row: dict[str, Any], split: str) -> dict[str, Any] | None:
    """Audita uma task exportada; retorna issue dict ou None se OK."""
    stem = task_stem_from_row(row)
    if not stem:
        return {"split": split, "stem": None, "task_id": row.get("id"), "issues": ["missing_stem"]}

    annotations = [a for a in (row.get("annotations") or row.get("completions") or []) if not a.get("was_cancelled")]
    issues: list[str] = []
    details: dict[str, Any] = {}

    if not annotations:
        return {"split": split, "stem": stem, "task_id": row.get("id"), "issues": ["no_annotation"]}
    if len(annotations) > 1:
        issues.append("multiple_annotations")

    result = annotations[0].get("result") or []
    name_counts: dict[str, int] = {}
    for item in result:
        name = item.get("from_name")
        if name:
            name_counts[name] = name_counts.get(name, 0) + 1
    for name, count in name_counts.items():
        if count > 1:
            issues.append(f"duplicate_from_name:{name}")

    raw = _extract_annotation_values(result)
    missing = [field for field in FIELDS if field not in raw]
    if missing:
        issues.append(f"missing_fields:{','.join(missing)}")

    for field in ("completo", "fabricante", "estado"):
        arr = raw.get(field)
        if not isinstance(arr, list):
            continue
        if len(arr) == 0:
            issues.append(f"array_empty:{field}")
        elif len(arr) > 1:
            unique = list(dict.fromkeys(arr))
            details[f"{field}_values"] = arr
            if len(unique) > 1:
                issues.append(f"array_multi_valor:{field}")
            else:
                issues.append(f"array_duplicado:{field}")

    inteiro = int(raw.get("inteiro", 0))
    completo_arr = raw.get("completo") or [""]
    completo = _last_list_value(completo_arr)
    if not completo:
        issues.append("completo_vazio")
    elif inteiro_from_completo(completo) is None:
        issues.append("completo_unparseable")
    elif not inteiro_matches_completo(inteiro, completo):
        issues.append("inconsistent_leitura")
        details["inteiro"] = inteiro
        details["decimal"] = int(raw.get("decimal", 0))
        details["completo"] = completo
        details["inteiro_from_completo"] = inteiro_from_completo(completo)

    if not issues:
        return None
    return {"split": split, "stem": stem, "task_id": row.get("id"), "issues": issues, **details}


def audit_studio_export(export_path: Path, split: str) -> dict[str, Any]:
    rows = json.loads(export_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("tasks") or rows.get("data") or [rows]

    issues: list[dict[str, Any]] = []
    for row in rows:
        issue = audit_studio_export_row(row, split)
        if issue:
            issues.append(issue)

    total = len(rows)
    multi = [i for i in issues if any(x.startswith("array_multi_valor:") for x in i["issues"])]
    inconsistent = [i for i in issues if "inconsistent_leitura" in i["issues"]]

    return {
        "split": split,
        "source": str(export_path),
        "total_tasks": total,
        "tasks_with_issues": len(issues),
        "array_multi_valor": len(multi),
        "inconsistent_leitura": len(inconsistent),
        "issues": issues,
    }


def task_stem_from_row(row: dict[str, Any]) -> str | None:
    """Extrai stem do crop (ex: 000042) a partir do campo image da task."""
    image = str(row.get("data", {}).get("image", ""))
    if "?d=" in image:
        image = image.split("?d=", 1)[1]
    if image.startswith("file://"):
        return Path(image.replace("file:///", "").replace("file://", "")).stem
    if image:
        return Path(image).stem
    return None


def convert_studio_export(export_path: Path, split: str, validated_dir: Path | None = None) -> int:
    """Converte export JSON do Label Studio em labels validados por crop."""
    paths = load_yaml("paths.yaml")
    out_root = validated_dir or resolve_path(paths["output"]["autolabel_validated"]) / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = json.loads(export_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("tasks") or rows.get("data") or [rows]

    written = 0
    skipped = 0
    for row in rows:
        annotations = row.get("annotations") or row.get("completions") or []
        annotations = [a for a in annotations if not a.get("was_cancelled")]
        if not annotations or not annotations[0].get("result"):
            skipped += 1
            continue

        stem = task_stem_from_row(row)
        if not stem:
            skipped += 1
            continue

        payload = label_from_studio_export(row)
        out_path = out_root / f"{stem}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    print(f"Convertido {export_path.name}: {written} labels -> {out_root} (pulados: {skipped})")
    return written


def label_from_studio_export(row: dict[str, Any]) -> dict[str, Any]:
    annotations = row.get("annotations") or row.get("completions") or []
    if not annotations:
        data = row.get("data", row)
        return {
            "leitura": LeituraStructured(
                inteiro=int(data.get("inteiro", 0)),
                decimal=int(data.get("decimal", 0)),
                completo=str(data.get("completo", "")),
            ).to_dict(),
            "fabricante": str(data.get("fabricante", "")),
            "estado": str(data.get("estado", "normal")),
        }

    raw = _extract_annotation_values(annotations[0].get("result", []))
    inteiro = int(raw.get("inteiro", 0))
    decimal = int(raw.get("decimal", 0))
    completo = _last_list_value(raw.get("completo"))
    fabricante = _last_list_value(raw.get("fabricante"))
    estado = _last_list_value(raw.get("estado"), default="normal")

    leitura = LeituraStructured(
        inteiro=inteiro,
        decimal=decimal,
        completo=completo,
    )
    return {
        "leitura": leitura.to_dict(),
        "fabricante": fabricante,
        "estado": estado,
    }


def apply_manual_review_csv(csv_path: Path, validated_root: Path | None = None) -> dict[str, int]:
    """Aplica correções do CSV de revisão manual nos labels validados."""
    paths = load_yaml("paths.yaml")
    root = validated_root or resolve_path(paths["output"]["autolabel_validated"])

    updated = 0
    skipped = 0
    for row in csv.DictReader(csv_path.open(encoding="utf-8")):
        split = row["split"].strip()
        stem = row["stem"].strip()
        label_path = root / split / f"{stem}.json"
        if not label_path.exists():
            skipped += 1
            continue

        payload = json.loads(label_path.read_text(encoding="utf-8"))
        changed = False

        inteiro = row.get("inteiro", "").strip()
        decimal = row.get("decimal", "").strip()
        completo = row.get("completo", "").strip()
        if inteiro and decimal and completo:
            payload["leitura"] = {
                "inteiro": int(inteiro),
                "decimal": int(decimal),
                "completo": completo,
            }
            changed = True

        fabricante_values = parse_list_field(row.get("fabricante_values", ""))
        if fabricante_values:
            fabricante = fabricante_values[-1]
            if payload.get("fabricante") != fabricante:
                payload["fabricante"] = fabricante
                changed = True

        if changed:
            label_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            updated += 1
        else:
            skipped += 1

    return {"updated": updated, "skipped": skipped}


def annotation_result_from_label(label: dict[str, Any]) -> list[dict[str, Any]]:
    """Resultado Label Studio com um único valor por campo (estado final validado)."""
    leitura = label["leitura"]
    inteiro = int(leitura["inteiro"])
    decimal = int(leitura["decimal"])
    completo = str(leitura["completo"])
    fabricante = str(label.get("fabricante", ""))
    estado = str(label.get("estado", "normal"))
    return [
        {
            "from_name": "inteiro",
            "to_name": "image",
            "type": "number",
            "value": {"number": inteiro},
        },
        {
            "from_name": "decimal",
            "to_name": "image",
            "type": "number",
            "value": {"number": decimal},
        },
        {
            "from_name": "completo",
            "to_name": "image",
            "type": "textarea",
            "value": {"text": [completo]},
        },
        {
            "from_name": "fabricante",
            "to_name": "image",
            "type": "textarea",
            "value": {"text": [fabricante]},
        },
        {
            "from_name": "estado",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": [estado]},
        },
    ]


def task_data_from_label(label: dict[str, Any], image_url: str) -> dict[str, str]:
    leitura = label["leitura"]
    return {
        "image": image_url,
        "inteiro": str(leitura["inteiro"]),
        "decimal": str(leitura["decimal"]),
        "completo": str(leitura["completo"]),
        "fabricante": str(label.get("fabricante", "")),
        "estado": str(label.get("estado", "normal")),
    }


def export_row_from_validated(task_id: int, image_url: str, label: dict[str, Any]) -> dict[str, Any]:
    """Task no formato de export Label Studio a partir do label validado."""
    return {
        "id": task_id,
        "annotations": [
            {
                "id": task_id,
                "result": annotation_result_from_label(label),
                "was_cancelled": False,
            }
        ],
        "data": task_data_from_label(label, image_url),
    }


def label_from_export_data(data: dict[str, Any]) -> dict[str, Any]:
    """Monta label raw a partir do campo data de um export Label Studio (prelabel original)."""
    return {
        "leitura": LeituraStructured(
            inteiro=int(data.get("inteiro", 0)),
            decimal=int(data.get("decimal", 0)),
            completo=str(data.get("completo", "")),
        ).to_dict(),
        "fabricante": str(data.get("fabricante", "")),
        "estado": str(data.get("estado", "normal")),
    }


def extract_raw_from_studio_export(export_path: Path, split: str) -> int:
    """Extrai prelabels do campo data de um export LS original → raw_json/{split}/."""
    paths = load_yaml("paths.yaml")
    out_root = ensure_dir(resolve_path(paths["output"]["autolabel_raw"]) / split)

    rows = json.loads(export_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("tasks") or rows.get("data") or [rows]

    written = 0
    for row in rows:
        stem = task_stem_from_row(row)
        if not stem:
            continue
        payload = label_from_export_data(row.get("data", {}))
        out_path = out_root / f"{stem}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written += 1

    print(f"raw_json extraído de {export_path.name}: {written} labels -> {out_root}")
    return written


def regenerate_label_studio_from_validated(split: str) -> tuple[int, Path, Path]:
    """Regenera import/ (raw/prelabel) e export/ (validado final)."""
    paths = load_yaml("paths.yaml")
    crops_base = resolve_path(paths["output"]["crops"])
    crops_root = crops_base / split
    document_root = crops_base.parent
    raw_root = resolve_path(paths["output"]["autolabel_raw"]) / split
    validated_root = resolve_path(paths["output"]["autolabel_validated"]) / split
    import_dir = ensure_dir(project_root() / "data" / "label_studio" / "import")
    export_dir = ensure_dir(project_root() / "data" / "label_studio" / "export")

    import_tasks: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    task_id = 1

    for crop_path in sorted(crops_root.glob("*.jpg")):
        validated_path = validated_root / f"{crop_path.stem}.json"
        if not validated_path.exists():
            continue

        validated_label = json.loads(validated_path.read_text(encoding="utf-8"))
        image_url = local_files_image_url(crop_path, document_root)

        raw_path = raw_root / f"{crop_path.stem}.json"
        prelabel_path = raw_path if raw_path.exists() else validated_path
        prelabel = json.loads(prelabel_path.read_text(encoding="utf-8"))
        import_tasks.append(label_studio_task(crop_path, prelabel, image_url))
        export_rows.append(export_row_from_validated(task_id, image_url, validated_label))
        task_id += 1

    import_path = import_dir / f"{split}_tasks.json"
    export_path = export_dir / f"{split}.json"
    import_path.write_text(json.dumps(import_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    export_path.write_text(json.dumps(export_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(import_tasks), import_path, export_path
