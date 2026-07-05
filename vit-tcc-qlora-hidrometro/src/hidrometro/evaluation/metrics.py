"""Métricas unificadas de avaliação do pipeline VLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hidrometro.autolabel.reading import LeituraStructured
from hidrometro.evaluation.cer import batch_cer, character_error_rate
from hidrometro.evaluation.f1_macro import classification_summary, macro_f1
from hidrometro.preprocessing.io import imread_unicode
from hidrometro.vlm.inference import VLMInference


@dataclass
class EvalSample:
    stem: str
    ref_completo: str
    hyp_completo: str
    ref_decimal: str
    hyp_decimal: str
    ref_fabricante: str
    hyp_fabricante: str
    ref_estado: str
    hyp_estado: str
    cer: float
    exact_match: bool
    parse_ok: bool
    raw_response: str = ""


def run_vlm_evaluation(
    vlm: VLMInference,
    crop_paths: list[Path],
    ground_truth: dict[str, dict],
) -> list[EvalSample]:
    """Executa inferência e compara predições com ground truth."""
    vlm.load()
    rows: list[EvalSample] = []
    for crop_path in crop_paths:
        label = ground_truth.get(crop_path.stem)
        if not label:
            continue
        image = imread_unicode(crop_path)
        if image is None:
            continue

        parse_ok = True
        raw_response = ""
        try:
            pred = vlm.predict(image)
        except ValueError as exc:
            parse_ok = False
            raw_response = str(exc)
            pred = type(
                "Pred",
                (),
                {
                    "leitura": LeituraStructured(inteiro=0, decimal=0, completo=""),
                    "fabricante": "",
                    "estado": "",
                },
            )()

        ref_completo = str(label["leitura"]["completo"])
        hyp_completo = str(pred.leitura.completo)
        rows.append(
            EvalSample(
                stem=crop_path.stem,
                ref_completo=ref_completo,
                hyp_completo=hyp_completo,
                ref_decimal=str(label["leitura"]["decimal"]),
                hyp_decimal=str(pred.leitura.decimal),
                ref_fabricante=str(label.get("fabricante", "")),
                hyp_fabricante=str(pred.fabricante),
                ref_estado=str(label.get("estado", "")),
                hyp_estado=str(pred.estado),
                cer=character_error_rate(ref_completo, hyp_completo),
                exact_match=ref_completo.strip() == hyp_completo.strip(),
                parse_ok=parse_ok,
                raw_response=raw_response,
            )
        )
    return rows


def compute_metrics(rows: list[EvalSample]) -> dict[str, Any]:
    """Calcula métricas brutas (0–1) e percentuais para relatório."""
    if not rows:
        return {}

    refs_c = [r.ref_completo for r in rows]
    hyps_c = [r.hyp_completo for r in rows]
    refs_d = [r.ref_decimal for r in rows]
    hyps_d = [r.hyp_decimal for r in rows]
    refs_f = [r.ref_fabricante for r in rows]
    hyps_f = [r.hyp_fabricante for r in rows]
    refs_e = [r.ref_estado for r in rows]
    hyps_e = [r.hyp_estado for r in rows]

    cer_completo = batch_cer(list(zip(refs_c, hyps_c)))
    cer_decimal = batch_cer(list(zip(refs_d, hyps_d)))
    exact_match_rate = sum(r.exact_match for r in rows) / len(rows)
    parse_success_rate = sum(r.parse_ok for r in rows) / len(rows)

    fab_report = classification_summary(refs_f, hyps_f)
    est_report = classification_summary(refs_e, hyps_e)

    return {
        "samples": len(rows),
        "parse_success_rate": parse_success_rate,
        "cer_completo": cer_completo,
        "cer_decimal": cer_decimal,
        "exact_match_rate": exact_match_rate,
        "reading_char_accuracy": 1.0 - cer_completo,
        "decimal_char_accuracy": 1.0 - cer_decimal,
        "f1_fabricante_macro": macro_f1(refs_f, hyps_f),
        "f1_estado_macro": macro_f1(refs_e, hyps_e),
        "accuracy_fabricante": float(fab_report.get("accuracy", 0.0)),
        "accuracy_estado": float(est_report.get("accuracy", 0.0)),
        "precision_fabricante_macro": float(fab_report["macro avg"]["precision"]),
        "recall_fabricante_macro": float(fab_report["macro avg"]["recall"]),
        "precision_estado_macro": float(est_report["macro avg"]["precision"]),
        "recall_estado_macro": float(est_report["macro avg"]["recall"]),
        "fabricante_report": fab_report,
        "estado_report": est_report,
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def metrics_summary_table(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Tabela legível para notebook e relatório TCC (valores em %)."""
    if not metrics:
        return []

    return [
        {
            "métrica": "Amostras avaliadas",
            "valor": str(metrics["samples"]),
            "descrição": "Total de crops com ground truth",
        },
        {
            "métrica": "Taxa de JSON válido (parse)",
            "valor": format_percent(metrics["parse_success_rate"]),
            "descrição": "Predições que geraram JSON estruturado",
        },
        {
            "métrica": "Acurácia exata da leitura (exact match)",
            "valor": format_percent(metrics["exact_match_rate"]),
            "descrição": "Leitura.completo idêntica ao ground truth",
        },
        {
            "métrica": "Acurácia por caractere (1 - CER)",
            "valor": format_percent(metrics["reading_char_accuracy"]),
            "descrição": "Proximidade caractere a caractere do visor",
        },
        {
            "métrica": "CER completo",
            "valor": format_percent(metrics["cer_completo"]),
            "descrição": "Character Error Rate (menor = melhor)",
        },
        {
            "métrica": "Acurácia fabricante",
            "valor": format_percent(metrics["accuracy_fabricante"]),
            "descrição": "Classificação correta da marca/placa",
        },
        {
            "métrica": "Precisão fabricante (macro)",
            "valor": format_percent(metrics["precision_fabricante_macro"]),
            "descrição": "Média macro de precisão por classe",
        },
        {
            "métrica": "Acurácia estado",
            "valor": format_percent(metrics["accuracy_estado"]),
            "descrição": "Classificação correta do estado do visor",
        },
        {
            "métrica": "Precisão estado (macro)",
            "valor": format_percent(metrics["precision_estado_macro"]),
            "descrição": "Média macro de precisão por classe",
        },
        {
            "métrica": "F1 fabricante (macro)",
            "valor": format_percent(metrics["f1_fabricante_macro"]),
            "descrição": "Balanceamento precisão/recall por fabricante",
        },
        {
            "métrica": "F1 estado (macro)",
            "valor": format_percent(metrics["f1_estado_macro"]),
            "descrição": "Balanceamento precisão/recall por estado",
        },
    ]


def samples_to_dicts(rows: list[EvalSample]) -> list[dict[str, Any]]:
    return [asdict(r) for r in rows]
