"""Testes de conversão e auditoria de exports Label Studio."""

from hidrometro.autolabel.label_studio import (
    audit_studio_export_row,
    inteiro_from_completo,
    inteiro_matches_completo,
    label_from_studio_export,
)


def _annotation_row(completo_texts, fabricante_texts=None, inteiro=1, decimal=82):
    fabricante_texts = fabricante_texts or ["ENERGYRUS"]
    return {
        "id": 648,
        "data": {"image": "/data/local-files/?d=crops/test/000000.jpg"},
        "annotations": [
            {
                "was_cancelled": False,
                "result": [
                    {"from_name": "inteiro", "value": {"number": inteiro}},
                    {"from_name": "decimal", "value": {"number": decimal}},
                    {"from_name": "completo", "value": {"text": completo_texts}},
                    {"from_name": "fabricante", "value": {"text": fabricante_texts}},
                    {"from_name": "estado", "value": {"choices": ["normal"]}},
                ],
            }
        ],
    }


def test_label_from_studio_export_uses_last_text_value():
    row = _annotation_row(["0118,72", "0001,82"])
    payload = label_from_studio_export(row)
    assert payload["leitura"]["inteiro"] == 1
    assert payload["leitura"]["decimal"] == 82
    assert payload["leitura"]["completo"] == "0001,82"


def test_label_from_studio_export_uses_last_fabricante():
    row = _annotation_row(["0001,82"], ["ENERGIE", "ENERGYRUS"])
    payload = label_from_studio_export(row)
    assert payload["fabricante"] == "ENERGYRUS"


def test_label_from_studio_export_keeps_literal_completo_with_decimal_zero():
    row = _annotation_row(["0038,08"], inteiro=38, decimal=0)
    payload = label_from_studio_export(row)
    assert payload["leitura"]["completo"] == "0038,08"
    assert payload["leitura"]["decimal"] == 0


def test_audit_detects_array_multi_valor():
    row = _annotation_row(["0118,72", "0001,82"], ["Sensus", "SAGA"])
    issue = audit_studio_export_row(row, "test")
    assert issue is not None
    assert "array_multi_valor:completo" in issue["issues"]
    assert "array_multi_valor:fabricante" in issue["issues"]


def test_audit_allows_decimal_zero_with_suffix_in_completo():
    row = _annotation_row(["0038,08"], inteiro=38, decimal=0)
    issue = audit_studio_export_row(row, "train")
    assert issue is None


def test_inteiro_from_completo():
    assert inteiro_from_completo("0001,82") == 1
    assert inteiro_from_completo("0038,08") == 38
    assert inteiro_matches_completo(38, "0038,08")
