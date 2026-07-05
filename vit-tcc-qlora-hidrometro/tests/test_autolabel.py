"""Testes de leitura estruturada inteiro/decimal/completo."""

import pytest

from hidrometro.autolabel.openai_labeler import parse_meter_response
from hidrometro.autolabel.reading import LeituraStructured, parse_leitura


def test_parse_leitura_nested():
    payload = {
        "leitura": {"inteiro": 302, "decimal": 21, "completo": "0302,21"},
        "fabricante": "Itron",
        "estado": "normal",
    }
    leitura = parse_leitura(payload)
    assert leitura.inteiro == 302
    assert leitura.decimal == 21
    assert leitura.completo == "0302,21"


def test_parse_leitura_rejects_flat_string():
    with pytest.raises(ValueError, match="objeto JSON"):
        parse_leitura({"leitura": "001370,45"})


def test_parse_meter_response_nested_json():
    text = (
        '{"leitura": {"inteiro": 1370, "decimal": 45, "completo": "001370,45"}, '
        '"fabricante": "Itron", "estado": "normal"}'
    )
    reading = parse_meter_response(text)
    assert reading.leitura == LeituraStructured(1370, 45, "001370,45")
    assert reading.fabricante == "Itron"


def test_parse_meter_response_markdown_fence():
    text = """```json
{
  "leitura": {"inteiro": 99, "decimal": 5, "completo": "0099,05"},
  "fabricante": "Baylan",
  "estado": "embacado"
}
```"""
    reading = parse_meter_response(text)
    assert reading.leitura.completo == "0099,05"
    assert reading.leitura.decimal == 5
