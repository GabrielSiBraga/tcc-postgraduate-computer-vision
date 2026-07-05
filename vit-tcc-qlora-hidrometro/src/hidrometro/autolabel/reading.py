"""Schema e parsing da leitura estruturada do hidrômetro."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LeituraStructured:
    inteiro: int
    decimal: int
    completo: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _to_int(value: Any, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"Campo '{field_name}' ausente ou vazio.")
    if isinstance(value, bool):
        raise ValueError(f"Campo '{field_name}' inválido.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"Campo '{field_name}' deve conter apenas dígitos: {value!r}")
    return int(text)


def parse_leitura(payload: dict[str, Any]) -> LeituraStructured:
    """Parse objeto aninhado leitura: { inteiro, decimal, completo }."""
    leitura_obj = payload.get("leitura")
    if not isinstance(leitura_obj, dict):
        raise ValueError(
            "Campo 'leitura' deve ser um objeto JSON com inteiro, decimal e completo."
        )

    inteiro = _to_int(leitura_obj.get("inteiro"), "inteiro")
    decimal = _to_int(leitura_obj.get("decimal"), "decimal")
    completo = str(leitura_obj.get("completo", "")).strip()
    if not completo:
        raise ValueError("Campo 'leitura.completo' ausente ou vazio.")

    return LeituraStructured(inteiro=inteiro, decimal=decimal, completo=completo)
