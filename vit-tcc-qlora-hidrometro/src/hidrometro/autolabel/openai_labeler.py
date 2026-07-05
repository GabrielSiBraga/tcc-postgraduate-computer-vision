"""Auto-labelling zero-shot via OpenAI GPT-4o."""

from __future__ import annotations

import ast
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hidrometro.autolabel.reading import LeituraStructured, parse_leitura
from hidrometro.config import load_yaml


@dataclass
class MeterReading:
    leitura: LeituraStructured
    fabricante: str
    estado: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "leitura": self.leitura.to_dict(),
            "fabricante": self.fabricante,
            "estado": self.estado,
        }


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _normalize_json_like(text: str) -> str:
    text = _strip_markdown_fences(text)
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _parse_with_literal_eval(text: str) -> dict[str, Any]:
    payload = ast.literal_eval(text)
    if not isinstance(payload, dict):
        raise ValueError("Resposta não é um objeto JSON/dict.")
    return payload


def _parse_with_regex(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    nested = re.search(
        r'"leitura"\s*:\s*\{([^}]+)\}',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if nested:
        block = nested.group(1)
        inteiro = re.search(r'"inteiro"\s*:\s*(\d+)', block)
        decimal = re.search(r'"decimal"\s*:\s*(\d+)', block)
        completo = re.search(r'"completo"\s*:\s*"([^"]*)"', block)
        if inteiro and decimal and completo:
            payload["leitura"] = {
                "inteiro": int(inteiro.group(1)),
                "decimal": int(decimal.group(1)),
                "completo": completo.group(1),
            }

    for key in ("fabricante", "estado"):
        match = re.search(
            rf'"{key}"\s*:\s*"([^"]*)"|\'{key}\'\s*:\s*\'([^\']*)\'',
            text,
            flags=re.IGNORECASE,
        )
        if match:
            payload[key] = next(group for group in match.groups() if group is not None)

    if "leitura" not in payload:
        raise ValueError(f"Objeto leitura não encontrado na resposta: {text[:200]}")
    return payload


def _extract_json(text: str) -> dict[str, Any]:
    candidates: list[str] = []
    cleaned = _normalize_json_like(text)
    candidates.append(cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(_normalize_json_like(match.group(0)))

    last_error: Exception | None = None
    for candidate in candidates:
        for parser in (json.loads, _parse_with_literal_eval):
            try:
                payload = parser(candidate)
                if isinstance(payload, dict):
                    return payload
            except Exception as exc:  # noqa: BLE001
                last_error = exc

    try:
        return _parse_with_regex(text)
    except Exception as exc:  # noqa: BLE001
        last_error = exc

    raise ValueError(f"JSON inválido na resposta: {text[:300]}") from last_error


def parse_meter_response(text: str) -> MeterReading:
    payload = _extract_json(text)
    leitura = parse_leitura(payload)
    return MeterReading(
        leitura=leitura,
        fabricante=str(payload.get("fabricante", "")).strip(),
        estado=str(payload.get("estado", "")).strip(),
    )


class OpenAILabeler:
    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        self.cfg = load_yaml("autolabel.yaml")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", self.cfg["openai"]["model"])
        self.prompt = self.cfg["prompt"].strip()
        self.max_retries = int(self.cfg["openai"]["max_retries"])
        self.retry_delay = float(self.cfg["openai"]["retry_delay_seconds"])
        self.use_json_mode = bool(self.cfg["openai"].get("json_mode", True))

    def _encode_image(self, image_path: Path) -> str:
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def _call_api(self, mime: str, b64: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": float(self.cfg["openai"]["temperature"]),
            "max_tokens": int(self.cfg["openai"]["max_tokens"]),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
        }
        if self.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def label_image(self, image_path: Path) -> MeterReading:
        suffix = image_path.suffix.lower().lstrip(".") or "jpeg"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        b64 = self._encode_image(image_path)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                content = self._call_api(mime, b64)
                return parse_meter_response(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"Falha ao rotular {image_path}") from last_error

    def label_and_save(self, image_path: Path, output_path: Path) -> MeterReading:
        reading = self.label_image(image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(reading.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return reading
