"""Inferência VLM Florence-2 + adaptador LoRA."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from hidrometro.autolabel.openai_labeler import MeterReading, parse_meter_response
from hidrometro.config import load_yaml, resolve_path


class VLMInference:
    def __init__(self, adapter_dir: Path | None = None, use_adapter: bool = True) -> None:
        self.cfg = load_yaml("qlora.yaml")
        paths = load_yaml("paths.yaml")
        self.adapter_dir = adapter_dir or resolve_path(paths["output"]["lora_adapter"])
        self.use_adapter = use_adapter
        self.prompt = load_yaml("autolabel.yaml")["sft_prompt"]
        self.model = None
        self.processor = None
        self.torch_dtype = getattr(torch, self.cfg["model"]["torch_dtype"])

    def _prepare_inputs(self, inputs: dict) -> dict:
        device = next(self.model.parameters()).device
        prepared: dict = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                prepared[key] = value
                continue
            if value.is_floating_point():
                prepared[key] = value.to(device=device, dtype=self.torch_dtype)
            else:
                prepared[key] = value.to(device=device)
        return prepared

    def load(self) -> None:
        if self.model is not None:
            return

        model_cfg = self.cfg["model"]
        model_name = model_cfg["name"]
        torch_dtype = getattr(torch, model_cfg["torch_dtype"])
        trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=bool(model_cfg["load_in_4bit"]),
            bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=torch_dtype,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            attn_implementation="eager",
        )

        if (
            self.use_adapter
            and self.adapter_dir.exists()
            and any(self.adapter_dir.iterdir())
        ):
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
        else:
            self.model = base_model
        self.model.eval()

    def predict(self, image_bgr) -> MeterReading:
        import cv2

        self.load()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        prompt = self.prompt if self.prompt.startswith("<") else f"<VQA>{self.prompt}"
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = self._prepare_inputs(inputs)

        max_new_tokens = int(self.cfg["training"].get("max_new_tokens", 256))
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )

        decoded = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        return parse_meter_response(self._extract_response(decoded))

    @staticmethod
    def _extract_response(decoded: str) -> str:
        if decoded.strip().startswith("{"):
            return decoded.strip()
        match = re.search(r"\{.*\}", decoded, flags=re.DOTALL)
        return match.group(0) if match else decoded


def sanitize_json(reading: MeterReading) -> dict[str, Any]:
    payload = reading.to_dict()
    payload["fabricante"] = payload.get("fabricante", "").strip()
    payload["estado"] = payload.get("estado", "normal").strip().lower()
    leitura = payload["leitura"]
    if isinstance(leitura, dict):
        leitura["completo"] = str(leitura.get("completo", "")).strip()
    return payload
