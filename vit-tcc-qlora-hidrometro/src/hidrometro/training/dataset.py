"""Dataset SFT para Florence-2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from hidrometro.config import load_yaml, project_root


@dataclass
class SFTExample:
    image_path: Path
    prompt: str
    response: str


def load_jsonl(path: Path) -> list[SFTExample]:
    examples: list[SFTExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                SFTExample(
                    image_path=Path(row["image"]),
                    prompt=row["prompt"],
                    response=row["response"],
                )
            )
    return examples


class FlorenceSFTDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor) -> None:
        self.examples = load_jsonl(jsonl_path)
        self.processor = processor
        cfg = load_yaml("qlora.yaml")
        self.max_new_tokens = int(cfg["training"].get("max_new_tokens", 256))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        image_path = example.image_path
        if not image_path.is_absolute():
            image_path = (project_root() / image_path).resolve()
        image = Image.open(image_path).convert("RGB")

        prompt = example.prompt if example.prompt.startswith("<") else f"<VQA>{example.prompt}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}

        target = self.processor.tokenizer(
            example.response,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_new_tokens,
        )
        labels = target["input_ids"].squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    keys = batch[0].keys()
    output: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output

