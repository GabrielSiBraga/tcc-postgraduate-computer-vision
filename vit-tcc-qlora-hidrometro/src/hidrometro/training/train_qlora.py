"""Treinamento QLoRA com Florence-2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from hidrometro.config import load_yaml, resolve_path
from hidrometro.training.dataset import FlorenceSFTDataset, collate_fn


def build_bnb_config(cfg: dict[str, Any]) -> BitsAndBytesConfig:
    model_cfg = cfg["model"]
    compute_dtype = getattr(torch, model_cfg["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
        bnb_4bit_quant_type=model_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model_and_processor(cfg: dict[str, Any]):
    model_name = cfg["model"]["name"]
    torch_dtype = getattr(torch, cfg["model"]["torch_dtype"])
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))
    bnb_config = build_bnb_config(cfg)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    return model, processor


def train_qlora(
    train_jsonl: Path,
    val_jsonl: Path | None = None,
    config_name: str = "qlora.yaml",
    output_dir: Path | None = None,
    num_epochs: int | None = None,
) -> Path:
    from transformers import Trainer, TrainingArguments

    cfg = load_yaml(config_name)
    model, processor = load_model_and_processor(cfg)
    train_dataset = FlorenceSFTDataset(train_jsonl, processor)
    eval_dataset = FlorenceSFTDataset(val_jsonl, processor) if val_jsonl else None

    train_cfg = cfg["training"]
    out_dir = output_dir or resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    use_eval = eval_dataset is not None
    load_best = bool(train_cfg.get("load_best_model_at_end", True)) and use_eval

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=num_epochs or int(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        optim=train_cfg["optim"],
        bf16=bool(train_cfg["bf16"]),
        fp16=bool(train_cfg["fp16"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_strategy="steps" if use_eval else "epoch",
        eval_strategy="steps" if use_eval else "no",
        eval_steps=int(train_cfg["eval_steps"]) if use_eval else None,
        load_best_model_at_end=load_best,
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        save_total_limit=int(train_cfg.get("save_total_limit", 3)) if use_eval else None,
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        report_to=train_cfg["report_to"],
        remove_unused_columns=False,
    )

    callbacks = []
    patience = train_cfg.get("early_stopping_patience")
    if use_eval and patience:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(patience)))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if best_ckpt:
        print(f"Melhor checkpoint (eval_loss): {best_ckpt}")
        print(f"Melhor metric: {trainer.state.best_metric}")
    print(f"Train loss final: {train_result.training_loss:.4f}")

    return out_dir
