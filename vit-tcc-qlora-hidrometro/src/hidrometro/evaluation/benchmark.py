"""Benchmark de engenharia: VRAM, tamanho LoRA e throughput."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch


@dataclass
class BenchmarkResult:
    vram_peak_mb: float
    adapter_size_mb: float
    throughput_ips: float
    num_samples: int


def adapter_size_mb(adapter_dir: Path) -> float:
    if not adapter_dir.exists():
        return 0.0
    total = sum(
        f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file()
    )
    return total / (1024 * 1024)


def measure_vram_peak() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    peak = torch.cuda.max_memory_allocated()
    return peak / (1024 * 1024)


def benchmark_throughput(run_fn, samples: list, warmup: int = 1) -> BenchmarkResult:
    if not samples:
        return BenchmarkResult(0.0, 0.0, 0.0, 0)

    for sample in samples[:warmup]:
        run_fn(sample)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = perf_counter()
    for sample in samples:
        run_fn(sample)
    elapsed = perf_counter() - start

    vram = measure_vram_peak()
    return BenchmarkResult(
        vram_peak_mb=vram,
        adapter_size_mb=0.0,
        throughput_ips=len(samples) / max(elapsed, 1e-6),
        num_samples=len(samples),
    )
