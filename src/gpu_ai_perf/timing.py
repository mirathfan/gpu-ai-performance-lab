"""Timing helpers for inference benchmarks.

CUDA kernels launch asynchronously, so wall-clock timing can under-report GPU
work unless we synchronize around the measured section. These helpers centralize
that behavior so benchmark scripts do not accidentally measure only launch time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import torch

T = TypeVar("T")


def synchronize_if_needed(device: torch.device | str | None = None) -> None:
    """Synchronize CUDA work when the selected device is CUDA."""
    if not torch.cuda.is_available():
        return

    if device is None:
        torch.cuda.synchronize()
        return

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def benchmark_latency_ms(
    fn: Callable[[], T],
    *,
    warmup: int,
    iters: int,
    device: torch.device | str | None = None,
) -> list[float]:
    """Run warmup iterations, then return per-iteration latencies in ms.

    Warmup iterations give PyTorch, cuDNN, and the GPU a chance to settle before
    measurement. The returned raw samples make it possible to compute additional
    statistics later without rerunning the benchmark.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iters <= 0:
        raise ValueError("iters must be > 0")

    for _ in range(warmup):
        fn()

    synchronize_if_needed(device)

    latencies_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        synchronize_if_needed(device)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    return latencies_ms
