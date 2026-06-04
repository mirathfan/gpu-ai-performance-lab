"""Metric helpers for benchmark latency samples."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_latency(latencies_ms: list[float], *, batch_size: int) -> dict[str, Any]:
    """Compute latency percentiles and throughput from raw latency samples."""
    if not latencies_ms:
        raise ValueError("latencies_ms must contain at least one sample")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    samples = np.asarray(latencies_ms, dtype=np.float64)
    mean_latency_ms = float(np.mean(samples))
    throughput = float(batch_size * 1000.0 / mean_latency_ms)

    return {
        "mean_latency_ms": mean_latency_ms,
        "p50_latency_ms": float(np.percentile(samples, 50)),
        "median_latency_ms": float(np.percentile(samples, 50)),
        "p90_latency_ms": float(np.percentile(samples, 90)),
        "p95_latency_ms": float(np.percentile(samples, 95)),
        "p99_latency_ms": float(np.percentile(samples, 99)),
        "min_latency_ms": float(np.min(samples)),
        "max_latency_ms": float(np.max(samples)),
        "throughput_samples_per_sec": throughput,
    }
