from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_results import (
    backend_label,
    detect_benchmark_csvs,
    optional_float,
    rows_by_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ResNet18 inference benchmark CSV results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing benchmark CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory for generated PNG figures.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def metric_series(
    detected: dict[str, tuple[Path, list[dict[str, str]]]], metric: str
) -> list[tuple[str, list[int], list[float]]]:
    ordered_keys = [
        "pytorch_fp32",
        "pytorch_fp16",
        "onnxruntime_fp32",
        "tensorrt_ep_fp32",
        "tensorrt_ep_fp16",
    ]
    series: list[tuple[str, list[int], list[float]]] = []

    for key in ordered_keys:
        if key not in detected:
            continue

        _, rows = detected[key]
        by_batch = rows_by_batch(rows)
        batch_sizes: list[int] = []
        values: list[float] = []
        for batch_size, row in sorted(by_batch.items()):
            value = optional_float(row, metric)
            if value is None:
                continue
            batch_sizes.append(batch_size)
            values.append(value)

        if batch_sizes:
            series.append((backend_label(key), batch_sizes, values))

    return series


def plot_metric(
    *,
    plt,
    series: list[tuple[str, list[int], list[float]]],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if not series:
        raise ValueError(f"No numeric data available for plot: {title}")

    all_batch_sizes = sorted({batch_size for _, xs, _ in series for batch_size in xs})

    plt.figure(figsize=(8, 5))
    for label, batch_sizes, values in series:
        plt.plot(batch_sizes, values, marker="o", linewidth=2, label=label)

    plt.xlabel("Batch size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(all_batch_sizes)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    results_dir = resolve_repo_path(args.results_dir)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc

    detected = detect_benchmark_csvs(results_dir)

    plot_metric(
        plt=plt,
        series=metric_series(detected, "p50_latency_ms"),
        ylabel="P50 latency (ms)",
        title="ResNet18 P50 Latency vs Batch Size",
        output_path=output_dir / "resnet18_latency_vs_batch.png",
    )

    plot_metric(
        plt=plt,
        series=metric_series(detected, "throughput_samples_per_sec"),
        ylabel="Throughput (samples/sec)",
        title="ResNet18 Throughput vs Batch Size",
        output_path=output_dir / "resnet18_throughput_vs_batch.png",
    )

    plot_metric(
        plt=plt,
        series=metric_series(detected, "peak_gpu_memory_mb"),
        ylabel="Peak GPU memory (MB)",
        title="ResNet18 Peak GPU Memory vs Batch Size",
        output_path=output_dir / "resnet18_memory_vs_batch.png",
    )

    print(f"Wrote figures to: {output_dir}")
    if "onnxruntime_fp32" not in detected:
        print(
            "Note: results/resnet18_onnxruntime.csv was not found, so ONNX "
            "Runtime was not included in latency/throughput plots yet."
        )
    if not any(key.startswith("tensorrt_ep_") for key in detected):
        print(
            "Note: results/resnet18_tensorrt_ep*.csv was not found, so "
            "TensorRT EP was not included in plots yet."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
