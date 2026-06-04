from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_results import as_float, as_int, detect_benchmark_csvs, rows_by_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ResNet18 PyTorch benchmark CSV results."
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


def matched_series(
    fp32_rows: list[dict[str, str]], fp16_rows: list[dict[str, str]], metric: str
) -> tuple[list[int], list[float], list[float]]:
    fp32_by_batch = rows_by_batch(fp32_rows)
    fp16_by_batch = rows_by_batch(fp16_rows)
    batch_sizes = sorted(set(fp32_by_batch).intersection(fp16_by_batch))
    if not batch_sizes:
        raise ValueError("FP32 and FP16 CSV files do not share any batch sizes.")

    fp32_values = [as_float(fp32_by_batch[batch_size], metric) for batch_size in batch_sizes]
    fp16_values = [as_float(fp16_by_batch[batch_size], metric) for batch_size in batch_sizes]
    return batch_sizes, fp32_values, fp16_values


def plot_metric(
    *,
    plt,
    batch_sizes: list[int],
    fp32_values: list[float],
    fp16_values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, fp32_values, marker="o", linewidth=2, label="FP32")
    plt.plot(batch_sizes, fp16_values, marker="o", linewidth=2, label="FP16")
    plt.xlabel("Batch size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(batch_sizes)
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
    _, fp32_rows = detected["fp32"]
    _, fp16_rows = detected["fp16"]

    batch_sizes, fp32_latency, fp16_latency = matched_series(
        fp32_rows, fp16_rows, "p50_latency_ms"
    )
    plot_metric(
        plt=plt,
        batch_sizes=batch_sizes,
        fp32_values=fp32_latency,
        fp16_values=fp16_latency,
        ylabel="P50 latency (ms)",
        title="ResNet18 PyTorch P50 Latency vs Batch Size",
        output_path=output_dir / "resnet18_latency_vs_batch.png",
    )

    batch_sizes, fp32_throughput, fp16_throughput = matched_series(
        fp32_rows, fp16_rows, "throughput_samples_per_sec"
    )
    plot_metric(
        plt=plt,
        batch_sizes=batch_sizes,
        fp32_values=fp32_throughput,
        fp16_values=fp16_throughput,
        ylabel="Throughput (samples/sec)",
        title="ResNet18 PyTorch Throughput vs Batch Size",
        output_path=output_dir / "resnet18_throughput_vs_batch.png",
    )

    batch_sizes, fp32_memory, fp16_memory = matched_series(
        fp32_rows, fp16_rows, "peak_gpu_memory_mb"
    )
    plot_metric(
        plt=plt,
        batch_sizes=batch_sizes,
        fp32_values=fp32_memory,
        fp16_values=fp16_memory,
        ylabel="Peak GPU memory (MB)",
        title="ResNet18 PyTorch Peak GPU Memory vs Batch Size",
        output_path=output_dir / "resnet18_memory_vs_batch.png",
    )

    print(f"Wrote figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
