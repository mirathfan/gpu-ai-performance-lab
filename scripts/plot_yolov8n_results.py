from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from summarize_yolov8n_results import (
    as_float,
    backend_label,
    detect_yolo_csvs,
    rows_by_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot YOLOv8n benchmark results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing YOLOv8n benchmark CSV files.",
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
    ordered_keys = ["onnxruntime_fp32", "tensorrt_ep_fp32", "tensorrt_ep_fp16"]
    series: list[tuple[str, list[int], list[float]]] = []
    for key in ordered_keys:
        if key not in detected:
            continue
        _, rows = detected[key]
        by_batch = rows_by_batch(rows)
        batch_sizes = sorted(by_batch)
        values = [as_float(by_batch[batch_size], metric) for batch_size in batch_sizes]
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
        raise ValueError("No YOLOv8n benchmark CSV files were found to plot.")

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

    detected = detect_yolo_csvs(results_dir)
    plot_metric(
        plt=plt,
        series=metric_series(detected, "p50_latency_ms"),
        ylabel="P50 latency (ms)",
        title="YOLOv8n P50 Latency vs Batch Size",
        output_path=output_dir / "yolov8n_latency_vs_batch.png",
    )
    plot_metric(
        plt=plt,
        series=metric_series(detected, "throughput_images_per_sec"),
        ylabel="Throughput (images/sec)",
        title="YOLOv8n Throughput vs Batch Size",
        output_path=output_dir / "yolov8n_throughput_vs_batch.png",
    )

    print(f"Wrote YOLOv8n figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
