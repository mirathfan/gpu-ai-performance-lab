from __future__ import annotations

import argparse
import csv
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv8n ONNX Runtime CUDA inference latency."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/yolov8n.onnx"),
        help="Path to the exported YOLOv8n ONNX model.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/yolov8n_onnxruntime.csv"),
        help="CSV path for benchmark results.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc
    return ort


def validate_args(args: argparse.Namespace) -> None:
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive integers.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be > 0.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0.")


def select_providers(ort) -> list[str]:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available in ONNX Runtime. Install "
            "`onnxruntime-gpu` and verify your CUDA/cuDNN environment."
        )
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def validate_input_shape(session, batch_size: int, image_size: int) -> None:
    shape = session.get_inputs()[0].shape
    if not shape or len(shape) != 4:
        return

    expected = [batch_size, 3, image_size, image_size]
    for actual_dim, expected_dim in zip(shape, expected):
        if isinstance(actual_dim, int) and actual_dim != expected_dim:
            raise ValueError(
                f"ONNX model input shape {shape} is incompatible with requested "
                f"input shape {expected}. Re-export with `--dynamic` or use the "
                "matching benchmark arguments."
            )


def get_gpu_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(torch.cuda.current_device())


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No benchmark rows were generated.")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_results_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "batch_size",
        "image_size",
        "backend",
        "precision",
        "provider",
        "p50_latency_ms",
        "p90_latency_ms",
        "throughput_images_per_sec",
    ]
    table_rows = []
    for row in rows:
        formatted = {}
        for column in columns:
            value = row[column]
            formatted[column] = f"{value:.3f}" if isinstance(value, float) else value
        table_rows.append(formatted)

    try:
        from tabulate import tabulate

        return tabulate(table_rows, headers="keys", tablefmt="github")
    except ImportError:
        widths = {
            column: max(len(column), *(len(str(row[column])) for row in table_rows))
            for column in columns
        }
        header = " | ".join(column.ljust(widths[column]) for column in columns)
        separator = "-+-".join("-" * widths[column] for column in columns)
        body = [
            " | ".join(str(row[column]).ljust(widths[column]) for column in columns)
            for row in table_rows
        ]
        return "\n".join([header, separator, *body])


def main() -> int:
    args = parse_args()
    validate_args(args)

    onnx_path = resolve_repo_path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"YOLOv8n ONNX model not found: {onnx_path}. Run "
            "`python scripts/export_yolov8n_onnx.py --dynamic` first."
        )

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc

    ort = import_onnxruntime()
    from gpu_ai_perf.metrics import summarize_latency
    from gpu_ai_perf.timing import benchmark_latency_ms

    providers = select_providers(ort)
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as exc:
        raise RuntimeError(f"Failed to create ONNX Runtime session: {exc}") from exc

    input_name = session.get_inputs()[0].name
    session_providers = session.get_providers()
    provider = session_providers[0]

    print(f"ONNX model: {onnx_path}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Session providers: {session_providers}")

    timestamp = datetime.now(timezone.utc).isoformat()
    gpu_name = get_gpu_name()
    rows: list[dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        validate_input_shape(session, batch_size, args.imgsz)
        x = np.random.rand(batch_size, 3, args.imgsz, args.imgsz).astype(np.float32)

        def infer() -> list[np.ndarray]:
            return session.run(None, {input_name: x})

        print(f"Benchmarking batch size {batch_size}...")
        latencies_ms = benchmark_latency_ms(
            infer,
            warmup=args.warmup,
            iters=args.iters,
            device=args.device,
        )
        metrics = summarize_latency(latencies_ms, batch_size=batch_size)

        rows.append(
            {
                "model": "yolov8n",
                "backend": "onnxruntime",
                "precision": "fp32",
                "device": args.device,
                "batch_size": batch_size,
                "image_size": args.imgsz,
                "warmup_iters": args.warmup,
                "measurement_iters": args.iters,
                "provider": provider,
                "mean_latency_ms": metrics["mean_latency_ms"],
                "p50_latency_ms": metrics["p50_latency_ms"],
                "p90_latency_ms": metrics["p90_latency_ms"],
                "p95_latency_ms": metrics["p95_latency_ms"],
                "p99_latency_ms": metrics["p99_latency_ms"],
                "min_latency_ms": metrics["min_latency_ms"],
                "max_latency_ms": metrics["max_latency_ms"],
                "throughput_images_per_sec": metrics["throughput_samples_per_sec"],
                "timestamp": timestamp,
                "platform": platform.platform(),
                "python_version": sys.version.replace("\n", " "),
                "onnxruntime_version": ort.__version__,
                "gpu_name": gpu_name,
            }
        )

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_path)

    print()
    print(format_results_table(rows))
    print()
    print(f"Saved benchmark results to: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
