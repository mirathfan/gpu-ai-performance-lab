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
        description="Benchmark ResNet18 using ONNX Runtime TensorRT Execution Provider."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/resnet18.onnx"),
        help="Path to the exported ONNX model.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/resnet18_tensorrt_ep.csv"),
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--trt-cache-dir",
        type=Path,
        default=Path(".trt_cache/resnet18"),
        help="Directory used by TensorRT EP engine caching.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Ask TensorRT EP to enable FP16 engine building.",
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
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0.")


def validate_batch_size(session, batch_size: int) -> None:
    input_shape = session.get_inputs()[0].shape
    batch_dim = input_shape[0] if input_shape else None
    if isinstance(batch_dim, int) and batch_dim != batch_size:
        raise ValueError(
            f"ONNX model has fixed batch size {batch_dim}, but benchmark requested "
            f"batch size {batch_size}. Re-export with "
            "`python scripts/export_resnet18_onnx.py --dynamic-batch`."
        )


def get_gpu_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(torch.cuda.current_device())


def build_tensorrt_provider_options(cache_dir: Path, fp16: bool) -> dict[str, str]:
    """Return TensorRT EP options used by this benchmark.

    ONNX Runtime accepts provider options as strings. These options enable
    TensorRT engine caching so repeated benchmark runs can reuse build output
    instead of rebuilding the engine every time.
    """
    return {
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(cache_dir),
        "trt_fp16_enable": "True" if fp16 else "False",
    }


def create_session(ort, onnx_path: Path, trt_options: dict[str, str]):
    available = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in available:
        raise RuntimeError(
            "TensorrtExecutionProvider is not available in ONNX Runtime. "
            "Install a compatible onnxruntime-gpu build and TensorRT runtime, "
            "then confirm providers with `python -c \"import onnxruntime as "
            "ort; print(ort.get_available_providers())\"`."
        )

    providers = [
        ("TensorrtExecutionProvider", trt_options),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as exc:
        raise RuntimeError(f"Failed to create TensorRT EP session: {exc}") from exc
    if "TensorrtExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "ONNX Runtime created a session without TensorrtExecutionProvider. "
            f"Session providers: {session.get_providers()}"
        )
    return session


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
        "backend",
        "precision",
        "device",
        "provider",
        "mean_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "throughput_samples_per_sec",
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
            f"ONNX model not found: {onnx_path}. Run "
            "`python scripts/export_resnet18_onnx.py --dynamic-batch` first."
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

    cache_dir = resolve_repo_path(args.trt_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    trt_options = build_tensorrt_provider_options(cache_dir, args.fp16)
    print(f"TensorRT provider options used: {trt_options}")
    session = create_session(ort, onnx_path, trt_options)

    input_name = session.get_inputs()[0].name
    session_providers = session.get_providers()
    provider = session_providers[0]
    timestamp = datetime.now(timezone.utc).isoformat()
    precision = "fp16" if args.fp16 else "fp32"

    print(f"ONNX model: {onnx_path}")
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Session providers: {session_providers}")
    print(
        "Note: the first run for a shape/precision may include TensorRT engine "
        "build overhead. Engine caching is enabled so later runs can reuse cache "
        "files when compatible."
    )

    rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        validate_batch_size(session, batch_size)
        x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

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
                "batch_size": batch_size,
                "backend": "tensorrt_ep",
                "precision": precision,
                "device": args.device,
                "warmup_iters": args.warmup,
                "measurement_iters": args.iters,
                "provider": provider,
                **metrics,
                "model": "resnet18",
                "onnx_model": onnx_path.relative_to(REPO_ROOT).as_posix(),
                "onnxruntime_version": ort.__version__,
                "timestamp": timestamp,
                "platform": platform.platform(),
                "python_version": sys.version.replace("\n", " "),
                "gpu_name": get_gpu_name(),
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
