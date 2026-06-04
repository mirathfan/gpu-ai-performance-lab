from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch ResNet18 inference latency."
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/resnet18_pytorch.csv"),
        help="CSV path for benchmark results.",
    )
    return parser.parse_args()


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc
    return torch


def validate_args(args: argparse.Namespace) -> None:
    torch = import_torch()

    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive integers.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available to PyTorch. Run "
            "`python scripts/check_gpu.py` to diagnose the environment."
        )
    if args.precision == "fp16" and args.device != "cuda":
        raise RuntimeError("FP16 benchmarking is only enabled on CUDA for this script.")


def load_resnet18(device: torch.device, precision: str) -> torch.nn.Module:
    try:
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc

    weights_enum = getattr(models, "ResNet18_Weights", None)
    try:
        if weights_enum is not None:
            weights = weights_enum.DEFAULT
            model = models.resnet18(weights=weights)
            weights_label = str(weights)
        else:
            model = models.resnet18(pretrained=True)
            weights_label = "pretrained=True"
    except Exception as exc:
        print(
            "Could not load pretrained ResNet18 weights; falling back to weights=None. "
            f"Reason: {exc}"
        )
        if weights_enum is not None:
            model = models.resnet18(weights=None)
        else:
            model = models.resnet18(pretrained=False)
        weights_label = "None"

    model.eval()
    model.to(device)
    if precision == "fp16":
        model.half()

    model.weights_label = weights_label  # type: ignore[attr-defined]
    return model


def make_input(batch_size: int, device: torch.device, precision: str) -> torch.Tensor:
    torch = import_torch()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    if precision == "fp16":
        x = x.half()
    return x


def benchmark_batch_size(
    *,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    precision: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    torch = import_torch()
    from gpu_ai_perf.metrics import summarize_latency
    from gpu_ai_perf.timing import benchmark_latency_ms

    x = make_input(batch_size, device, precision)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def infer() -> torch.Tensor:
        with torch.no_grad():
            return model(x)

    latencies_ms = benchmark_latency_ms(
        infer,
        warmup=warmup,
        iters=iters,
        device=device,
    )

    metrics = summarize_latency(latencies_ms, batch_size=batch_size)
    peak_gpu_memory_mb = None
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "batch_size": batch_size,
        "precision": precision,
        "device": str(device),
        "warmup_iters": warmup,
        "measurement_iters": iters,
        **metrics,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }


def format_results_table(rows: list[dict[str, Any]]) -> str:
    display_columns = [
        "batch_size",
        "precision",
        "device",
        "mean_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "throughput_samples_per_sec",
        "peak_gpu_memory_mb",
    ]
    table_rows = []
    for row in rows:
        table_row = {}
        for column in display_columns:
            value = row[column]
            if isinstance(value, float):
                table_row[column] = f"{value:.3f}"
            elif value is None:
                table_row[column] = "n/a"
            else:
                table_row[column] = value
        table_rows.append(table_row)

    try:
        from tabulate import tabulate

        return tabulate(table_rows, headers="keys", tablefmt="github")
    except ImportError:
        return format_plain_table(table_rows, display_columns)


def format_plain_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Small fallback table formatter used if tabulate is not installed yet."""
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(str(row[column]).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No benchmark rows were generated.")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_output_path(output: Path) -> Path:
    if output.is_absolute():
        return output
    return REPO_ROOT / output


def main() -> int:
    args = parse_args()
    validate_args(args)

    torch = import_torch()
    from gpu_ai_perf.system_info import get_system_info

    device = torch.device(args.device)
    model = load_resnet18(device, args.precision)
    system_info = get_system_info()

    rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        print(f"Benchmarking batch size {batch_size}...")
        row = benchmark_batch_size(
            model=model,
            batch_size=batch_size,
            device=device,
            precision=args.precision,
            warmup=args.warmup,
            iters=args.iters,
        )
        row["model"] = "resnet18"
        row["weights"] = getattr(model, "weights_label", "unknown")
        row.update(system_info)
        rows.append(row)

    output_path = resolve_output_path(args.output)
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
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
