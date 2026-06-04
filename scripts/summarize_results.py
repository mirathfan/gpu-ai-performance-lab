from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

PYTORCH_REQUIRED_COLUMNS = {
    "batch_size",
    "precision",
    "device",
    "warmup_iters",
    "measurement_iters",
    "mean_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "throughput_samples_per_sec",
    "peak_gpu_memory_mb",
    "model",
    "weights",
    "timestamp",
    "platform",
    "python_version",
    "pytorch_version",
    "cuda_available",
    "cuda_version",
    "gpu_name",
    "gpu_vram_gb",
}

ONNXRUNTIME_REQUIRED_COLUMNS = {
    "batch_size",
    "backend",
    "precision",
    "device",
    "warmup_iters",
    "measurement_iters",
    "mean_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "throughput_samples_per_sec",
    "provider",
    "model",
    "timestamp",
    "platform",
    "python_version",
    "onnxruntime_version",
    "gpu_name",
}

TENSORRT_EP_REQUIRED_COLUMNS = {
    "batch_size",
    "backend",
    "precision",
    "device",
    "warmup_iters",
    "measurement_iters",
    "mean_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "throughput_samples_per_sec",
    "provider",
    "model",
    "onnxruntime_version",
    "timestamp",
    "platform",
    "python_version",
    "gpu_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a combined ResNet18 inference benchmark report."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing benchmark CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/resnet18_inference_benchmark.md"),
        help="Markdown report output path.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def relative_repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} is empty.")

    for row in rows:
        row["source_csv"] = relative_repo_path(path)
    return rows


def validate_required_columns(
    path: Path, rows: list[dict[str, str]], required_columns: set[str]
) -> None:
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")


def is_resnet18_pytorch_csv(path: Path, rows: list[dict[str, str]]) -> bool:
    filename = path.name.lower()
    model = rows[0].get("model", "").lower()
    return "resnet18" in filename and "pytorch" in filename and model == "resnet18"


def is_resnet18_onnxruntime_csv(path: Path, rows: list[dict[str, str]]) -> bool:
    filename = path.name.lower()
    backend = rows[0].get("backend", "").lower()
    model = rows[0].get("model", "").lower()
    return (
        "resnet18" in filename
        and "onnxruntime" in filename
        and backend == "onnxruntime"
        and model == "resnet18"
    )


def is_resnet18_tensorrt_ep_csv(path: Path, rows: list[dict[str, str]]) -> bool:
    filename = path.name.lower()
    backend = rows[0].get("backend", "").lower()
    model = rows[0].get("model", "").lower()
    return (
        "resnet18" in filename
        and "tensorrt_ep" in filename
        and backend == "tensorrt_ep"
        and model == "resnet18"
    )


def csv_score(path: Path, expected_filename: str) -> tuple[int, float]:
    score = 100 if path.name.lower() == expected_filename else 0
    return score, path.stat().st_mtime


def detect_benchmark_csvs(results_dir: Path) -> dict[str, tuple[Path, list[dict[str, str]]]]:
    """Find the project CSVs used by the combined ResNet18 report.

    PyTorch FP32 and FP16 are required because they are the Milestone 1 baseline.
    ONNX Runtime FP32 and TensorRT EP results are optional so the report and
    plots still work before the user has run those benchmark scripts.
    """
    candidates: dict[str, list[tuple[tuple[int, float], Path, list[dict[str, str]]]]] = {
        "pytorch_fp32": [],
        "pytorch_fp16": [],
        "onnxruntime_fp32": [],
        "tensorrt_ep_fp32": [],
        "tensorrt_ep_fp16": [],
    }

    for csv_path in sorted(results_dir.glob("*.csv")):
        rows = read_csv_rows(csv_path)

        if is_resnet18_pytorch_csv(csv_path, rows):
            precision = rows[0].get("precision", "").lower()
            if precision in {"fp32", "fp16"}:
                validate_required_columns(csv_path, rows, PYTORCH_REQUIRED_COLUMNS)
                key = f"pytorch_{precision}"
                candidates[key].append(
                    (
                        csv_score(csv_path, f"resnet18_pytorch_{precision}.csv"),
                        csv_path,
                        rows,
                    )
                )
            continue

        if is_resnet18_onnxruntime_csv(csv_path, rows):
            precision = rows[0].get("precision", "").lower()
            if precision == "fp32":
                validate_required_columns(csv_path, rows, ONNXRUNTIME_REQUIRED_COLUMNS)
                candidates["onnxruntime_fp32"].append(
                    (csv_score(csv_path, "resnet18_onnxruntime.csv"), csv_path, rows)
                )
            continue

        if is_resnet18_tensorrt_ep_csv(csv_path, rows):
            precision = rows[0].get("precision", "").lower()
            if precision in {"fp32", "fp16"}:
                validate_required_columns(csv_path, rows, TENSORRT_EP_REQUIRED_COLUMNS)
                candidates[f"tensorrt_ep_{precision}"].append(
                    (
                        csv_score(csv_path, f"resnet18_tensorrt_ep_{precision}.csv"),
                        csv_path,
                        rows,
                    )
                )

    detected: dict[str, tuple[Path, list[dict[str, str]]]] = {}
    for key, matches in candidates.items():
        if matches:
            _, csv_path, rows = max(matches, key=lambda match: match[0])
            detected[key] = (csv_path, rows)

    missing = [
        key.replace("pytorch_", "PyTorch ").upper()
        for key in ("pytorch_fp32", "pytorch_fp16")
        if key not in detected
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            "Could not find required ResNet18 PyTorch benchmark CSVs for: "
            f"{missing_text}. Expected files such as "
            "results/resnet18_pytorch_fp32.csv and "
            "results/resnet18_pytorch_fp16.csv."
        )

    return detected


def as_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} must contain numeric values.") from exc


def as_int(row: dict[str, str], column: str) -> int:
    try:
        return int(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Column {column!r} must contain integer values.") from exc


def optional_float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    if value in ("", "None", "nan", "NaN"):
        return None
    return float(value)


def has_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    return any(optional_float(row, column) is not None for row in rows)


def rows_by_batch(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {as_int(row, "batch_size"): row for row in rows}


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def first_value(rows: list[dict[str, str]], column: str) -> str:
    for row in rows:
        value = row.get(column, "")
        if value not in ("", "None"):
            return value
    return "n/a"


def backend_label(key: str) -> str:
    labels = {
        "pytorch_fp32": "PyTorch FP32",
        "pytorch_fp16": "PyTorch FP16",
        "onnxruntime_fp32": "ONNX Runtime FP32",
        "tensorrt_ep_fp32": "TensorRT EP FP32",
        "tensorrt_ep_fp16": "TensorRT EP FP16",
    }
    return labels[key]


def result_table(rows: list[dict[str, str]], *, include_provider: bool = False) -> str:
    include_memory = has_numeric_column(rows, "peak_gpu_memory_mb")
    headers = [
        "Batch",
        "Mean ms",
        "P50 ms",
        "P90 ms",
        "P95 ms",
        "P99 ms",
        "Min ms",
        "Max ms",
        "Throughput samples/s",
    ]
    if include_memory:
        headers.append("Peak memory MB")
    if include_provider:
        headers.append("Provider")

    table_rows: list[list[Any]] = []
    for row in sorted(rows, key=lambda item: as_int(item, "batch_size")):
        table_row: list[Any] = [
            as_int(row, "batch_size"),
            fmt_num(as_float(row, "mean_latency_ms")),
            fmt_num(as_float(row, "p50_latency_ms")),
            fmt_num(as_float(row, "p90_latency_ms")),
            fmt_num(as_float(row, "p95_latency_ms")),
            fmt_num(as_float(row, "p99_latency_ms")),
            fmt_num(as_float(row, "min_latency_ms")),
            fmt_num(as_float(row, "max_latency_ms")),
            fmt_num(as_float(row, "throughput_samples_per_sec")),
        ]
        if include_memory:
            table_row.append(fmt_num(optional_float(row, "peak_gpu_memory_mb")))
        if include_provider:
            table_row.append(first_value([row], "provider"))
        table_rows.append(table_row)

    return markdown_table(headers, table_rows)


def system_metadata_table(
    detected: dict[str, tuple[Path, list[dict[str, str]]]]
) -> str:
    all_rows = [row for _, rows in detected.values() for row in rows]
    table_rows = [
        ["GPU", first_value(all_rows, "gpu_name")],
        ["GPU VRAM GB", first_value(all_rows, "gpu_vram_gb")],
        ["Platform", first_value(all_rows, "platform")],
        ["Python", first_value(all_rows, "python_version")],
        ["PyTorch", first_value(all_rows, "pytorch_version")],
        ["ONNX Runtime", first_value(all_rows, "onnxruntime_version")],
        ["CUDA available", first_value(all_rows, "cuda_available")],
        ["CUDA version", first_value(all_rows, "cuda_version")],
    ]

    for key in (
        "pytorch_fp32",
        "pytorch_fp16",
        "onnxruntime_fp32",
        "tensorrt_ep_fp32",
        "tensorrt_ep_fp16",
    ):
        if key in detected:
            path, rows = detected[key]
            table_rows.append([f"{backend_label(key)} source CSV", relative_repo_path(path)])
            table_rows.append([f"{backend_label(key)} timestamp", first_value(rows, "timestamp")])

    return markdown_table(["Field", "Value"], table_rows)


def benchmark_config_table(
    detected: dict[str, tuple[Path, list[dict[str, str]]]]
) -> str:
    all_rows = [row for _, rows in detected.values() for row in rows]
    batch_sizes = sorted({as_int(row, "batch_size") for row in all_rows})
    backends = [backend_label(key) for key in detected]
    devices = sorted({row["device"] for row in all_rows})
    precisions = sorted({row["precision"] for row in all_rows})

    return markdown_table(
        ["Field", "Value"],
        [
            ["Model", first_value(all_rows, "model")],
            ["Weights", first_value(all_rows, "weights")],
            ["Backends", ", ".join(backends)],
            ["Input shape", "[batch_size, 3, 224, 224]"],
            ["Batch sizes", ", ".join(str(batch_size) for batch_size in batch_sizes)],
            ["Precision modes", ", ".join(precisions)],
            ["Device", ", ".join(devices)],
            ["Warmup iterations", first_value(all_rows, "warmup_iters")],
            ["Measurement iterations", first_value(all_rows, "measurement_iters")],
        ],
    )


def comparison_table(
    detected: dict[str, tuple[Path, list[dict[str, str]]]]
) -> str:
    series = {key: rows_by_batch(rows) for key, (_, rows) in detected.items()}
    required_keys = ["pytorch_fp32", "pytorch_fp16"]
    optional_keys = [
        key
        for key in ("onnxruntime_fp32", "tensorrt_ep_fp32", "tensorrt_ep_fp16")
        if key in detected
    ]
    compare_keys = required_keys + optional_keys
    batch_sets = [set(series[key]) for key in compare_keys]
    batch_sizes = sorted(set.intersection(*batch_sets))
    if not batch_sizes:
        raise ValueError("Detected CSV files do not share any comparable batch sizes.")

    headers = [
        "Batch",
        "PyTorch FP32 P50 ms",
        "PyTorch FP16 P50 ms",
        "FP16 latency delta",
        "PyTorch FP32 samples/s",
        "PyTorch FP16 samples/s",
        "FP16 throughput delta",
    ]
    if "onnxruntime_fp32" in detected:
        headers.extend(
            [
                "ONNX Runtime FP32 P50 ms",
                "ONNX latency delta",
                "ONNX Runtime samples/s",
                "ONNX throughput delta",
            ]
        )
    for key in ("tensorrt_ep_fp32", "tensorrt_ep_fp16"):
        if key in detected:
            label = backend_label(key)
            headers.extend(
                [
                    f"{label} P50 ms",
                    f"{label} latency delta",
                    f"{label} samples/s",
                    f"{label} throughput delta",
                ]
            )

    table_rows: list[list[Any]] = []
    for batch_size in batch_sizes:
        torch_fp32 = series["pytorch_fp32"][batch_size]
        torch_fp16 = series["pytorch_fp16"][batch_size]

        torch_fp32_p50 = as_float(torch_fp32, "p50_latency_ms")
        torch_fp16_p50 = as_float(torch_fp16, "p50_latency_ms")
        torch_fp32_throughput = as_float(torch_fp32, "throughput_samples_per_sec")
        torch_fp16_throughput = as_float(torch_fp16, "throughput_samples_per_sec")

        row: list[Any] = [
            batch_size,
            fmt_num(torch_fp32_p50),
            fmt_num(torch_fp16_p50),
            fmt_pct((torch_fp16_p50 - torch_fp32_p50) / torch_fp32_p50 * 100.0),
            fmt_num(torch_fp32_throughput),
            fmt_num(torch_fp16_throughput),
            fmt_pct(
                (torch_fp16_throughput - torch_fp32_throughput)
                / torch_fp32_throughput
                * 100.0
            ),
        ]

        if "onnxruntime_fp32" in detected:
            onnx_row = series["onnxruntime_fp32"][batch_size]
            onnx_p50 = as_float(onnx_row, "p50_latency_ms")
            onnx_throughput = as_float(onnx_row, "throughput_samples_per_sec")
            row.extend(
                [
                    fmt_num(onnx_p50),
                    fmt_pct((onnx_p50 - torch_fp32_p50) / torch_fp32_p50 * 100.0),
                    fmt_num(onnx_throughput),
                    fmt_pct(
                        (onnx_throughput - torch_fp32_throughput)
                        / torch_fp32_throughput
                        * 100.0
                    ),
                ]
            )

        for key in ("tensorrt_ep_fp32", "tensorrt_ep_fp16"):
            if key not in detected:
                continue
            trt_row = series[key][batch_size]
            trt_p50 = as_float(trt_row, "p50_latency_ms")
            trt_throughput = as_float(trt_row, "throughput_samples_per_sec")
            row.extend(
                [
                    fmt_num(trt_p50),
                    fmt_pct((trt_p50 - torch_fp32_p50) / torch_fp32_p50 * 100.0),
                    fmt_num(trt_throughput),
                    fmt_pct(
                        (trt_throughput - torch_fp32_throughput)
                        / torch_fp32_throughput
                        * 100.0
                    ),
                ]
            )

        table_rows.append(row)

    return markdown_table(headers, table_rows)


def fp16_memory_observation(
    fp32_rows: list[dict[str, str]], fp16_rows: list[dict[str, str]]
) -> str:
    fp32_by_batch = rows_by_batch(fp32_rows)
    fp16_by_batch = rows_by_batch(fp16_rows)
    batch_sizes = sorted(set(fp32_by_batch).intersection(fp16_by_batch))

    deltas = []
    fp32_values = []
    for batch_size in batch_sizes:
        fp32_memory = as_float(fp32_by_batch[batch_size], "peak_gpu_memory_mb")
        fp16_memory = as_float(fp16_by_batch[batch_size], "peak_gpu_memory_mb")
        deltas.append(fp16_memory - fp32_memory)
        fp32_values.append(fp32_memory)

    if not deltas:
        return "Peak GPU memory could not be compared because there are no shared batch sizes."

    lower_count = sum(delta < 0 for delta in deltas)
    average_delta = sum(deltas) / len(deltas)
    average_fp32 = sum(fp32_values) / len(fp32_values)
    average_delta_pct = average_delta / average_fp32 * 100.0

    if lower_count == len(deltas):
        direction = "lower than FP32 for every matched batch size"
    elif lower_count == 0:
        direction = "not lower than FP32 for the matched batch sizes"
    else:
        direction = (
            f"lower than FP32 for {lower_count} of {len(deltas)} matched batch sizes"
        )

    return (
        f"In these local CSVs, FP16 peak GPU memory is {direction}. "
        f"The average FP16 minus FP32 memory delta is {fmt_num(average_delta)} MB "
        f"({fmt_pct(average_delta_pct)})."
    )


def build_report(
    detected: dict[str, tuple[Path, list[dict[str, str]]]]
) -> str:
    _, fp32_rows = detected["pytorch_fp32"]
    _, fp16_rows = detected["pytorch_fp16"]
    onnx_available = "onnxruntime_fp32" in detected
    trt_keys = [
        key for key in ("tensorrt_ep_fp32", "tensorrt_ep_fp16") if key in detected
    ]

    sections = [
        "# ResNet18 Inference Benchmark Report",
        "",
        "This report summarizes locally generated ResNet18 inference benchmark "
        "CSV files from `results/`. Benchmark numbers should come from actual "
        "runs only and should not be manually invented.",
        "",
        "TensorRT results in this report use ONNX Runtime "
        "`TensorrtExecutionProvider`; raw TensorRT engine API benchmarks are "
        "not added yet.",
        "",
        "## System and Hardware Metadata",
        "",
        system_metadata_table(detected),
        "",
        "## Benchmark Configuration",
        "",
        benchmark_config_table(detected),
        "",
        "## PyTorch FP32 Results",
        "",
        result_table(fp32_rows),
        "",
        "## PyTorch FP16 Results",
        "",
        result_table(fp16_rows),
        "",
    ]

    if onnx_available:
        _, onnx_rows = detected["onnxruntime_fp32"]
        sections.extend(
            [
                "## ONNX Runtime FP32 Results",
                "",
                result_table(onnx_rows, include_provider=True),
                "",
            ]
        )
    else:
        sections.extend(
            [
                "## ONNX Runtime FP32 Results",
                "",
                "`results/resnet18_onnxruntime.csv` was not found. Run the "
                "Milestone 2 ONNX Runtime benchmark to add this section.",
                "",
            ]
        )

    if trt_keys:
        for key in trt_keys:
            _, trt_rows = detected[key]
            sections.extend(
                [
                    f"## {backend_label(key)} Results",
                    "",
                    result_table(trt_rows, include_provider=True),
                    "",
                ]
            )
    else:
        sections.extend(
            [
                "## TensorRT EP Results",
                "",
                "`results/resnet18_tensorrt_ep*.csv` was not found. Run the "
                "Milestone 3 TensorRT EP benchmark to add this section.",
                "",
            ]
        )

    sections.extend(
        [
            "## Backend Comparison",
            "",
            comparison_table(detected),
            "",
            "## TensorRT EP Notes",
            "",
            "TensorRT EP uses TensorRT through ONNX Runtime provider selection. "
            "The first run for a model shape and precision may build and cache "
            "a TensorRT engine, so initial runs can include build overhead. "
            "These results can differ from raw TensorRT API benchmarks because "
            "ONNX Runtime still owns session setup, graph partitioning, and "
            "provider fallback behavior. TensorRT is expected to help most when "
            "graph optimizations, precision lowering, and kernel selection "
            "improve execution for the target GPU.",
            "",
            "## Latency vs Throughput",
            "",
            "Latency measures how long one inference batch takes to complete. "
            "Throughput measures how many input samples are processed per second. "
            "A larger batch can increase throughput even when per-batch latency "
            "also increases, so both metrics are useful when choosing a deployment "
            "batch size.",
            "",
            "## FP16 Memory Observation",
            "",
            fp16_memory_observation(fp32_rows, fp16_rows),
            "",
            "ONNX Runtime peak GPU memory is not reported by the current benchmark "
            "script because ONNX Runtime does not expose a simple peak CUDA "
            "allocation counter through this beginner-readable path.",
            "",
            "## Reproducibility Note",
            "",
            "These tables are generated from local CSV files in `results/`. Do not "
            "edit benchmark numbers by hand; rerun the benchmark scripts when the "
            "environment, code, driver, or dependencies change.",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> int:
    args = parse_args()
    results_dir = resolve_repo_path(args.results_dir)
    output_path = resolve_repo_path(args.output)

    detected = detect_benchmark_csvs(results_dir)
    report = build_report(detected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Wrote report: {output_path}")
    if "onnxruntime_fp32" not in detected:
        print(
            "Note: results/resnet18_onnxruntime.csv was not found, so ONNX "
            "Runtime results were not included yet."
        )
    if not any(key.startswith("tensorrt_ep_") for key in detected):
        print(
            "Note: results/resnet18_tensorrt_ep*.csv was not found, so "
            "TensorRT EP results were not included yet."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
