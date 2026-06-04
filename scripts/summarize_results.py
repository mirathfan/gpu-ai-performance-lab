from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "resnet18_pytorch_baseline.md"

REQUIRED_COLUMNS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary for ResNet18 PyTorch CSV results."
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
        default=Path("reports/resnet18_pytorch_baseline.md"),
        help="Markdown report output path.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def relative_repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def validate_required_columns(path: Path, rows: list[dict[str, str]]) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(rows[0].keys())
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_csv_rows(path: Path, *, validate: bool = True) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} is empty.")

    if validate:
        validate_required_columns(path, rows)

    for row in rows:
        row["source_csv"] = relative_repo_path(path)
    return rows


def is_resnet18_pytorch_csv(path: Path, rows: list[dict[str, str]]) -> bool:
    filename = path.name.lower()
    model = rows[0].get("model", "").lower()
    return "resnet18" in filename and "pytorch" in filename and model == "resnet18"


def csv_score(path: Path, precision: str) -> tuple[int, float]:
    filename = path.name.lower()
    score = 0
    if precision in filename:
        score += 10
    if filename == f"resnet18_pytorch_{precision}.csv":
        score += 100
    return score, path.stat().st_mtime


def detect_benchmark_csvs(results_dir: Path) -> dict[str, tuple[Path, list[dict[str, str]]]]:
    candidates: dict[str, list[tuple[tuple[int, float], Path, list[dict[str, str]]]]] = {
        "fp32": [],
        "fp16": [],
    }

    for csv_path in sorted(results_dir.glob("*.csv")):
        rows = read_csv_rows(csv_path, validate=False)
        if not is_resnet18_pytorch_csv(csv_path, rows):
            continue

        precision = rows[0].get("precision", "").lower()
        if precision in candidates:
            validate_required_columns(csv_path, rows)
            candidates[precision].append((csv_score(csv_path, precision), csv_path, rows))

    detected: dict[str, tuple[Path, list[dict[str, str]]]] = {}
    for precision, matches in candidates.items():
        if matches:
            _, csv_path, rows = max(matches, key=lambda match: match[0])
            detected[precision] = (csv_path, rows)

    missing = [precision for precision in ("fp32", "fp16") if precision not in detected]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            "Could not find ResNet18 PyTorch benchmark CSVs for: "
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


def rows_by_batch(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {as_int(row, "batch_size"): row for row in rows}


def fmt_num(value: float, digits: int = 3) -> str:
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


def result_table(rows: list[dict[str, str]]) -> str:
    sorted_rows = sorted(rows, key=lambda row: as_int(row, "batch_size"))
    table_rows: list[list[Any]] = []
    for row in sorted_rows:
        table_rows.append(
            [
                as_int(row, "batch_size"),
                fmt_num(as_float(row, "mean_latency_ms")),
                fmt_num(as_float(row, "p50_latency_ms")),
                fmt_num(as_float(row, "p90_latency_ms")),
                fmt_num(as_float(row, "p95_latency_ms")),
                fmt_num(as_float(row, "p99_latency_ms")),
                fmt_num(as_float(row, "min_latency_ms")),
                fmt_num(as_float(row, "max_latency_ms")),
                fmt_num(as_float(row, "throughput_samples_per_sec")),
                fmt_num(as_float(row, "peak_gpu_memory_mb")),
            ]
        )

    return markdown_table(
        [
            "Batch",
            "Mean ms",
            "P50 ms",
            "P90 ms",
            "P95 ms",
            "P99 ms",
            "Min ms",
            "Max ms",
            "Throughput samples/s",
            "Peak memory MB",
        ],
        table_rows,
    )


def comparison_table(
    fp32_rows: list[dict[str, str]], fp16_rows: list[dict[str, str]]
) -> str:
    fp32_by_batch = rows_by_batch(fp32_rows)
    fp16_by_batch = rows_by_batch(fp16_rows)
    batch_sizes = sorted(set(fp32_by_batch).intersection(fp16_by_batch))
    if not batch_sizes:
        raise ValueError("FP32 and FP16 CSV files do not share any batch sizes.")

    table_rows: list[list[Any]] = []
    for batch_size in batch_sizes:
        fp32 = fp32_by_batch[batch_size]
        fp16 = fp16_by_batch[batch_size]

        fp32_p50 = as_float(fp32, "p50_latency_ms")
        fp16_p50 = as_float(fp16, "p50_latency_ms")
        fp32_throughput = as_float(fp32, "throughput_samples_per_sec")
        fp16_throughput = as_float(fp16, "throughput_samples_per_sec")
        fp32_memory = as_float(fp32, "peak_gpu_memory_mb")
        fp16_memory = as_float(fp16, "peak_gpu_memory_mb")

        latency_delta_pct = (fp16_p50 - fp32_p50) / fp32_p50 * 100.0
        throughput_delta_pct = (
            (fp16_throughput - fp32_throughput) / fp32_throughput * 100.0
        )
        memory_delta_mb = fp16_memory - fp32_memory

        table_rows.append(
            [
                batch_size,
                fmt_num(fp32_p50),
                fmt_num(fp16_p50),
                fmt_pct(latency_delta_pct),
                fmt_num(fp32_throughput),
                fmt_num(fp16_throughput),
                fmt_pct(throughput_delta_pct),
                fmt_num(fp32_memory),
                fmt_num(fp16_memory),
                fmt_num(memory_delta_mb),
            ]
        )

    return markdown_table(
        [
            "Batch",
            "FP32 P50 ms",
            "FP16 P50 ms",
            "FP16 latency delta",
            "FP32 samples/s",
            "FP16 samples/s",
            "FP16 throughput delta",
            "FP32 memory MB",
            "FP16 memory MB",
            "Memory delta MB",
        ],
        table_rows,
    )


def first_value(rows: list[dict[str, str]], column: str) -> str:
    for row in rows:
        value = row.get(column, "")
        if value not in ("", "None"):
            return value
    return "n/a"


def system_metadata_table(
    fp32_path: Path,
    fp32_rows: list[dict[str, str]],
    fp16_path: Path,
    fp16_rows: list[dict[str, str]],
) -> str:
    rows = fp32_rows + fp16_rows
    return markdown_table(
        ["Field", "Value"],
        [
            ["GPU", first_value(rows, "gpu_name")],
            ["GPU VRAM GB", first_value(rows, "gpu_vram_gb")],
            ["Platform", first_value(rows, "platform")],
            ["Python", first_value(rows, "python_version")],
            ["PyTorch", first_value(rows, "pytorch_version")],
            ["CUDA available", first_value(rows, "cuda_available")],
            ["CUDA version", first_value(rows, "cuda_version")],
            ["FP32 source CSV", relative_repo_path(fp32_path)],
            ["FP16 source CSV", relative_repo_path(fp16_path)],
            ["FP32 timestamp", first_value(fp32_rows, "timestamp")],
            ["FP16 timestamp", first_value(fp16_rows, "timestamp")],
        ],
    )


def benchmark_config_table(
    fp32_rows: list[dict[str, str]], fp16_rows: list[dict[str, str]]
) -> str:
    rows = fp32_rows + fp16_rows
    batch_sizes = sorted({as_int(row, "batch_size") for row in rows})
    precisions = sorted({row["precision"] for row in rows})
    devices = sorted({row["device"] for row in rows})

    return markdown_table(
        ["Field", "Value"],
        [
            ["Model", first_value(rows, "model")],
            ["Weights", first_value(rows, "weights")],
            ["Framework", "PyTorch"],
            ["Input shape", "[batch_size, 3, 224, 224]"],
            ["Batch sizes", ", ".join(str(batch_size) for batch_size in batch_sizes)],
            ["Precision modes", ", ".join(precisions)],
            ["Device", ", ".join(devices)],
            ["Warmup iterations", first_value(rows, "warmup_iters")],
            ["Measurement iterations", first_value(rows, "measurement_iters")],
        ],
    )


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
    fp32_path: Path,
    fp32_rows: list[dict[str, str]],
    fp16_path: Path,
    fp16_rows: list[dict[str, str]],
) -> str:
    sections = [
        "# ResNet18 PyTorch Baseline Report",
        "",
        "This report summarizes locally generated PyTorch ResNet18 inference "
        "benchmark CSV files from `results/`. Benchmark numbers should come "
        "from actual runs only and should not be manually invented.",
        "",
        "## System and Hardware Metadata",
        "",
        system_metadata_table(fp32_path, fp32_rows, fp16_path, fp16_rows),
        "",
        "## Benchmark Configuration",
        "",
        benchmark_config_table(fp32_rows, fp16_rows),
        "",
        "## FP32 Results",
        "",
        result_table(fp32_rows),
        "",
        "## FP16 Results",
        "",
        result_table(fp16_rows),
        "",
        "## FP32 vs FP16 Comparison",
        "",
        comparison_table(fp32_rows, fp16_rows),
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
        "## Reproducibility Note",
        "",
        "These tables are generated from local CSV files in `results/`. Do not "
        "edit benchmark numbers by hand; rerun the benchmark scripts when the "
        "environment, code, driver, or dependencies change.",
        "",
    ]
    return "\n".join(sections)


def main() -> int:
    args = parse_args()
    results_dir = resolve_repo_path(args.results_dir)
    output_path = resolve_repo_path(args.output)

    detected = detect_benchmark_csvs(results_dir)
    fp32_path, fp32_rows = detected["fp32"]
    fp16_path, fp16_rows = detected["fp16"]

    report = build_report(fp32_path, fp32_rows, fp16_path, fp16_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Wrote report: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
