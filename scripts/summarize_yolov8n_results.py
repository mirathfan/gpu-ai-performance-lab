from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = {
    "model",
    "backend",
    "precision",
    "device",
    "batch_size",
    "image_size",
    "warmup_iters",
    "measurement_iters",
    "provider",
    "mean_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "throughput_images_per_sec",
    "timestamp",
    "platform",
    "python_version",
    "onnxruntime_version",
    "gpu_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a YOLOv8n inference benchmark Markdown report."
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
        default=Path("reports/yolov8n_inference_benchmark.md"),
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


def validate_required_columns(path: Path, rows: list[dict[str, str]]) -> None:
    missing = REQUIRED_COLUMNS.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")


def is_yolov8n_csv(path: Path, rows: list[dict[str, str]]) -> bool:
    filename = path.name.lower()
    model = rows[0].get("model", "").lower()
    return "yolov8n" in filename and model == "yolov8n"


def csv_score(path: Path, expected_filename: str) -> tuple[int, float]:
    score = 100 if path.name.lower() == expected_filename else 0
    return score, path.stat().st_mtime


def detect_yolo_csvs(results_dir: Path) -> dict[str, tuple[Path, list[dict[str, str]]]]:
    candidates: dict[str, list[tuple[tuple[int, float], Path, list[dict[str, str]]]]] = {
        "onnxruntime_fp32": [],
        "tensorrt_ep_fp32": [],
        "tensorrt_ep_fp16": [],
    }

    for csv_path in sorted(results_dir.glob("*.csv")):
        rows = read_csv_rows(csv_path)
        if not is_yolov8n_csv(csv_path, rows):
            continue
        validate_required_columns(csv_path, rows)

        backend = rows[0].get("backend", "").lower()
        precision = rows[0].get("precision", "").lower()
        if backend == "onnxruntime" and precision == "fp32":
            candidates["onnxruntime_fp32"].append(
                (csv_score(csv_path, "yolov8n_onnxruntime.csv"), csv_path, rows)
            )
        elif backend == "tensorrt_ep" and precision in {"fp32", "fp16"}:
            key = f"tensorrt_ep_{precision}"
            candidates[key].append(
                (csv_score(csv_path, f"yolov8n_tensorrt_ep_{precision}.csv"), csv_path, rows)
            )

    detected: dict[str, tuple[Path, list[dict[str, str]]]] = {}
    for key, matches in candidates.items():
        if matches:
            _, csv_path, rows = max(matches, key=lambda match: match[0])
            detected[key] = (csv_path, rows)
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


def first_value(rows: list[dict[str, str]], column: str) -> str:
    for row in rows:
        value = row.get(column, "")
        if value not in ("", "None"):
            return value
    return "n/a"


def backend_label(key: str) -> str:
    labels = {
        "onnxruntime_fp32": "ONNX Runtime CUDA FP32",
        "tensorrt_ep_fp32": "TensorRT EP FP32",
        "tensorrt_ep_fp16": "TensorRT EP FP16",
    }
    return labels[key]


def all_rows(detected: dict[str, tuple[Path, list[dict[str, str]]]]) -> list[dict[str, str]]:
    return [row for _, rows in detected.values() for row in rows]


def system_metadata_table(detected: dict[str, tuple[Path, list[dict[str, str]]]]) -> str:
    rows = all_rows(detected)
    table_rows = [
        ["GPU", first_value(rows, "gpu_name")],
        ["Platform", first_value(rows, "platform")],
        ["Python", first_value(rows, "python_version")],
        ["ONNX Runtime", first_value(rows, "onnxruntime_version")],
    ]

    for key in ("onnxruntime_fp32", "tensorrt_ep_fp32", "tensorrt_ep_fp16"):
        if key in detected:
            path, result_rows = detected[key]
            table_rows.append([f"{backend_label(key)} source CSV", relative_repo_path(path)])
            table_rows.append([f"{backend_label(key)} timestamp", first_value(result_rows, "timestamp")])

    return markdown_table(["Field", "Value"], table_rows)


def benchmark_config_table(detected: dict[str, tuple[Path, list[dict[str, str]]]]) -> str:
    rows = all_rows(detected)
    if not rows:
        return markdown_table(
            ["Field", "Value"],
            [
                ["Model", "yolov8n"],
                ["Backends", "n/a"],
                ["Input shape", "[batch_size, 3, image_size, image_size]"],
                ["Image size", "n/a"],
                ["Batch sizes", "n/a"],
                ["Warmup iterations", "n/a"],
                ["Measurement iterations", "n/a"],
            ],
        )

    batch_sizes = sorted({as_int(row, "batch_size") for row in rows})
    image_sizes = sorted({as_int(row, "image_size") for row in rows})
    return markdown_table(
        ["Field", "Value"],
        [
            ["Model", "yolov8n"],
            ["Backends", ", ".join(backend_label(key) for key in detected)],
            ["Input shape", "[batch_size, 3, image_size, image_size]"],
            ["Image size", ", ".join(str(size) for size in image_sizes)],
            ["Batch sizes", ", ".join(str(batch_size) for batch_size in batch_sizes)],
            ["Warmup iterations", first_value(rows, "warmup_iters")],
            ["Measurement iterations", first_value(rows, "measurement_iters")],
        ],
    )


def result_table(rows: list[dict[str, str]]) -> str:
    table_rows = []
    for row in sorted(rows, key=lambda item: as_int(item, "batch_size")):
        table_rows.append(
            [
                as_int(row, "batch_size"),
                as_int(row, "image_size"),
                fmt_num(as_float(row, "mean_latency_ms")),
                fmt_num(as_float(row, "p50_latency_ms")),
                fmt_num(as_float(row, "p90_latency_ms")),
                fmt_num(as_float(row, "p95_latency_ms")),
                fmt_num(as_float(row, "p99_latency_ms")),
                fmt_num(as_float(row, "min_latency_ms")),
                fmt_num(as_float(row, "max_latency_ms")),
                fmt_num(as_float(row, "throughput_images_per_sec")),
                first_value([row], "provider"),
            ]
        )

    return markdown_table(
        [
            "Batch",
            "Image size",
            "Mean ms",
            "P50 ms",
            "P90 ms",
            "P95 ms",
            "P99 ms",
            "Min ms",
            "Max ms",
            "Throughput images/s",
            "Provider",
        ],
        table_rows,
    )


def comparison_table(detected: dict[str, tuple[Path, list[dict[str, str]]]]) -> str:
    if not detected:
        return "No YOLOv8n benchmark CSV files were found in `results/`."

    series = {key: rows_by_batch(rows) for key, (_, rows) in detected.items()}
    batch_sizes = sorted(set.intersection(*(set(rows) for rows in series.values())))
    if not batch_sizes:
        return "Detected YOLOv8n CSV files do not share comparable batch sizes."

    keys = [key for key in ("onnxruntime_fp32", "tensorrt_ep_fp32", "tensorrt_ep_fp16") if key in detected]
    baseline_key = keys[0]
    headers = ["Batch"]
    for key in keys:
        label = backend_label(key)
        headers.extend([f"{label} P50 ms", f"{label} images/s"])
        if key != baseline_key:
            headers.extend([f"{label} latency delta", f"{label} throughput delta"])

    table_rows: list[list[Any]] = []
    for batch_size in batch_sizes:
        baseline = series[baseline_key][batch_size]
        baseline_p50 = as_float(baseline, "p50_latency_ms")
        baseline_throughput = as_float(baseline, "throughput_images_per_sec")
        row: list[Any] = [batch_size]

        for key in keys:
            current = series[key][batch_size]
            p50 = as_float(current, "p50_latency_ms")
            throughput = as_float(current, "throughput_images_per_sec")
            row.extend([fmt_num(p50), fmt_num(throughput)])
            if key != baseline_key:
                row.extend(
                    [
                        fmt_pct((p50 - baseline_p50) / baseline_p50 * 100.0),
                        fmt_pct((throughput - baseline_throughput) / baseline_throughput * 100.0),
                    ]
                )
        table_rows.append(row)

    return markdown_table(headers, table_rows)


def build_report(detected: dict[str, tuple[Path, list[dict[str, str]]]]) -> str:
    sections = [
        "# YOLOv8n Inference Benchmark Report",
        "",
        "This report summarizes locally generated YOLOv8n object detection "
        "inference benchmark CSV files from `results/`. Benchmark numbers "
        "should come from actual runs only and should not be manually invented.",
        "",
        "Random image tensors are used only for backend performance measurement. "
        "They are not an accuracy, mAP, or detection-quality evaluation.",
        "",
        "## System Metadata",
        "",
        system_metadata_table(detected),
        "",
        "## Benchmark Configuration",
        "",
        benchmark_config_table(detected),
        "",
    ]

    for key in ("onnxruntime_fp32", "tensorrt_ep_fp32", "tensorrt_ep_fp16"):
        if key in detected:
            _, rows = detected[key]
            sections.extend([f"## {backend_label(key)} Results", "", result_table(rows), ""])
        else:
            sections.extend(
                [
                    f"## {backend_label(key)} Results",
                    "",
                    f"`results/yolov8n_{key.replace('_fp32', '').replace('_fp16', '')}*.csv` was not found.",
                    "",
                ]
            )

    sections.extend(
        [
            "## Backend Comparison",
            "",
            comparison_table(detected),
            "",
            "## Latency vs Throughput",
            "",
            "Object detection inference latency measures the time to run the "
            "model forward pass for one image batch. Throughput measures images "
            "processed per second. Larger batches can improve throughput while "
            "also increasing per-batch latency, so real-time applications often "
            "need to evaluate both.",
            "",
            "## Reproducibility Note",
            "",
            "These tables are generated from local CSV files in `results/`. Do "
            "not edit benchmark numbers by hand; rerun the benchmark scripts "
            "when the environment, model export, driver, or dependencies change.",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> int:
    args = parse_args()
    results_dir = resolve_repo_path(args.results_dir)
    output_path = resolve_repo_path(args.output)

    detected = detect_yolo_csvs(results_dir)
    report = build_report(detected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print(f"Wrote report: {output_path}")
    if not detected:
        print("Note: no YOLOv8n benchmark CSV files were found yet.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
