#!/usr/bin/env python3
"""Plot CUDA Monte Carlo benchmark results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    module_dir = script_dir.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=module_dir / "results" / "monte_carlo_benchmark.csv",
        type=Path,
        help="CSV file produced by monte_carlo_option_pricing",
    )
    parser.add_argument(
        "--reports-dir",
        default=module_dir / "reports",
        type=Path,
        help="Directory where PNG plots will be written",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"CSV file has no benchmark rows: {csv_path}")

    return rows


def group_xy(
    rows: list[dict[str, str]],
    y_column: str,
    include_cpu: bool = True,
) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        backend = row["backend"]
        if not include_cpu and backend.upper() == "CPU":
            continue
        grouped[backend].append((int(row["paths"]), float(row[y_column])))

    for backend_rows in grouped.values():
        backend_rows.sort(key=lambda item: item[0])

    return grouped


def plot_lines(
    grouped: dict[str, list[tuple[int, float]]],
    title: str,
    ylabel: str,
    output_path: Path,
    y_log: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))

    for backend, points in sorted(grouped.items()):
        paths = [point[0] for point in points]
        values = [point[1] for point in points]
        plt.plot(paths, values, marker="o", linewidth=2, label=backend)

    plt.title(title)
    plt.xlabel("Monte Carlo paths")
    plt.ylabel(ylabel)
    plt.xscale("log")
    if y_log:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    runtime_data = group_xy(rows, "runtime_ms")
    plot_lines(
        runtime_data,
        "Runtime vs Monte Carlo Paths",
        "Runtime (ms)",
        args.reports_dir / "runtime_vs_paths.png",
        y_log=True,
    )

    speedup_data = group_xy(rows, "speedup")
    plot_lines(
        speedup_data,
        "Speedup vs Monte Carlo Paths",
        "Speedup vs CPU",
        args.reports_dir / "speedup_vs_paths.png",
    )


if __name__ == "__main__":
    main()
