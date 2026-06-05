#!/usr/bin/env python3
"""Generate a Markdown report for CUDA Monte Carlo benchmark results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


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
        "--output",
        default=module_dir / "reports" / "monte_carlo_benchmark.md",
        type=Path,
        help="Markdown report output path",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"CSV file has no benchmark rows: {csv_path}")

    required_columns = {
        "backend",
        "option_type",
        "paths",
        "S0",
        "K",
        "r",
        "sigma",
        "T",
        "seed",
        "option_price",
        "runtime_ms",
        "paths_per_sec",
        "speedup",
    }
    missing = required_columns - set(rows[0])
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_text}")

    return rows


def fmt_float(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_int(value: str | int) -> str:
    return f"{int(value):,}"


def grouped_by_paths(rows: list[dict[str, str]]) -> dict[int, dict[str, dict[str, str]]]:
    grouped: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[int(row["paths"])][row["backend"].upper()] = row
    return dict(sorted(grouped.items()))


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def generate_report(rows: list[dict[str, str]], csv_path: Path, repo_root: Path) -> str:
    grouped = grouped_by_paths(rows)
    first = rows[0]

    cuda_rows = [row for row in rows if row["backend"].upper() == "CUDA"]
    if not cuda_rows:
        raise ValueError("CSV does not contain CUDA rows")

    best_cuda = max(cuda_rows, key=lambda row: float(row["speedup"]))
    slow_cuda_rows = [row for row in cuda_rows if float(row["speedup"]) < 1.0]

    lines: list[str] = []
    lines.append("# CUDA Monte Carlo Option Pricing Benchmark")
    lines.append("")
    lines.append("## Benchmark Setup")
    lines.append("")
    lines.append("These results were generated locally from:")
    lines.append("")
    lines.append(f"- CSV input: `{display_path(csv_path, repo_root)}`")
    lines.append(f"- Option type: `{first['option_type']}`")
    lines.append(f"- Initial stock price S0: `{first['S0']}`")
    lines.append(f"- Strike price K: `{first['K']}`")
    lines.append(f"- Risk-free rate r: `{first['r']}`")
    lines.append(f"- Volatility sigma: `{first['sigma']}`")
    lines.append(f"- Time to maturity T: `{first['T']}`")
    lines.append(f"- Seed: `{first['seed']}`")
    lines.append(f"- Path counts: `{', '.join(fmt_int(paths) for paths in grouped)}`")
    lines.append("")
    lines.append("## Option Pricing Formula")
    lines.append("")
    lines.append("Each path samples the terminal stock price:")
    lines.append("")
    lines.append("```text")
    lines.append("S_T = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)")
    lines.append("```")
    lines.append("")
    lines.append("The payoff is `max(S_T - K, 0)` for calls and `max(K - S_T, 0)` for puts.")
    lines.append("The reported price is the discounted average payoff:")
    lines.append("")
    lines.append("```text")
    lines.append("price = exp(-r * T) * average_payoff")
    lines.append("```")
    lines.append("")
    lines.append("## CPU and CUDA Results")
    lines.append("")
    lines.append("| Paths | Backend | Option price | Runtime ms | Paths/sec | Speedup vs CPU |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for paths, backends in grouped.items():
        for backend in ("CPU", "CUDA"):
            row = backends.get(backend)
            if row is None:
                continue
            lines.append(
                "| "
                f"{fmt_int(paths)} | "
                f"{backend} | "
                f"{fmt_float(row['option_price'], 6)} | "
                f"{fmt_float(row['runtime_ms'], 3)} | "
                f"{fmt_float(row['paths_per_sec'], 2)} | "
                f"{fmt_float(row['speedup'], 3)}x |"
            )
    lines.append("")
    lines.append("## CUDA Speedup")
    lines.append("")
    lines.append("| Paths | CUDA runtime ms | CPU runtime ms | CUDA speedup vs CPU |")
    lines.append("| --- | ---: | ---: | ---: |")
    for paths, backends in grouped.items():
        cpu = backends.get("CPU")
        cuda = backends.get("CUDA")
        if cpu is None or cuda is None:
            continue
        lines.append(
            "| "
            f"{fmt_int(paths)} | "
            f"{fmt_float(cuda['runtime_ms'], 3)} | "
            f"{fmt_float(cpu['runtime_ms'], 3)} | "
            f"{fmt_float(cuda['speedup'], 3)}x |"
        )
    lines.append("")
    lines.append("## Best Observed Speedup")
    lines.append("")
    lines.append(
        f"The best observed CUDA speedup was **{fmt_float(best_cuda['speedup'], 3)}x** "
        f"at **{fmt_int(best_cuda['paths'])} paths**."
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if slow_cuda_rows:
        slow_paths = ", ".join(fmt_int(row["paths"]) for row in slow_cuda_rows)
        lines.append(
            "CUDA was slower than the CPU baseline at smaller workloads "
            f"({slow_paths} paths in this run). Kernel launch, random-state setup, "
            "and host/device coordination overhead dominate when there are not enough "
            "Monte Carlo paths to keep the GPU busy."
        )
        lines.append("")
    lines.append(
        "At larger path counts, GPU throughput improves because the fixed CUDA overhead "
        "is amortized across more simulated paths and more parallel work is available."
    )
    lines.append("")
    lines.append("## Reproducibility Note")
    lines.append("")
    lines.append(
        "These results are generated locally from the CSV output and should not be "
        "edited manually. Re-run the benchmark and regenerate this report when hardware, "
        "drivers, compiler versions, or benchmark settings change."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    rows = read_rows(args.input)
    report = generate_report(rows, args.input, repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
