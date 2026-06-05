# CUDA Monte Carlo Option Pricing

This module benchmarks European call and put option pricing with a CPU Monte
Carlo baseline and a CUDA implementation. It is intended as a clear,
resume-grade finance and GPU performance project for CUDA/NVIDIA FSI roles.

No benchmark numbers are included here unless they were generated locally.
Runtime, throughput, and speedup depend on the GPU, CPU, CUDA version, compiler,
and current system load.

## Model

For each simulated path, the terminal stock price is sampled from geometric
Brownian motion:

```text
S_T = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
```

where `Z` is a standard normal random variable.

The payoff is:

```text
call = max(S_T - K, 0)
put  = max(K - S_T, 0)
```

The discounted Monte Carlo price estimate is:

```text
price = exp(-r * T) * average_payoff
```

## CPU vs CUDA Benchmark

The CPU baseline uses C++17, `std::mt19937`, and
`std::normal_distribution<double>`. It times the full CPU simulation with
`std::chrono`.

The CUDA path uses a kernel where each thread simulates one or more paths with
cuRAND normal random numbers, computes payoffs, and reduces each block to a
partial sum. The host copies the block sums back, finishes the small final sum,
and computes the discounted option price. CUDA events time the GPU kernel work.

The CPU and CUDA random streams are independent, so their estimated option
prices should be close for large path counts but not bit-identical.

## Build

From the repository root:

```bash
cmake -S cuda_finance/monte_carlo_option_pricing \
  -B cuda_finance/monte_carlo_option_pricing/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build cuda_finance/monte_carlo_option_pricing/build --config Release
```

From the module root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The CMake file defaults to CUDA architecture `86`, which matches the RTX 3060
Laptop GPU. You can override this when configuring, for example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
```

## Run

From the repository root on Linux or WSL:

```bash
./cuda_finance/monte_carlo_option_pricing/build/monte_carlo_option_pricing \
  --paths 1000000 \
  --S0 100 \
  --K 100 \
  --r 0.05 \
  --sigma 0.2 \
  --T 1.0 \
  --type call \
  --seed 1234 \
  --output cuda_finance/monte_carlo_option_pricing/results/monte_carlo_benchmark.csv
```

On Windows with a multi-config generator, the executable may be under
`build/Release/monte_carlo_option_pricing.exe`.

The program prints a table with this shape:

```text
backend   option_price   runtime_ms   paths/sec   speedup
CPU       local value     local value  local value 1.000x
CUDA      local value     local value  local value local value
```

It also appends CSV rows to the output path.

## Benchmark Script

From the repository root:

```bash
bash cuda_finance/monte_carlo_option_pricing/scripts/run_benchmarks.sh
```

From the module root:

```bash
bash scripts/run_benchmarks.sh
```

The script builds the executable, runs these path counts, and writes one CSV:

```text
100000
1000000
5000000
10000000
```

Then it generates:

```text
reports/runtime_vs_paths.png
reports/speedup_vs_paths.png
```

Generated CSV and PNG files are ignored by Git. The script uses
`.venv/bin/python` from the repository root when available, then falls back to
`python3`.

## Plot Only

```bash
.venv/bin/python cuda_finance/monte_carlo_option_pricing/scripts/plot_results.py \
  --input cuda_finance/monte_carlo_option_pricing/results/monte_carlo_benchmark.csv \
  --reports-dir cuda_finance/monte_carlo_option_pricing/reports
```

The plotting script uses matplotlib only. If your system Python does not have
matplotlib, use the repository virtual environment as shown above, or run the
same command with `python3` after installing the project requirements.

## Summarize Results

Generate a Markdown report from the local CSV:

```bash
.venv/bin/python cuda_finance/monte_carlo_option_pricing/scripts/summarize_results.py
```

This writes:

```text
cuda_finance/monte_carlo_option_pricing/reports/monte_carlo_benchmark.md
```

The Markdown report may be committed after it is generated from a real local
benchmark CSV. Do not manually edit benchmark numbers.

## Results

Populate this section only after running the benchmark locally. Do not invent
or paste placeholder benchmark numbers.

| Paths | Option | CPU runtime ms | CUDA runtime ms | Speedup |
| --- | --- | --- | --- | --- |
| locally generated | locally generated | locally generated | locally generated | locally generated |

Small path counts can be slower on CUDA because kernel launch, cuRAND state
setup, and host/device coordination overhead can dominate short runs. Larger
path counts provide enough parallel work to amortize that overhead.

## Troubleshooting

### `python: command not found`

Use `python3` or the repository virtual environment:

```bash
.venv/bin/python cuda_finance/monte_carlo_option_pricing/scripts/plot_results.py
```

### `ModuleNotFoundError: matplotlib`

Install the repository Python dependencies or use the existing virtual
environment if it has already been set up:

```bash
pip install -r requirements.txt
.venv/bin/python cuda_finance/monte_carlo_option_pricing/scripts/plot_results.py
```

### `nvcc not found`

Install the CUDA toolkit inside WSL or ensure `nvcc` is on `PATH`:

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

Then verify:

```bash
bash cuda_finance/monte_carlo_option_pricing/scripts/check_cuda_toolchain.sh
```

### CUDA is slower for small path counts

This is expected for small workloads. The GPU has fixed launch and setup costs,
so the CPU can win at 100K or 1M paths. CUDA becomes more useful as the path
count grows and the work is large enough to keep the GPU busy.

## Files

```text
include/option_pricing.hpp   Shared option types and pricing function APIs
src/monte_carlo_cpu.cpp      CPU Monte Carlo baseline
src/monte_carlo_cuda.cu      CUDA/cuRAND Monte Carlo implementation
src/main.cpp                 CLI, table output, and CSV writing
scripts/run_benchmarks.sh    Build and benchmark path counts
scripts/plot_results.py      Generate runtime and speedup plots
scripts/summarize_results.py Generate a Markdown report from CSV results
results/.gitkeep             Keeps the generated-results directory present
reports/.gitkeep             Keeps the generated-reports directory present
```
