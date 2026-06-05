# CUDA Monte Carlo Option Pricing Benchmark

## Benchmark Setup

These results were generated locally from:

- CSV input: `cuda_finance/monte_carlo_option_pricing/results/monte_carlo_benchmark.csv`
- Option type: `call`
- Initial stock price S0: `100.0000000000`
- Strike price K: `100.0000000000`
- Risk-free rate r: `0.0500000000`
- Volatility sigma: `0.2000000000`
- Time to maturity T: `1.0000000000`
- Seed: `1234`
- Path counts: `100,000, 1,000,000, 5,000,000, 10,000,000`

## Option Pricing Formula

Each path samples the terminal stock price:

```text
S_T = S0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
```

The payoff is `max(S_T - K, 0)` for calls and `max(K - S_T, 0)` for puts.
The reported price is the discounted average payoff:

```text
price = exp(-r * T) * average_payoff
```

## CPU and CUDA Results

| Paths | Backend | Option price | Runtime ms | Paths/sec | Speedup vs CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| 100,000 | CPU | 10.385460 | 4.140 | 24156386.51 | 1.000x |
| 100,000 | CUDA | 10.495197 | 57.030 | 1753474.39 | 0.073x |
| 1,000,000 | CPU | 10.445642 | 41.049 | 24360874.19 | 1.000x |
| 1,000,000 | CUDA | 10.441791 | 50.312 | 19875898.46 | 0.816x |
| 5,000,000 | CPU | 10.440085 | 202.363 | 24708014.15 | 1.000x |
| 5,000,000 | CUDA | 10.438460 | 51.737 | 96643426.74 | 3.911x |
| 10,000,000 | CPU | 10.445455 | 412.722 | 24229376.09 | 1.000x |
| 10,000,000 | CUDA | 10.443979 | 56.898 | 175754531.51 | 7.254x |

## CUDA Speedup

| Paths | CUDA runtime ms | CPU runtime ms | CUDA speedup vs CPU |
| --- | ---: | ---: | ---: |
| 100,000 | 57.030 | 4.140 | 0.073x |
| 1,000,000 | 50.312 | 41.049 | 0.816x |
| 5,000,000 | 51.737 | 202.363 | 3.911x |
| 10,000,000 | 56.898 | 412.722 | 7.254x |

## Best Observed Speedup

The best observed CUDA speedup was **7.254x** at **10,000,000 paths**.

## Interpretation

CUDA was slower than the CPU baseline at smaller workloads (100,000, 1,000,000 paths in this run). Kernel launch, random-state setup, and host/device coordination overhead dominate when there are not enough Monte Carlo paths to keep the GPU busy.

At larger path counts, GPU throughput improves because the fixed CUDA overhead is amortized across more simulated paths and more parallel work is available.

## Reproducibility Note

These results are generated locally from the CSV output and should not be edited manually. Re-run the benchmark and regenerate this report when hardware, drivers, compiler versions, or benchmark settings change.
