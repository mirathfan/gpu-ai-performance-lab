# ResNet18 PyTorch Baseline Report

This report summarizes locally generated PyTorch ResNet18 inference benchmark CSV files from `results/`. Benchmark numbers should come from actual runs only and should not be manually invented.

## System and Hardware Metadata

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| GPU VRAM GB | 5.99951171875 |
| Platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| Python | 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0] |
| PyTorch | 2.5.1+cu121 |
| CUDA available | True |
| CUDA version | 12.1 |
| FP32 source CSV | results/resnet18_pytorch_fp32.csv |
| FP16 source CSV | results/resnet18_pytorch_fp16.csv |
| FP32 timestamp | 2026-06-04T06:35:42.246693+00:00 |
| FP16 timestamp | 2026-06-04T06:37:11.312906+00:00 |

## Benchmark Configuration

| Field | Value |
| --- | --- |
| Model | resnet18 |
| Weights | ResNet18_Weights.IMAGENET1K_V1 |
| Framework | PyTorch |
| Input shape | [batch_size, 3, 224, 224] |
| Batch sizes | 1, 4, 8, 16 |
| Precision modes | fp16, fp32 |
| Device | cuda |
| Warmup iterations | 20 |
| Measurement iterations | 100 |

## FP32 Results

| Batch | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput samples/s | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.971 | 3.397 | 4.576 | 9.046 | 10.367 | 3.212 | 11.638 | 251.820 | 63.089 |
| 4 | 3.716 | 3.628 | 3.873 | 3.926 | 4.075 | 3.519 | 5.403 | 1076.372 | 80.190 |
| 8 | 6.407 | 5.969 | 7.094 | 9.369 | 9.496 | 5.543 | 13.004 | 1248.586 | 106.409 |
| 16 | 11.111 | 11.067 | 11.295 | 11.404 | 11.592 | 10.846 | 11.797 | 1440.036 | 162.003 |

## FP16 Results

| Batch | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput samples/s | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.552 | 4.531 | 4.786 | 4.938 | 5.159 | 4.216 | 5.701 | 219.688 | 37.334 |
| 4 | 4.403 | 4.373 | 4.702 | 4.975 | 5.593 | 3.876 | 6.568 | 908.431 | 47.118 |
| 8 | 4.187 | 4.094 | 4.552 | 4.607 | 5.030 | 3.943 | 5.712 | 1910.554 | 61.196 |
| 16 | 7.476 | 7.355 | 8.283 | 8.634 | 9.118 | 6.797 | 10.227 | 2140.321 | 91.055 |

## FP32 vs FP16 Comparison

| Batch | FP32 P50 ms | FP16 P50 ms | FP16 latency delta | FP32 samples/s | FP16 samples/s | FP16 throughput delta | FP32 memory MB | FP16 memory MB | Memory delta MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.397 | 4.531 | +33.4% | 251.820 | 219.688 | -12.8% | 63.089 | 37.334 | -25.755 |
| 4 | 3.628 | 4.373 | +20.5% | 1076.372 | 908.431 | -15.6% | 80.190 | 47.118 | -33.073 |
| 8 | 5.969 | 4.094 | -31.4% | 1248.586 | 1910.554 | +53.0% | 106.409 | 61.196 | -45.213 |
| 16 | 11.067 | 7.355 | -33.5% | 1440.036 | 2140.321 | +48.6% | 162.003 | 91.055 | -70.948 |

## Latency vs Throughput

Latency measures how long one inference batch takes to complete. Throughput measures how many input samples are processed per second. A larger batch can increase throughput even when per-batch latency also increases, so both metrics are useful when choosing a deployment batch size.

## FP16 Memory Observation

In these local CSVs, FP16 peak GPU memory is lower than FP32 for every matched batch size. The average FP16 minus FP32 memory delta is -43.747 MB (-42.5%).

## Reproducibility Note

These tables are generated from local CSV files in `results/`. Do not edit benchmark numbers by hand; rerun the benchmark scripts when the environment, code, driver, or dependencies change.
