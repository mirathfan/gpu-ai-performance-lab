# ResNet18 Inference Benchmark Report

This report summarizes locally generated ResNet18 inference benchmark CSV files from `results/`. Benchmark numbers should come from actual runs only and should not be manually invented.

TensorRT results in this report use ONNX Runtime `TensorrtExecutionProvider`; raw TensorRT engine API benchmarks are not added yet.

## System and Hardware Metadata

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| GPU VRAM GB | 5.99951171875 |
| Platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| Python | 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0] |
| PyTorch | 2.5.1+cu121 |
| ONNX Runtime | 1.26.0 |
| CUDA available | True |
| CUDA version | 12.1 |
| PyTorch FP32 source CSV | results/resnet18_pytorch_fp32.csv |
| PyTorch FP32 timestamp | 2026-06-04T06:35:42.246693+00:00 |
| PyTorch FP16 source CSV | results/resnet18_pytorch_fp16.csv |
| PyTorch FP16 timestamp | 2026-06-04T06:37:11.312906+00:00 |
| ONNX Runtime FP32 source CSV | results/resnet18_onnxruntime.csv |
| ONNX Runtime FP32 timestamp | 2026-06-04T07:30:18.945875+00:00 |
| TensorRT EP FP32 source CSV | results/resnet18_tensorrt_ep_fp32.csv |
| TensorRT EP FP32 timestamp | 2026-06-04T08:01:00.280831+00:00 |
| TensorRT EP FP16 source CSV | results/resnet18_tensorrt_ep_fp16.csv |
| TensorRT EP FP16 timestamp | 2026-06-04T08:02:10.896571+00:00 |

## Benchmark Configuration

| Field | Value |
| --- | --- |
| Model | resnet18 |
| Weights | ResNet18_Weights.IMAGENET1K_V1 |
| Backends | PyTorch FP32, PyTorch FP16, ONNX Runtime FP32, TensorRT EP FP32, TensorRT EP FP16 |
| Input shape | [batch_size, 3, 224, 224] |
| Batch sizes | 1, 4, 8, 16 |
| Precision modes | fp16, fp32 |
| Device | cuda |
| Warmup iterations | 20 |
| Measurement iterations | 100 |

## PyTorch FP32 Results

| Batch | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput samples/s | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.971 | 3.397 | 4.576 | 9.046 | 10.367 | 3.212 | 11.638 | 251.820 | 63.089 |
| 4 | 3.716 | 3.628 | 3.873 | 3.926 | 4.075 | 3.519 | 5.403 | 1076.372 | 80.190 |
| 8 | 6.407 | 5.969 | 7.094 | 9.369 | 9.496 | 5.543 | 13.004 | 1248.586 | 106.409 |
| 16 | 11.111 | 11.067 | 11.295 | 11.404 | 11.592 | 10.846 | 11.797 | 1440.036 | 162.003 |

## PyTorch FP16 Results

| Batch | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput samples/s | Peak memory MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.552 | 4.531 | 4.786 | 4.938 | 5.159 | 4.216 | 5.701 | 219.688 | 37.334 |
| 4 | 4.403 | 4.373 | 4.702 | 4.975 | 5.593 | 3.876 | 6.568 | 908.431 | 47.118 |
| 8 | 4.187 | 4.094 | 4.552 | 4.607 | 5.030 | 3.943 | 5.712 | 1910.554 | 61.196 |
| 16 | 7.476 | 7.355 | 8.283 | 8.634 | 9.118 | 6.797 | 10.227 | 2140.321 | 91.055 |

## ONNX Runtime FP32 Results

| Batch | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput samples/s | Provider |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.115 | 2.316 | 6.510 | 7.000 | 7.420 | 2.122 | 8.286 | 320.986 | CUDAExecutionProvider |
| 4 | 4.317 | 4.223 | 4.461 | 5.131 | 5.490 | 4.058 | 5.635 | 926.603 | CUDAExecutionProvider |
| 8 | 7.047 | 6.957 | 7.332 | 7.693 | 8.229 | 6.794 | 8.429 | 1135.189 | CUDAExecutionProvider |
| 16 | 12.567 | 12.507 | 12.830 | 12.994 | 13.142 | 12.268 | 13.253 | 1273.175 | CUDAExecutionProvider |

## TensorRT EP FP32 Results

| Batch | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput samples/s | Provider |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.861 | 4.815 | 5.468 | 5.653 | 6.100 | 1.380 | 6.611 | 258.976 | TensorrtExecutionProvider |
| 4 | 4.143 | 4.028 | 4.740 | 5.085 | 5.464 | 3.725 | 6.061 | 965.403 | TensorrtExecutionProvider |
| 8 | 5.674 | 5.602 | 5.980 | 6.088 | 6.165 | 5.373 | 6.198 | 1409.922 | TensorrtExecutionProvider |
| 16 | 9.173 | 9.142 | 9.432 | 9.533 | 9.657 | 8.788 | 9.814 | 1744.329 | TensorrtExecutionProvider |

## TensorRT EP FP16 Results

| Batch | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput samples/s | Provider |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.877 | 0.860 | 0.949 | 0.975 | 1.053 | 0.808 | 1.468 | 1140.229 | TensorrtExecutionProvider |
| 4 | 1.667 | 1.504 | 2.172 | 2.215 | 2.320 | 1.435 | 2.327 | 2399.911 | TensorrtExecutionProvider |
| 8 | 2.617 | 2.588 | 2.750 | 2.794 | 2.978 | 2.486 | 3.409 | 3056.456 | TensorrtExecutionProvider |
| 16 | 4.797 | 4.768 | 4.899 | 4.958 | 5.109 | 4.670 | 5.233 | 3335.212 | TensorrtExecutionProvider |

## Backend Comparison

| Batch | PyTorch FP32 P50 ms | PyTorch FP16 P50 ms | FP16 latency delta | PyTorch FP32 samples/s | PyTorch FP16 samples/s | FP16 throughput delta | ONNX Runtime FP32 P50 ms | ONNX latency delta | ONNX Runtime samples/s | ONNX throughput delta | TensorRT EP FP32 P50 ms | TensorRT EP FP32 latency delta | TensorRT EP FP32 samples/s | TensorRT EP FP32 throughput delta | TensorRT EP FP16 P50 ms | TensorRT EP FP16 latency delta | TensorRT EP FP16 samples/s | TensorRT EP FP16 throughput delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.397 | 4.531 | +33.4% | 251.820 | 219.688 | -12.8% | 2.316 | -31.8% | 320.986 | +27.5% | 4.815 | +41.7% | 258.976 | +2.8% | 0.860 | -74.7% | 1140.229 | +352.8% |
| 4 | 3.628 | 4.373 | +20.5% | 1076.372 | 908.431 | -15.6% | 4.223 | +16.4% | 926.603 | -13.9% | 4.028 | +11.0% | 965.403 | -10.3% | 1.504 | -58.6% | 2399.911 | +123.0% |
| 8 | 5.969 | 4.094 | -31.4% | 1248.586 | 1910.554 | +53.0% | 6.957 | +16.6% | 1135.189 | -9.1% | 5.602 | -6.1% | 1409.922 | +12.9% | 2.588 | -56.6% | 3056.456 | +144.8% |
| 16 | 11.067 | 7.355 | -33.5% | 1440.036 | 2140.321 | +48.6% | 12.507 | +13.0% | 1273.175 | -11.6% | 9.142 | -17.4% | 1744.329 | +21.1% | 4.768 | -56.9% | 3335.212 | +131.6% |

## TensorRT EP Notes

TensorRT EP uses TensorRT through ONNX Runtime provider selection. The first run for a model shape and precision may build and cache a TensorRT engine, so initial runs can include build overhead. These results can differ from raw TensorRT API benchmarks because ONNX Runtime still owns session setup, graph partitioning, and provider fallback behavior. TensorRT is expected to help most when graph optimizations, precision lowering, and kernel selection improve execution for the target GPU.

## Latency vs Throughput

Latency measures how long one inference batch takes to complete. Throughput measures how many input samples are processed per second. A larger batch can increase throughput even when per-batch latency also increases, so both metrics are useful when choosing a deployment batch size.

## FP16 Memory Observation

In these local CSVs, FP16 peak GPU memory is lower than FP32 for every matched batch size. The average FP16 minus FP32 memory delta is -43.747 MB (-42.5%).

ONNX Runtime peak GPU memory is not reported by the current benchmark script because ONNX Runtime does not expose a simple peak CUDA allocation counter through this beginner-readable path.

## Reproducibility Note

These tables are generated from local CSV files in `results/`. Do not edit benchmark numbers by hand; rerun the benchmark scripts when the environment, code, driver, or dependencies change.
