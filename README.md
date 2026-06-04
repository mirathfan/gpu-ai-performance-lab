# GPU AI Performance Lab

GPU AI Performance Lab is a hands-on project for deep learning inference
optimization and GPU benchmarking. The goal is to build a resume-grade,
reproducible lab for PyTorch, ONNX Runtime, TensorRT, CUDA, and LLM performance
work without inventing results.

The first milestones build a ResNet18 inference benchmark path across PyTorch
and ONNX Runtime. The project checks that GPU execution works, runs inference
only, measures latency with warmup and CUDA synchronization, writes locally
generated results to CSV, and generates a small Markdown report plus simple
plots from those CSV files.

## Results Summary

Local benchmark environment: WSL2 Ubuntu 24.04, Python 3.12.3, PyTorch
2.5.1+cu121, ONNX Runtime 1.26.0, NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB
VRAM.

| Backend | Precision | Provider | Batch-1 P50 latency ms | Batch-16 throughput samples/s | Batch-16 throughput vs PyTorch FP32 |
| --- | --- | --- | --- | --- | --- |
| PyTorch | FP32 | CUDA | 3.397 | 1440.036 | baseline |
| PyTorch | FP16 | CUDA | 4.531 | 2140.321 | +48.6% |
| ONNX Runtime | FP32 | CUDAExecutionProvider | 2.316 | 1273.175 | -11.6% |
| TensorRT EP | FP32 | TensorrtExecutionProvider | 4.815 | 1744.329 | +21.1% |
| **TensorRT EP** | **FP16** | **TensorrtExecutionProvider** | **0.860** | **3335.212** | **+131.6%** |

Best observed results from the generated report:

- TensorRT EP FP16 reached **0.860 ms** p50 latency at batch size 1.
- TensorRT EP FP16 reached **3335.212 samples/sec** at batch size 16.
- TensorRT EP FP16 improved batch-16 throughput by **+131.6%** versus PyTorch FP32.

## Architecture Pipeline

```text
PyTorch model -> ONNX export -> ONNX Runtime CUDA -> ONNX Runtime TensorRT Execution Provider
```

## What This Demonstrates

- GPU inference benchmarking
- ONNX export and validation
- TensorRT acceleration
- FP32 vs FP16 tradeoffs
- Latency/throughput analysis
- Reproducible local benchmarking on RTX 3060 Laptop GPU

## Setup: WSL2 Ubuntu

From inside WSL2 Ubuntu, clone or open this repository and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify that PyTorch can see and use the GPU:

```bash
python scripts/check_gpu.py
```

## Milestone 1: PyTorch ResNet18 Baseline

Run the FP32 benchmark:

```bash
python scripts/benchmark_resnet18_pytorch.py \
  --batch-sizes 1 4 8 16 32 \
  --warmup 20 \
  --iters 100 \
  --precision fp32 \
  --device cuda \
  --output results/resnet18_pytorch_fp32.csv
```

Run the FP16 benchmark:

```bash
python scripts/benchmark_resnet18_pytorch.py \
  --batch-sizes 1 4 8 16 32 \
  --warmup 20 \
  --iters 100 \
  --precision fp16 \
  --device cuda \
  --output results/resnet18_pytorch_fp16.csv
```

Generate the Markdown summary report:

```bash
python scripts/summarize_results.py
```

Generate plots:

```bash
python scripts/plot_results.py
```

For a CPU-only smoke test, use `--device cpu`. FP16 is intentionally limited to
CUDA in this first benchmark script.

## Milestone 2: ONNX Runtime ResNet18 Baseline

Install or refresh dependencies:

```bash
pip install -r requirements.txt
```

Export ResNet18 to ONNX with a dynamic batch dimension:

```bash
python scripts/export_resnet18_onnx.py \
  --batch-size 1 \
  --opset 17 \
  --output models/resnet18.onnx \
  --dynamic-batch
```

Validate ONNX Runtime output against PyTorch:

```bash
python scripts/validate_resnet18_onnx.py \
  --onnx models/resnet18.onnx \
  --batch-size 1 \
  --device cuda \
  --atol 1e-2 \
  --rtol 1e-3
```

The validation script prints the actual max and mean absolute differences, the
configured tolerances, and the top-1 class from both PyTorch and ONNX Runtime.
Small CUDA numerical differences are expected; validation passes when
`numpy.allclose` passes, or when top-1 matches and the max absolute difference
is still small.

Benchmark ONNX Runtime FP32:

```bash
python scripts/benchmark_resnet18_onnxruntime.py \
  --onnx models/resnet18.onnx \
  --batch-sizes 1 4 8 16 \
  --warmup 20 \
  --iters 100 \
  --device cuda \
  --output results/resnet18_onnxruntime.csv
```

Regenerate the combined Markdown report:

```bash
python scripts/summarize_results.py
```

Regenerate plots:

```bash
python scripts/plot_results.py
```

TensorRT is not part of Milestone 2.

## Milestone 3: TensorRT Execution Provider Baseline

Confirm that ONNX Runtime can see the TensorRT execution provider:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The expected provider list should include `TensorrtExecutionProvider`,
`CUDAExecutionProvider`, and `CPUExecutionProvider`.

Run the TensorRT EP FP32 benchmark:

```bash
python scripts/benchmark_resnet18_tensorrt_ep.py \
  --onnx models/resnet18.onnx \
  --batch-sizes 1 4 8 16 \
  --warmup 20 \
  --iters 100 \
  --device cuda \
  --output results/resnet18_tensorrt_ep_fp32.csv
```

Run the TensorRT EP FP16 benchmark:

```bash
python scripts/benchmark_resnet18_tensorrt_ep.py \
  --onnx models/resnet18.onnx \
  --batch-sizes 1 4 8 16 \
  --warmup 20 \
  --iters 100 \
  --device cuda \
  --fp16 \
  --output results/resnet18_tensorrt_ep_fp16.csv
```

TensorRT EP uses TensorRT through ONNX Runtime. The first run for a model shape
and precision may build and cache a TensorRT engine under `.trt_cache/`, so the
first run can include engine build overhead. This milestone does not use the raw
TensorRT engine API yet.

Regenerate the combined Markdown report:

```bash
python scripts/summarize_results.py
```

Regenerate plots:

```bash
python scripts/plot_results.py
```

## Results

Benchmark numbers should be generated locally and not manually invented. Keep
raw CSV outputs under `results/`; generated CSV and JSON files are ignored by
Git so local hardware measurements do not get committed by accident.

Generated reports and figures under `reports/` may be committed because they
are presentation artifacts derived from local CSV files. The raw CSV files stay
ignored by Git. ONNX model files under `models/` are also ignored because they
are generated artifacts. TensorRT cache files under `.trt_cache/` are ignored
for the same reason.

Placeholder for locally generated results:

| Benchmark | Device | Precision | Output |
| --- | --- | --- | --- |
| ResNet18 PyTorch baseline | local | fp32 | `results/resnet18_pytorch_fp32.csv` |
| ResNet18 PyTorch baseline | local | fp16 | `results/resnet18_pytorch_fp16.csv` |
| ResNet18 ONNX Runtime baseline | local | fp32 | `results/resnet18_onnxruntime.csv` |
| ResNet18 TensorRT EP baseline | local | fp32 | `results/resnet18_tensorrt_ep_fp32.csv` |
| ResNet18 TensorRT EP baseline | local | fp16 | `results/resnet18_tensorrt_ep_fp16.csv` |

Generated report artifacts:

| Artifact | Path |
| --- | --- |
| Combined Markdown report | `reports/resnet18_inference_benchmark.md` |
| Latency plot | `reports/figures/resnet18_latency_vs_batch.png` |
| Throughput plot | `reports/figures/resnet18_throughput_vs_batch.png` |
| Memory plot | `reports/figures/resnet18_memory_vs_batch.png` |

## Next Milestones

- Raw TensorRT engine API benchmark
- CUDA Monte Carlo simulation
- CUDA deep learning kernels
- LLM inference benchmark
