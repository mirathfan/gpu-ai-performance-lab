# GPU AI Performance Lab

GPU AI Performance Lab is a hands-on project for deep learning inference
optimization and GPU benchmarking. The goal is to build a resume-grade,
reproducible lab for PyTorch, ONNX Runtime, TensorRT, CUDA, and LLM performance
work without inventing results.

The current first milestone is a PyTorch ResNet18 baseline benchmark. It checks
that GPU execution works, runs inference only, measures latency with warmup and
CUDA synchronization, and writes locally generated results to CSV.

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

Run the first PyTorch ResNet18 benchmark:

```bash
python scripts/benchmark_resnet18_pytorch.py \
  --batch-sizes 1 4 8 16 32 \
  --warmup 20 \
  --iters 100 \
  --precision fp32 \
  --device cuda \
  --output results/resnet18_pytorch.csv
```

For a CPU-only smoke test, use `--device cpu`. FP16 is intentionally limited to
CUDA in this first benchmark script.

## Results

Benchmark numbers should be generated locally and not manually invented. Keep
raw CSV outputs under `results/`; generated CSV and JSON files are ignored by
Git so local hardware measurements do not get committed by accident.

Placeholder for locally generated results:

| Benchmark | Device | Precision | Output |
| --- | --- | --- | --- |
| ResNet18 PyTorch baseline | local | fp32/fp16 | `results/resnet18_pytorch.csv` |

## Next Milestones

- ONNX export and ONNX Runtime benchmark
- TensorRT FP32/FP16 benchmark
- CUDA Monte Carlo simulation
- CUDA deep learning kernels
- LLM inference benchmark
