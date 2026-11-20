# Published report tables

These CSV files are exact exports of the **rounded tables in the existing reports**. They contain no new measurements and are not raw latency samples. Original local benchmark CSVs were not available in this checkout.

Source commit: [`347add2`](https://github.com/mirathfan/gpu-ai-performance-lab/tree/347add221331d6dbc6c415dfb7978bf7dba2f9ba). [Manifest and SHA-256 hashes](manifest.json) identify each original report.

- [resnet18_pytorch_fp32_results.csv](resnet18_pytorch_fp32_results.csv): 4 rows from “PyTorch FP32 Results” in `reports/resnet18_inference_benchmark.md`.
- [resnet18_pytorch_fp16_results.csv](resnet18_pytorch_fp16_results.csv): 4 rows from “PyTorch FP16 Results” in `reports/resnet18_inference_benchmark.md`.
- [resnet18_onnx_runtime_fp32_results.csv](resnet18_onnx_runtime_fp32_results.csv): 4 rows from “ONNX Runtime FP32 Results” in `reports/resnet18_inference_benchmark.md`.
- [resnet18_tensorrt_ep_fp32_results.csv](resnet18_tensorrt_ep_fp32_results.csv): 4 rows from “TensorRT EP FP32 Results” in `reports/resnet18_inference_benchmark.md`.
- [resnet18_tensorrt_ep_fp16_results.csv](resnet18_tensorrt_ep_fp16_results.csv): 4 rows from “TensorRT EP FP16 Results” in `reports/resnet18_inference_benchmark.md`.
- [yolov8n_onnx_runtime_cuda_fp32_results.csv](yolov8n_onnx_runtime_cuda_fp32_results.csv): 4 rows from “ONNX Runtime CUDA FP32 Results” in `reports/yolov8n_inference_benchmark.md`.
- [yolov8n_tensorrt_ep_fp16_results.csv](yolov8n_tensorrt_ep_fp16_results.csv): 4 rows from “TensorRT EP FP16 Results” in `reports/yolov8n_inference_benchmark.md`.
- [monte_carlo_cpu_and_cuda_results.csv](monte_carlo_cpu_and_cuda_results.csv): 8 rows from “CPU and CUDA Results” in `cuda_finance/monte_carlo_option_pricing/reports/monte_carlo_benchmark.md`.

Preserve printed precision. These exports cannot establish uncertainty, reproduce an original run, or supply missing provenance. See the [evidence guide](../../docs/benchmark-evidence.md) for publishing future measurements.
