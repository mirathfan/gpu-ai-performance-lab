# YOLOv8n Inference Benchmark Report

This report summarizes locally generated YOLOv8n object detection inference benchmark CSV files from `results/`. Benchmark numbers should come from actual runs only and should not be manually invented.

Random image tensors are used only for backend performance measurement. They are not an accuracy, mAP, or detection-quality evaluation.

## System Metadata

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| Python | 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0] |
| ONNX Runtime | 1.26.0 |
| ONNX Runtime CUDA FP32 source CSV | results/yolov8n_onnxruntime.csv |
| ONNX Runtime CUDA FP32 timestamp | 2026-06-04T15:10:07.404723+00:00 |
| TensorRT EP FP16 source CSV | results/yolov8n_tensorrt_ep_fp16.csv |
| TensorRT EP FP16 timestamp | 2026-06-04T15:11:38.296358+00:00 |

## Benchmark Configuration

| Field | Value |
| --- | --- |
| Model | yolov8n |
| Backends | ONNX Runtime CUDA FP32, TensorRT EP FP16 |
| Input shape | [batch_size, 3, image_size, image_size] |
| Image size | 640 |
| Batch sizes | 1, 2, 4, 8 |
| Warmup iterations | 20 |
| Measurement iterations | 100 |

## ONNX Runtime CUDA FP32 Results

| Batch | Image size | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput images/s | Provider |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 640 | 8.068 | 8.504 | 9.376 | 9.574 | 10.975 | 6.401 | 11.111 | 123.940 | CUDAExecutionProvider |
| 2 | 640 | 11.194 | 11.016 | 11.466 | 11.668 | 23.860 | 10.046 | 26.034 | 178.671 | CUDAExecutionProvider |
| 4 | 640 | 18.431 | 18.105 | 18.370 | 18.703 | 27.562 | 17.790 | 27.784 | 217.031 | CUDAExecutionProvider |
| 8 | 640 | 34.194 | 34.132 | 34.716 | 34.969 | 35.503 | 33.383 | 36.141 | 233.962 | CUDAExecutionProvider |

## TensorRT EP FP32 Results

`results/yolov8n_tensorrt_ep*.csv` was not found.

## TensorRT EP FP16 Results

| Batch | Image size | Mean ms | P50 ms | P90 ms | P95 ms | P99 ms | Min ms | Max ms | Throughput images/s | Provider |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 640 | 4.765 | 4.714 | 6.941 | 7.436 | 7.852 | 3.620 | 8.053 | 209.873 | TensorrtExecutionProvider |
| 2 | 640 | 6.139 | 6.253 | 6.584 | 6.799 | 6.881 | 5.463 | 7.039 | 325.793 | TensorrtExecutionProvider |
| 4 | 640 | 10.195 | 10.022 | 11.745 | 11.958 | 12.172 | 9.435 | 12.254 | 392.343 | TensorrtExecutionProvider |
| 8 | 640 | 18.503 | 18.427 | 18.913 | 19.144 | 20.051 | 17.845 | 20.772 | 432.366 | TensorrtExecutionProvider |

## Backend Comparison

| Batch | ONNX Runtime CUDA FP32 P50 ms | ONNX Runtime CUDA FP32 images/s | TensorRT EP FP16 P50 ms | TensorRT EP FP16 images/s | TensorRT EP FP16 latency delta | TensorRT EP FP16 throughput delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.504 | 123.940 | 4.714 | 209.873 | -44.6% | +69.3% |
| 2 | 11.016 | 178.671 | 6.253 | 325.793 | -43.2% | +82.3% |
| 4 | 18.105 | 217.031 | 10.022 | 392.343 | -44.6% | +80.8% |
| 8 | 34.132 | 233.962 | 18.427 | 432.366 | -46.0% | +84.8% |

## Latency vs Throughput

Object detection inference latency measures the time to run the model forward pass for one image batch. Throughput measures images processed per second. Larger batches can improve throughput while also increasing per-batch latency, so real-time applications often need to evaluate both.

## Reproducibility Note

These tables are generated from local CSV files in `results/`. Do not edit benchmark numbers by hand; rerun the benchmark scripts when the environment, model export, driver, or dependencies change.
