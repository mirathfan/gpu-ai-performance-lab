# Career notes

## Project summary

- Built a reproducible GPU inference benchmarking lab on WSL2 Ubuntu with an RTX 3060 Laptop GPU, measuring latency percentiles and throughput across PyTorch, ONNX Runtime CUDA, and ONNX Runtime TensorRT Execution Provider.
- Exported and validated deep learning models through ONNX, then benchmarked FP32 and FP16 inference paths for ResNet18 classification and YOLOv8n object detection with local CSV, report, and plot generation.
- Demonstrated TensorRT EP acceleration and precision tradeoffs, including ResNet18 TensorRT EP FP16 at 0.860 ms batch-1 p50 latency and YOLOv8n TensorRT EP FP16 at 4.714 ms batch-1 p50 latency.
- Implemented a C++/CUDA Monte Carlo European option pricing benchmark with CPU and cuRAND GPU paths for finance-oriented GPU performance analysis.
- Measured CUDA Monte Carlo scaling behavior on an RTX 3060 Laptop GPU, including a 7.25x speedup at 10M simulated paths and slower CUDA performance for small path counts where launch/setup overhead dominates.
