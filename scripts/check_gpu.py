from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"Python version: {sys.version.replace(chr(10), ' ')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "CUDA is not available to PyTorch. If you expected GPU support, check "
            "your NVIDIA driver, WSL2 GPU setup, and that you installed a CUDA-enabled "
            "PyTorch build."
        )
        return 0

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)

    print(f"CUDA version reported by PyTorch: {torch.version.cuda}")
    print(f"GPU name: {props.name}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {device_index}")
    print(f"Total VRAM: {props.total_memory / (1024**3):.2f} GB")

    x = torch.ones((2, 2), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"Tiny CUDA tensor operation succeeded. Result sum: {y.sum().item():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
