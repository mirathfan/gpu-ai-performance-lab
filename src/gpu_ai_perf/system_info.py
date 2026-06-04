"""System metadata captured alongside benchmark results."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Any

import torch


def get_system_info() -> dict[str, Any]:
    """Return system metadata useful for interpreting benchmark output."""
    cuda_available = torch.cuda.is_available()

    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": None,
        "gpu_vram_gb": None,
    }

    if cuda_available:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = props.total_memory / (1024**3)

    return info
