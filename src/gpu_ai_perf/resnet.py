"""Model-loading helpers shared by ResNet18 benchmark scripts."""

from __future__ import annotations

from typing import Any


def load_resnet18_model(
    *,
    device: Any,
    precision: str = "fp32",
) -> tuple[Any, str]:
    """Load torchvision ResNet18 with the project-standard weight fallback.

    The benchmark should prefer the same pretrained weights across PyTorch and
    ONNX runs. If weights cannot be loaded in the local environment, the scripts
    fall back to random weights and print the reason instead of pretending that
    pretrained weights were used.
    """
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and torchvision are required. Activate your virtual "
            "environment and run `pip install -r requirements.txt` from the "
            "repo root."
        ) from exc

    weights_enum = getattr(models, "ResNet18_Weights", None)
    try:
        if weights_enum is not None:
            weights = weights_enum.DEFAULT
            model = models.resnet18(weights=weights)
            weights_label = str(weights)
        else:
            model = models.resnet18(pretrained=True)
            weights_label = "pretrained=True"
    except Exception as exc:
        print(
            "Could not load pretrained ResNet18 weights; falling back to "
            f"weights=None. Reason: {exc}"
        )
        # Keep fallback weights deterministic so ONNX export and validation can
        # still compare the same model if pretrained weights are unavailable.
        torch.manual_seed(0)
        if weights_enum is not None:
            model = models.resnet18(weights=None)
        else:
            model = models.resnet18(pretrained=False)
        weights_label = "None"

    model.eval()
    model.to(device)
    if precision == "fp16":
        if str(device) == "cpu":
            raise RuntimeError("FP16 ResNet18 loading is only supported on CUDA.")
        model.half()
    elif precision != "fp32":
        raise ValueError("precision must be 'fp32' or 'fp16'.")

    return model, weights_label
