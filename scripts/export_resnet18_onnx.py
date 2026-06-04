from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ResNet18 from PyTorch to ONNX.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/resnet18.onnx"),
        help="Path for the exported ONNX model.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export the batch dimension as dynamic.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc
    return torch


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.opset <= 0:
        raise ValueError("--opset must be > 0.")


def main() -> int:
    args = parse_args()
    validate_args(args)

    torch = import_torch()
    from gpu_ai_perf.resnet import load_resnet18_model

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, weights_label = load_resnet18_model(device=device, precision="fp32")
    dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=device)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    print(f"Exporting ResNet18 to ONNX: {output_path}")
    print(f"Weights: {weights_label}")
    print(f"Opset: {args.opset}")
    print(f"Dynamic batch: {args.dynamic_batch}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    if not output_path.exists():
        raise RuntimeError(f"ONNX export failed; file was not created: {output_path}")

    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX checker validation: passed")
    except ImportError:
        print("Warning: onnx is not installed, so ONNX checker validation was skipped.")
    except Exception as exc:
        raise RuntimeError(f"ONNX checker validation failed: {exc}") from exc

    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"Exported ONNX file size: {file_size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
