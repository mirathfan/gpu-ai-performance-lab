from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch ResNet18 outputs against an ONNX Runtime model."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/resnet18.onnx"),
        help="Path to the exported ONNX model.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance passed to numpy.allclose.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance passed to numpy.allclose.",
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


def import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc
    return ort


def select_ort_providers(ort, requested_device: str) -> list[str]:
    available = ort.get_available_providers()
    if requested_device == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print(
            "Warning: CUDAExecutionProvider is not available in ONNX Runtime; "
            "falling back to CPUExecutionProvider for validation."
        )
    return ["CPUExecutionProvider"]


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.atol < 0:
        raise ValueError("--atol must be >= 0.")
    if args.rtol < 0:
        raise ValueError("--rtol must be >= 0.")

    onnx_path = resolve_repo_path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. Run "
            "`python scripts/export_resnet18_onnx.py --dynamic-batch` first."
        )

    torch = import_torch()
    ort = import_onnxruntime()
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc

    from gpu_ai_perf.resnet import load_resnet18_model

    torch_device = torch.device("cpu")
    if args.device == "cuda":
        if torch.cuda.is_available():
            torch_device = torch.device("cuda")
        else:
            print(
                "Warning: CUDA was requested but PyTorch CUDA is not available; "
                "running the PyTorch reference on CPU."
            )

    providers = select_ort_providers(ort, args.device)
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as exc:
        raise RuntimeError(f"Failed to create ONNX Runtime session: {exc}") from exc
    input_name = session.get_inputs()[0].name

    model, weights_label = load_resnet18_model(device=torch_device, precision="fp32")

    # Use one CPU tensor as the source of truth so PyTorch and ONNX Runtime see
    # the same random values even when either backend runs on CUDA.
    torch.manual_seed(0)
    x_cpu = torch.randn(args.batch_size, 3, 224, 224)
    x_torch = x_cpu.to(torch_device)

    with torch.no_grad():
        torch_output = model(x_torch).detach().cpu().numpy()

    ort_output = session.run(None, {input_name: x_cpu.numpy()})[0]

    diff = np.abs(torch_output - ort_output)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    allclose_passed = bool(
        np.allclose(torch_output, ort_output, atol=args.atol, rtol=args.rtol)
    )
    torch_top1 = np.argmax(torch_output, axis=1).astype(int).tolist()
    ort_top1 = np.argmax(ort_output, axis=1).astype(int).tolist()
    top1_matches = torch_top1 == ort_top1

    # Logits can differ slightly across CUDA backends while still producing the
    # same prediction. This fallback is intentionally small and printed below so
    # validation remains transparent.
    top1_max_abs_tolerance = max(args.atol, 5e-2)
    top1_small_diff_passed = top1_matches and max_abs_diff <= top1_max_abs_tolerance
    passed = allclose_passed or top1_small_diff_passed

    print(f"ONNX model: {onnx_path}")
    print(f"Weights: {weights_label}")
    print(f"Requested device: {args.device}")
    print(f"PyTorch device: {torch_device}")
    print(f"ONNX Runtime available providers: {ort.get_available_providers()}")
    print(f"ONNX Runtime session providers: {session.get_providers()}")
    print(f"Max absolute difference: {max_abs_diff:.8f}")
    print(f"Mean absolute difference: {mean_abs_diff:.8f}")
    print(f"atol: {args.atol}")
    print(f"rtol: {args.rtol}")
    print(f"numpy.allclose passed: {allclose_passed}")
    print(f"PyTorch top-1 class: {torch_top1[0] if len(torch_top1) == 1 else torch_top1}")
    print(f"ONNX Runtime top-1 class: {ort_top1[0] if len(ort_top1) == 1 else ort_top1}")
    print(f"Top-1 class matches: {top1_matches}")
    print(f"Top-1 fallback max absolute tolerance: {top1_max_abs_tolerance}")
    print(f"Top-1 small-difference fallback passed: {top1_small_diff_passed}")
    print(f"Validation passed: {passed}")

    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
