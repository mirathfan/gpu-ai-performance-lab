from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/yolov8n.onnx"),
        help="Path for the exported ONNX model.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic batch/input axes when supported by Ultralytics.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def validate_args(args: argparse.Namespace) -> None:
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be > 0.")
    if args.opset <= 0:
        raise ValueError("--opset must be > 0.")


def main() -> int:
    args = parse_args()
    validate_args(args)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Activate your virtual environment and run "
            "`pip install -r requirements.txt` from the repo root."
        ) from exc

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
    )

    exported_path = Path(exported)
    if not exported_path.is_absolute():
        exported_path = REPO_ROOT / exported_path

    if not exported_path.exists():
        raise RuntimeError(f"Ultralytics export did not create an ONNX file: {exported_path}")

    if exported_path.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(exported_path), str(output_path))

    if not output_path.exists():
        raise RuntimeError(f"ONNX export failed; file was not created: {output_path}")

    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"Exported YOLOv8n ONNX model: {output_path}")
    print(f"Image size: {args.imgsz}")
    print(f"Opset: {args.opset}")
    print(f"Dynamic export: {args.dynamic}")
    print(f"ONNX file size: {file_size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
