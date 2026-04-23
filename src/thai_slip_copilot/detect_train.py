"""YOLOv8 field detector training.

Thin wrapper around Ultralytics that fixes the hyperparameters we care
about and auto-picks the accelerator (CUDA > MPS > CPU). Training reads
`data/real_v1/dataset.yaml` (the merged pipat+colamarc bundle) and
writes weights + metrics to `experiments/runs/<name>/`.

Usage:
    uv run python -m thai_slip_copilot.detect_train
    uv run python -m thai_slip_copilot.detect_train --epochs 3 --name smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_v1/dataset.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="detector_v1")
    ap.add_argument("--device", default=None, help="cuda idx, 'mps', or 'cpu' (auto if unset)")
    ap.add_argument("--project", default="experiments/runs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    device = args.device or _pick_device()
    print(f"[detect_train] device={device}  data={args.data}  model={args.model}")

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        exist_ok=True,
        patience=20,
        cos_lr=True,
    )


if __name__ == "__main__":
    main()
