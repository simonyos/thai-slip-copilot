"""Smoke-test the OCR pipeline on a handful of phone slips.

Usage:
    uv run python scripts/ocr_smoke.py \
        --weights experiments/runs/detector_v2_1/weights/best.pt \
        --images /Volumes/SM-EXT/Documents/Slip-orc-sample \
        --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thai_slip_copilot.ocr import run_ocr_on_slip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dump-crops", default=None, help="Dir to save each crop for debugging")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.n]

    dump_dir = Path(args.dump_crops) if args.dump_crops else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    from thai_slip_copilot.ocr import crop_with_policy

    for p in paths:
        print(f"\n=== {p.name} ===")
        img = Image.open(p).convert("RGB")
        for i, f in enumerate(run_ocr_on_slip(p, model)):
            print(
                f"  {f.class_name:10s}  det={f.detector_conf:.2f}  "
                f"ocr={f.ocr_conf:.2f}  {f.raw_text!r}"
            )
            if dump_dir:
                crop_with_policy(img, f.bbox_xyxy, f.class_name).save(
                    dump_dir / f"{p.stem}__{i:02d}_{f.class_name}.png"
                )


if __name__ == "__main__":
    main()
