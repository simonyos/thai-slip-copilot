"""End-to-end smoke test: image → structured dict.

Usage:
    uv run python scripts/extract_smoke.py \
        --weights experiments/runs/detector_v2_1/weights/best.pt \
        --images /Volumes/SM-EXT/Documents/Slip-orc-sample \
        --n 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thai_slip_copilot.extract import extract_slip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.n]

    for p in paths:
        print(f"\n=== {p.name} ===")
        result = extract_slip(p, model)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
