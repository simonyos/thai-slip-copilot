"""Draw YOLO labels on top of images for visual QA.

Usage:
    uv run python scripts/preview_labels.py \
        --dataset data/real_phone_v1 \
        --split train \
        --out experiments/figures/real_phone_v1_preview \
        --n 30
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from thai_slip_copilot.fields import UNIFIED_CLASSES

CLASS_COLORS = {
    "bank":      (31, 119, 180),
    "name":      (255, 127, 14),
    "amount":    (44, 160, 44),
    "qr":        (214, 39, 40),
    "date":      (148, 103, 189),
    "accnum":    (140, 86, 75),
    "promptpay": (227, 119, 194),
    "reference": (127, 127, 127),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Root with images/, labels/, dataset.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset)
    images_dir = root / "images" / args.split
    labels_dir = root / "labels" / args.split
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(images_dir.iterdir())
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    paths = paths[: args.n]

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for img_path in paths:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            draw = ImageDraw.Draw(im)
            W, H = im.size
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H
                name = UNIFIED_CLASSES[cls_id]
                color = CLASS_COLORS[name]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1 + 2, max(0, y1 - 20)), name, fill=color, font=font)
            im.save(out_dir / img_path.name)
    print(f"wrote {args.n} previews → {out_dir}")


if __name__ == "__main__":
    main()
