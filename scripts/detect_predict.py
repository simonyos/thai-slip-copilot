"""Run the trained field detector on a folder of slip images.

For each image, emits:
    <out>/preview/<stem>.jpg   — annotated image with boxes + class labels
    <out>/preds/<stem>.json    — {"image", "width", "height", "detections": [...]}

Also writes:
    <out>/summary.json         — counts per class, missing-expected-field flags

Usage:
    uv run python scripts/detect_predict.py \
        --weights experiments/runs/detector_v1/weights/best.pt \
        --images /Volumes/SM-EXT/Documents/Slip-orc-sample \
        --out experiments/runs/detector_v1/predict_phone
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from thai_slip_copilot.fields import UNIFIED_CLASSES


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="Folder of .jpg/.jpeg/.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    preview_dir = out_dir / "preview"
    preds_dir = out_dir / "preds"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        paths = paths[: args.limit]

    device = args.device or _pick_device()
    print(f"[predict] device={device}  n_images={len(paths)}  weights={args.weights}")

    model = YOLO(args.weights)
    # stream=True: per-image generator, avoids loading all results into memory
    results = model.predict(
        source=[str(p) for p in paths],
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        stream=True,
        verbose=False,
    )

    class_counts: Counter[str] = Counter()
    per_image_class_presence: Counter[str] = Counter()
    missing_amount: list[str] = []
    missing_qr: list[str] = []
    n_imgs = 0

    for src_path, r in zip(paths, results, strict=False):
        n_imgs += 1
        stem = src_path.stem
        h, w = r.orig_shape  # (height, width)

        detections = []
        present_classes: set[str] = set()
        for box in r.boxes:
            cls_id = int(box.cls.item())
            cls_name = UNIFIED_CLASSES[cls_id]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                {
                    "class_id": cls_id,
                    "class": cls_name,
                    "conf": round(conf, 4),
                    "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )
            class_counts[cls_name] += 1
            present_classes.add(cls_name)

        for c in present_classes:
            per_image_class_presence[c] += 1
        if "amount" not in present_classes:
            missing_amount.append(stem)
        if "qr" not in present_classes:
            missing_qr.append(stem)

        (preds_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image": src_path.name,
                    "width": int(w),
                    "height": int(h),
                    "detections": detections,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        cv2.imwrite(str(preview_dir / f"{stem}.jpg"), r.plot())

    summary = {
        "n_images": n_imgs,
        "total_detections_per_class": dict(class_counts.most_common()),
        "images_with_class": dict(per_image_class_presence.most_common()),
        "coverage_pct": {
            c: round(100 * per_image_class_presence.get(c, 0) / max(n_imgs, 1), 1)
            for c in UNIFIED_CLASSES
        },
        "missing_amount_count": len(missing_amount),
        "missing_qr_count": len(missing_qr),
        "missing_amount_sample": missing_amount[:20],
        "missing_qr_sample": missing_qr[:20],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== summary ===")
    print(f"images:    {n_imgs}")
    print("coverage (% of images with at least one box of that class):")
    for c in UNIFIED_CLASSES:
        print(f"  {c:10s}  {summary['coverage_pct'][c]:5.1f}%")
    print(f"missing-amount: {len(missing_amount)}  missing-qr: {len(missing_qr)}")
    print(f"\noutputs → {out_dir}")


if __name__ == "__main__":
    main()
