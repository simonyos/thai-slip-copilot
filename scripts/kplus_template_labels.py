"""Generate YOLO labels for K+ phone slips via a fixed template.

Context: `detector_v1` — trained on merged pipat+colamarc Roboflow data —
scores 0.995 mAP50 on its own val set but collapses on real K+ phone
slips for 4 classes (date/accnum/promptpay/reference at 6–20% coverage).
Root cause: label-incompleteness across the source datasets, not a
domain gap in the visual features.

Fix: exploit the fact that K+ uses a truly fixed template. For each of
the 4 dead classes plus QR (which also degraded on phone slips), we
know the normalized position within ~1–2% across all K+ slips. We
generate synthetic-but-accurate labels by projecting a template onto
every image, then merge with the existing detector's working-class
predictions (bank/name/amount) to produce a full 8-class training set.

Template coords (x_center, y_center, w, h — all normalized to image
[w, h]) were calibrated by eye from 5 representative slips covering
the merchant, PromptPay, KBank-to-KBank, and KBank-to-BBL receiver
variants. They are intentionally slightly loose — YOLO trains fine on
±1–2% label noise and benefits from the tight-fit being learned from
the existing detector's accurate boxes on bank/name/amount.

Variant-dependent classes (receiver accnum vs promptpay) are skipped
in this pass — they need per-image disambiguation. Sender accnum is
always present and at a fixed position, so we include it.

Usage:
    uv run python scripts/kplus_template_labels.py \
        --images /Volumes/SM-EXT/Documents/Slip-orc-sample \
        --detector-preds experiments/runs/detector_v1/predict_phone/preds \
        --out data/real_phone_v1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from thai_slip_copilot.fields import CLASS_TO_ID, UNIFIED_CLASSES

# --- Template: normalized (x_center, y_center, w, h) for K+ fixed slots ---
# Calibrated by eye from IMG_5249, 5400, 5493, 5793, 5459, 6194 then
# corrected after viewing 6 template-labeled previews.
# "sender_accnum" = the "xxx-x-x7829-x" line under the sender's bank name.
KPLUS_TEMPLATE: dict[str, tuple[float, float, float, float]] = {
    # date + time line in the header, e.g. "4 ส.ค. 68  11:55 น."
    "date":          (0.215, 0.088, 0.32, 0.032),
    # QR code bottom-right. Override the detector's flaky QR detection
    # (44% coverage on phone slips) with the template box.
    "qr":            (0.800, 0.815, 0.260, 0.240),
    # sender's masked account number "xxx-x-x7829-x"
    # Slightly taller box to absorb per-screenshot vertical drift: image
    # heights vary 1287–1401 (6%) across iPhone generations, so a tight
    # 3% box would miss the text on the tallest screenshots.
    "accnum_sender": (0.260, 0.275, 0.230, 0.050),
}

# Reference number is at a fixed offset ABOVE the detected amount box.
# This anchors cleanly for both standard and bill-pay slip variants
# (bill-pay adds an extra biller row that shifts the whole bottom block
# down — anchoring to `amount` rather than a fixed image-y tracks that).
REFERENCE_OFFSET_FROM_AMOUNT_Y: float = -0.093  # normalized
REFERENCE_WH: tuple[float, float] = (0.34, 0.034)
REFERENCE_X_CENTER: float = 0.235


def _write_yolo_label(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [
        f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        for cls, xc, yc, w, h in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _detector_boxes_for(
    detector_preds_dir: Path, stem: str, img_w: int, img_h: int, min_conf: float
) -> tuple[list[tuple[int, float, float, float, float]], float | None]:
    """Read the v1 detector's JSON preds for one image. Returns:
      - YOLO-format boxes for classes we trust (bank/name/amount) above conf,
      - the normalized y_center of the `amount` box (or None if not found),
        used as the anchor for the reference-number template.
    """
    keep = {"bank", "name", "amount"}
    preds_path = detector_preds_dir / f"{stem}.json"
    if not preds_path.exists():
        return [], None
    data = json.loads(preds_path.read_text())
    out: list[tuple[int, float, float, float, float]] = []
    amount_yc: float | None = None
    for det in data.get("detections", []):
        if det["class"] not in keep or det["conf"] < min_conf:
            continue
        x1, y1, x2, y2 = det["xyxy"]
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        out.append((CLASS_TO_ID[det["class"]], xc, yc, w, h))
        if det["class"] == "amount":
            amount_yc = yc
    return out, amount_yc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--detector-preds", required=True,
                    help="Folder of per-image JSON from detect_predict.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-detector-conf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images_dir = Path(args.images)
    preds_dir = Path(args.detector_preds)
    out_dir = Path(args.out)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rng = random.Random(args.seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_set = set(shuffled[:n_val])

    n_written = 0
    n_skipped_no_preds = 0
    for p in paths:
        stem = p.stem
        with Image.open(p) as im:
            img_w, img_h = im.size

        # 1. detector boxes for bank/name/amount, plus the amount y-anchor
        det_boxes, amount_yc = _detector_boxes_for(
            preds_dir, stem, img_w, img_h, args.min_detector_conf
        )
        if not det_boxes:
            n_skipped_no_preds += 1

        # 2. fixed-position template boxes (date, qr, sender accnum)
        tpl_boxes: list[tuple[int, float, float, float, float]] = []
        for name, (xc, yc, w, h) in KPLUS_TEMPLATE.items():
            cls_name = "accnum" if name == "accnum_sender" else name
            tpl_boxes.append((CLASS_TO_ID[cls_name], xc, yc, w, h))

        # 3. reference box anchored to detected amount y. If the detector
        # didn't find an amount on this slip (~1% of cases), skip the
        # reference label rather than planting it at a guessed position.
        if amount_yc is not None:
            ref_yc = amount_yc + REFERENCE_OFFSET_FROM_AMOUNT_Y
            tpl_boxes.append(
                (CLASS_TO_ID["reference"], REFERENCE_X_CENTER, ref_yc, *REFERENCE_WH)
            )

        boxes = det_boxes + tpl_boxes
        split = "val" if p in val_set else "train"
        lbl_path = out_dir / "labels" / split / f"{stem}.txt"
        img_link = out_dir / "images" / split / p.name
        _write_yolo_label(lbl_path, boxes)
        if not img_link.exists():
            img_link.symlink_to(p.resolve())
        n_written += 1

    # 3. dataset.yaml
    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(UNIFIED_CLASSES)}\n"
        "names:\n"
    )
    for i, name in enumerate(UNIFIED_CLASSES):
        yaml_text += f"  {i}: {name}\n"
    (out_dir / "dataset.yaml").write_text(yaml_text)

    print(f"labeled {n_written} images  (no-detector-preds: {n_skipped_no_preds})")
    print(f"val split: {len(val_set)} images")
    print(f"out → {out_dir}")


if __name__ == "__main__":
    main()
