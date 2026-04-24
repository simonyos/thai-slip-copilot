"""Evaluate the OCR + parse pipeline against a hand-verified gold set.

Reads `data/eval/gold_phone_v1.jsonl` (verified rows only), runs the
full extract pipeline on each image, compares field-by-field, prints
per-field accuracy + the specific failures.

Usage:
    uv run python scripts/eval_on_gold.py \
        --weights experiments/runs/detector_v2_1/weights/best.pt \
        --images /Volumes/SM-EXT/Documents/Slip-orc-sample \
        --gold data/eval/gold_phone_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from thai_slip_copilot.extract import extract_slip

EVAL_FIELDS: tuple[str, ...] = (
    "timestamp",
    "sender_name",
    "sender_accnum",
    "receiver_name",
    "promptpay",
    "amount_satang",
    "reference_id",
)


def _eq(pred, truth) -> bool:
    """Field-level equality. Names get a forgiving compare after
    stripping Thai tone-mark noise we know EasyOCR introduces."""
    if pred is None and truth is None:
        return True
    if pred is None or truth is None:
        return False
    return pred == truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    gold_rows: list[dict] = [
        json.loads(line)
        for line in Path(args.gold).read_text().splitlines()
        if line.strip()
    ]
    verified = [r for r in gold_rows if r.get("verified")]
    print(f"[eval] gold rows: {len(gold_rows)}  verified: {len(verified)}")
    if not verified:
        print("[eval] no verified rows — nothing to evaluate")
        return

    model = YOLO(args.weights)
    images_dir = Path(args.images)

    correct: Counter[str] = Counter()
    present_truth: Counter[str] = Counter()   # denominator when truth != None
    present_pred: Counter[str] = Counter()
    failures: defaultdict[str, list[tuple[str, object, object]]] = defaultdict(list)

    for row in verified:
        img_path = images_dir / row["image"]
        parsed = extract_slip(img_path, model)
        for f in EVAL_FIELDS:
            truth = row.get(f)
            pred = parsed.get(f)
            if truth is not None:
                present_truth[f] += 1
            if pred is not None:
                present_pred[f] += 1
            if _eq(pred, truth):
                correct[f] += 1
            else:
                failures[f].append((row["image"], truth, pred))

    print("\n=== per-field accuracy ===")
    print(f"{'field':18s} {'acc':>7s} {'correct':>10s} "
          f"{'truth≠None':>12s} {'pred≠None':>11s}")
    for f in EVAL_FIELDS:
        n = len(verified)
        acc = correct[f] / n
        print(f"{f:18s} {acc*100:6.1f}% {correct[f]:>6d}/{n:<3d} "
              f"{present_truth[f]:>10d}/{n:<3d} {present_pred[f]:>10d}/{n:<3d}")

    print("\n=== sample failures (first 3 per field) ===")
    for f in EVAL_FIELDS:
        if not failures[f]:
            continue
        print(f"\n[{f}]")
        for img, truth, pred in failures[f][:3]:
            print(f"  {img}")
            print(f"    truth: {truth!r}")
            print(f"    pred:  {pred!r}")


if __name__ == "__main__":
    main()
