"""Pre-fill a 50-slip gold-transcription set.

Picks N slips deterministically (seed-stable) from a folder, runs the
full extract pipeline on each, and writes a JSONL with one record per
slip. Each record is pre-populated with the pipeline's current guess
plus a `verified: false` flag. You then open the JSONL in a text editor
and correct each field against the image, flipping `verified: true`
when you're done with a row.

Usage:
    uv run python scripts/build_gold_set.py \
        --weights experiments/runs/detector_v2_1/weights/best.pt \
        --images /Volumes/SM-EXT/Documents/Slip-orc-sample \
        --out data/eval/gold_phone_v1.jsonl \
        --n 50

Gold schema per line (JSONL):
    {
      "image": "IMG_5249.JPG",
      "verified": false,
      "timestamp":       "2025-08-04T11:55:00+07:00",   # ISO, Bangkok TZ
      "sender_name":     "นาย ซีโมน ย",
      "sender_accnum":   "xxx-x-x7829-x",
      "receiver_name":   "แกร็บแท็กซี่ (ประเทศไทย)",
      "receiver_accnum": null,
      "promptpay":       "202508042422984",
      "amount_satang":   10500,
      "reference_id":    "015216115552AQR06997",
      "notes":           ""
    }
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from thai_slip_copilot.extract import extract_slip


GOLD_KEYS: tuple[str, ...] = (
    "timestamp",
    "sender_name",
    "sender_accnum",
    "receiver_name",
    "receiver_accnum",
    "promptpay",
    "amount_satang",
    "reference_id",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: if the file exists, keep already-verified rows untouched
    # and only re-predict the unverified ones (lets you re-run after
    # improving the pipeline without blowing away your hand-corrections).
    existing: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            existing[rec["image"]] = rec

    paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    chosen = paths[: args.n]

    print(f"[gold-set] n={len(chosen)}  out={out_path}")
    model = YOLO(args.weights)

    records: list[dict] = []
    for p in chosen:
        prev = existing.get(p.name)
        if prev and prev.get("verified"):
            records.append(prev)
            continue

        parsed = extract_slip(p, model)
        rec: dict = {
            "image":           p.name,
            "verified":        False,
            "timestamp":       parsed["timestamp"],
            "sender_name":     parsed["sender_name"],
            "sender_accnum":   parsed["sender_accnum"],
            "receiver_name":   parsed["receiver_name"],
            "receiver_accnum": None,  # not yet parsed — fill by hand
            "promptpay":       parsed["promptpay"],
            "amount_satang":   parsed["amount_satang"],
            "reference_id":    parsed["reference_id"],
            "notes":           "",
        }
        records.append(rec)

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    )
    n_verified = sum(1 for r in records if r.get("verified"))
    print(f"[gold-set] wrote {len(records)} rows  verified={n_verified}")
    print("\nNext: open the file, correct each field against its image, set verified: true")


if __name__ == "__main__":
    main()
