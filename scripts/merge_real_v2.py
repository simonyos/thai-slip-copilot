"""Combine `data/real_v1` (merged pipat+colamarc Roboflow) and
`data/real_phone_v1` (K+ phone slips, template-labeled) into a single
YOLO training bundle `data/real_v2`.

Rationale: training on phone-only data would specialize the detector to
K+ and regress on the broader multi-bank Roboflow coverage. Mixing
preserves that coverage while adding the new K+ real-photo examples
that fix the 4 previously-dead classes.

Output layout (YOLOv8 standard):
    data/real_v2/
      images/{train,val}/*.{jpg,JPG}   (symlinks, prefixed by source)
      labels/{train,val}/*.txt
      dataset.yaml

Usage:
    uv run python scripts/merge_real_v2.py \
        --v1 data/real_v1 --phone data/real_phone_v1 --out data/real_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from thai_slip_copilot.fields import UNIFIED_CLASSES


def _link_pair(
    src_img: Path, src_lbl: Path, dst_img_dir: Path, dst_lbl_dir: Path, prefix: str
) -> None:
    new_stem = f"{prefix}__{src_img.stem}"
    dst_img = dst_img_dir / f"{new_stem}{src_img.suffix}"
    dst_lbl = dst_lbl_dir / f"{new_stem}.txt"
    if not dst_img.exists():
        # Follow the symlink to its real target so downstream rsyncs resolve.
        dst_img.symlink_to(src_img.resolve())
    if not dst_lbl.exists():
        dst_lbl.write_bytes(src_lbl.read_bytes())


def _merge_split(v1: Path, phone: Path, out: Path, split: str) -> int:
    img_out = out / "images" / split
    lbl_out = out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_root, prefix in ((v1, "rf"), (phone, "kp")):
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            _link_pair(img_path, lbl_path, img_out, lbl_out, prefix)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="data/real_v1")
    ap.add_argument("--phone", default="data/real_phone_v1")
    ap.add_argument("--out", default="data/real_v2")
    args = ap.parse_args()

    v1 = Path(args.v1)
    phone = Path(args.phone)
    out = Path(args.out)

    n_train = _merge_split(v1, phone, out, "train")
    n_val = _merge_split(v1, phone, out, "val")

    yaml_text = (
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(UNIFIED_CLASSES)}\n"
        "names:\n"
    )
    for i, name in enumerate(UNIFIED_CLASSES):
        yaml_text += f"  {i}: {name}\n"
    (out / "dataset.yaml").write_text(yaml_text)

    print(f"train={n_train}  val={n_val}  total={n_train + n_val}")
    print(f"out → {out}")


if __name__ == "__main__":
    main()
