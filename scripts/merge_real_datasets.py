"""Merge the two scraped Roboflow slip datasets into a unified YOLO training bundle.

Source layouts (from `.rf_cache/` after `scripts/scrape_roboflow.py`):

    .rf_cache/pipat__slip-k3xff__v2/{train,valid,test}/{images,labels}
    .rf_cache/colamarc__th-slip-ocr-k__v1/{train,valid,test}/{images,labels}

Output layout (standard YOLOv8):

    data/real_v1/
      images/{train,val}/*.jpg   (symlinks — avoid GB-scale duplication)
      labels/{train,val}/*.txt   (rewritten with unified class IDs)
      dataset.yaml

Class remap table lives in `src/thai_slip_copilot/fields.py`.

Colamarc has only a `train` split (no val); we hold out a deterministic
10% from pipat's train as the unified val. Pipat's separate `valid` set
becomes part of the unified train.

Usage:
    uv run python scripts/merge_real_datasets.py --out data/real_v1
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from thai_slip_copilot.fields import (
    COLAMARC_REMAP,
    N_CLASSES,
    PIPAT_REMAP,
    UNIFIED_CLASSES,
)

Remap = dict[int, int]

SOURCES: tuple[tuple[str, Path, Remap], ...] = (
    ("pipat",    Path(".rf_cache/pipat__slip-k3xff__v2"),      PIPAT_REMAP),
    ("colamarc", Path(".rf_cache/colamarc__th-slip-ocr-k__v1"), COLAMARC_REMAP),
)


def _remap_label(src: Path, dst: Path, remap: Remap) -> bool:
    """Rewrite a YOLO .txt label with class-id remap. Returns True if any
    bbox survived (source classes not in remap are dropped)."""
    out_lines: list[str] = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        src_cls = int(parts[0])
        if src_cls not in remap:
            continue
        out_lines.append(f"{remap[src_cls]} " + " ".join(parts[1:]))
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return bool(out_lines)


def _link_image(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def merge(out_dir: Path, val_frac: float = 0.10, seed: int = 42) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "skipped_no_label": 0, "skipped_no_bbox": 0}

    for prefix, root, remap in SOURCES:
        if not root.is_dir():
            print(f"[skip] {prefix}: {root} not found — run scrape_roboflow first")
            continue
        # Collect all (image, label) pairs from train+valid+test splits.
        pairs: list[tuple[Path, Path]] = []
        for split in ("train", "valid", "test"):
            img_dir = root / split / "images"
            lbl_dir = root / split / "labels"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.is_file():
                    counts["skipped_no_label"] += 1
                    continue
                pairs.append((img, lbl))

        # Deterministic val split inside each source, so each source contributes
        # to both train and val (avoids a domain split confound).
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_frac))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        for split, items in (("train", train_pairs), ("val", val_pairs)):
            for img, lbl in items:
                stem = f"{prefix}__{img.stem}"
                dst_img = out_dir / "images" / split / f"{stem}{img.suffix}"
                dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
                kept = _remap_label(lbl, dst_lbl, remap)
                if not kept:
                    dst_lbl.unlink(missing_ok=True)
                    counts["skipped_no_bbox"] += 1
                    continue
                _link_image(img, dst_img)
                counts[split] += 1

    yaml = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {N_CLASSES}",
        "names:",
    ]
    for i, name in enumerate(UNIFIED_CLASSES):
        yaml.append(f"  {i}: {name}")
    (out_dir / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/real_v1"))
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    counts = merge(args.out, val_frac=args.val_frac, seed=args.seed)
    total = counts["train"] + counts["val"]
    print(f"train={counts['train']}  val={counts['val']}  total={total}")
    print(f"skipped: no-label-file={counts['skipped_no_label']}  "
          f"no-bbox-after-remap={counts['skipped_no_bbox']}")
    print(f"wrote {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
