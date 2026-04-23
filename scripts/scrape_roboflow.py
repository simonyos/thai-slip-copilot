"""Scrape real Thai slip images from the two Roboflow Universe datasets that
contain actual slips (not synthetic mockups or bank-logo crops).

Sources (audited 2026-04; both CC-BY-4.0 per Roboflow Universe defaults):

  1. pipat/slip-k3xff        — 645 images, classes: Bank, Name, Price, QR
  2. colamarc/th-slip-ocr-k  — 35 images,  classes: date, accnum, amount, 0

Both ship with per-field bboxes that can bootstrap our stage-1 field detector
without any hand-labeling — a big head start over the plate project.

Output layout (mirrors the thai-plate-synth weekend-5 convention):

    data/real_scrape/roboflow/
      images/{prefix}_{filename}.jpg    — deduplicated images
      provenance.jsonl                   — one record per surviving image
      sources.json                       — per-source summary (license, counts)

Provenance records have workspace/project/version/license fields so the
eventual training-set ablation can trace every image back to its origin.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Each source: (workspace, project-slug, version, display-name).
# Roboflow Universe's default license is CC-BY-4.0 unless the owner overrides.
SOURCES: tuple[tuple[str, str, int, str], ...] = (
    ("pipat", "slip-k3xff", 2, "pipat/slip"),            # v2, 645 images
    ("colamarc", "th-slip-ocr-k", 1, "colamarc/th-slip-ocr-k"),  # v1, 103 images
)


@dataclass
class Fetched:
    source: str
    src_path: Path
    workspace: str
    project: str
    version: int
    split: str  # "train" / "valid" / "test"
    license: str


def _fetch_source(api_key: str, workspace: str, project: str, version: int,
                  work_dir: Path) -> tuple[list[Fetched], str]:
    """Download one Roboflow Universe source. Returns (fetched, license)."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    meta = proj.version(version)
    # Universe project license appears as `proj.license` when set; fall back
    # to the platform default.
    lic = getattr(proj, "license", None) or "CC BY 4.0"

    dl_dir = work_dir / f"{workspace}__{project}__v{version}"
    # Let the Roboflow SDK manage its own working directory — pre-creating an
    # empty one confuses its "already extracted" check and silently skips.
    work_dir.mkdir(parents=True, exist_ok=True)
    if dl_dir.exists():
        shutil.rmtree(dl_dir)

    # YOLO format is the most compatible (images/{split}/*.jpg + labels/*.txt).
    ds = meta.download("yolov8", location=str(dl_dir))
    root = Path(ds.location)

    fetched: list[Fetched] = []
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                fetched.append(Fetched(
                    source=f"{workspace}/{project}",
                    src_path=p,
                    workspace=workspace,
                    project=project,
                    version=version,
                    split=split,
                    license=lic,
                ))
    return fetched, lic


def _sha256(path: Path) -> str:
    """Exact byte-level fingerprint. Slips are template-heavy — aHash over-
    collapses distinct slips that share a bank template layout. SHA256 only
    drops TRUE exact-duplicate files (same image posted to multiple
    datasets), which is the honest dedupe semantics for this domain."""
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dedupe(fetched: Iterable[Fetched]) -> tuple[list[Fetched], list[Fetched]]:
    """Exact-bytes dedupe. Keep first occurrence, drop later exact copies."""
    keep: list[Fetched] = []
    seen: set[str] = set()
    drops: list[Fetched] = []
    for f in fetched:
        try:
            h = _sha256(f.src_path)
        except Exception as e:
            print(f"  skip (hash error): {f.src_path.name}: {e}", file=sys.stderr)
            continue
        if h in seen:
            drops.append(f)
            continue
        seen.add(h)
        keep.append(f)
    return keep, drops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default="HKfKtut6mgYiwQtOVz51",
                    help="Roboflow API key (reuse plate-project key by default).")
    ap.add_argument("--out", type=Path, default=Path("data/real_scrape/roboflow"))
    ap.add_argument("--work-dir", type=Path, default=Path(".rf_cache"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images").mkdir(exist_ok=True)

    all_fetched: list[Fetched] = []
    summary: dict[str, dict] = {}
    for workspace, project, version, display in SOURCES:
        print(f"[fetch] {display} v{version}")
        fetched, lic = _fetch_source(args.api_key, workspace, project, version,
                                     args.work_dir)
        print(f"  got {len(fetched)} images, license={lic}")
        all_fetched.extend(fetched)
        key = f"{workspace}/{project}"
        summary[key] = {
            "display": display,
            "workspace": workspace, "project": project, "version": version,
            "raw_count": len(fetched), "license": lic,
        }

    print(f"\n[dedupe] {len(all_fetched)} raw → …", flush=True)
    keep, drops = _dedupe(all_fetched)
    print(f"  {len(keep)} kept, {len(drops)} dropped as exact-byte dupes")

    # Copy kept images with a source prefix, emit provenance
    prov_path = args.out / "provenance.jsonl"
    src_path = args.out / "sources.json"
    with prov_path.open("w", encoding="utf-8") as prov_f:
        for f in keep:
            prefix = f"{f.workspace}__{f.project}"
            dst_name = f"{prefix}__{f.src_path.name}"
            dst = args.out / "images" / dst_name
            shutil.copy2(f.src_path, dst)
            prov_f.write(json.dumps({
                "image": dst_name,
                "source": f.source,
                "workspace": f.workspace,
                "project": f.project,
                "version": f.version,
                "split": f.split,
                "license": f.license,
                "src_path": str(f.src_path),
            }, ensure_ascii=False) + "\n")

    per_source = {key: {"kept": 0, **meta} for key, meta in summary.items()}
    for f in keep:
        per_source[f"{f.workspace}/{f.project}"]["kept"] += 1
    src_path.write_text(json.dumps(per_source, indent=2, ensure_ascii=False))

    print(f"\nwrote {len(keep)} images → {args.out / 'images'}")
    print(f"wrote provenance → {prov_path}")
    print(f"wrote source summary → {src_path}")


if __name__ == "__main__":
    main()
