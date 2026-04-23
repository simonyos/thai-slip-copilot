"""Render synthetic Thai bank-transfer slips with per-field bboxes.

Weekend-1 MVP — one generic "modern Thai bank app" template. Later
weekends add per-bank templates (SCB blue header, KBank green, BBL
yellow, …) and photographic degradation (phone capture glare, crop,
rotation). For now the renderer produces clean rasters that already
exercise the full schema → JSON ground-truth pipeline.

Output layout (per slip):

    out_dir/
      images/slip_000000.png          # RGB render
      labels/slip_000000.json         # { "slip": Slip, "bboxes": {field: [x1,y1,x2,y2]} }

Per-field bboxes let weekend-2 train a stage-1 text-line detector
that crops each field before OCR.

Usage:
    uv run python -m thai_slip_copilot.render --out data/synth_v1 --count 5000
    uv run python -m thai_slip_copilot.render --out experiments/figures/samples --count 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from thai_slip_copilot.sampler import sample_slip
from thai_slip_copilot.schema import Slip

CANVAS_W = 720
CANVAS_H = 1280  # portrait-orientation phone screenshot ratio

# Font resolution order — we try bundled repo assets first, then fall back
# to macOS system Thai fonts. Document the preferred one in
# assets/fonts/README.md.
REPO_ROOT = Path(__file__).resolve().parents[2]
FONT_CANDIDATES = (
    REPO_ROOT / "assets/fonts/Sarabun-Regular.ttf",
    Path("/System/Library/Fonts/Supplemental/Ayuthaya.ttf"),
    Path("/System/Library/Fonts/Supplemental/SukhumvitSet.ttc"),
    Path("/System/Library/Fonts/ThonburiUI.ttc"),
    Path("/System/Library/Fonts/Supplemental/Silom.ttf"),
)
FONT_BOLD_CANDIDATES = (
    REPO_ROOT / "assets/fonts/Sarabun-Bold.ttf",
    *FONT_CANDIDATES,  # reuse regular if no bold available
)


def _resolve_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for p in (FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES):
        if p.is_file():
            return ImageFont.truetype(str(p), size)
    raise FileNotFoundError(
        "No Thai font found. Install Sarabun to assets/fonts/ — see "
        "assets/fonts/README.md for the download command."
    )


@dataclass
class FieldRow:
    """One labelled row on the slip: `label (Thai + English)` → `value`."""
    key: str                 # schema field key (matches bboxes dict)
    label: str               # displayed label text
    value: str               # displayed value text
    bold_value: bool = False


def _format_amount(slip: Slip) -> str:
    baht = slip.amount_satang / 100
    return f"{baht:,.2f} THB"


def _format_ts(slip: Slip) -> str:
    # Thai slips typically display Western-year dates in 24h time,
    # with a space before THB day-of-week abbreviation omitted.
    return slip.timestamp.strftime("%d %b %Y, %H:%M")


def _format_party(p) -> str:
    parts = []
    if p.name:
        parts.append(p.name)
    if p.account_masked:
        parts.append(p.account_masked)
    if p.phone:
        parts.append(p.phone)
    if p.bank and p.bank != "PROMPTPAY":
        parts.append(p.bank)
    return "  ·  ".join(parts) if parts else "-"


def _slip_rows(slip: Slip) -> list[FieldRow]:
    return [
        FieldRow("amount",       "จำนวนเงิน / Amount",       _format_amount(slip),     bold_value=True),
        FieldRow("timestamp",    "วันที่ / Date-Time",        _format_ts(slip)),
        FieldRow("sender",       "จาก / From",               _format_party(slip.sender)),
        FieldRow("receiver",     "ถึง / To",                 _format_party(slip.receiver)),
        FieldRow("reference_id", "เลขที่อ้างอิง / Reference", slip.reference_id or "-"),
        FieldRow("memo",         "หมายเหตุ / Memo",           slip.memo or "-"),
    ]


def render_slip(slip: Slip, rng: random.Random) -> tuple[Image.Image, dict[str, list[int]]]:
    """Return (PIL image, field → [x1,y1,x2,y2] bboxes in image pixel space)."""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), (245, 247, 250))
    d = ImageDraw.Draw(img)

    title_font = _resolve_font(48, bold=True)
    label_font = _resolve_font(22)
    value_font = _resolve_font(28)
    value_bold = _resolve_font(32, bold=True)

    # --- header band with the channel name ---
    header_colour = {
        "SCB":       (91, 32, 121),   # purple
        "KBANK":     (29, 137, 87),   # green
        "BBL":       (12, 47, 105),   # navy blue
        "KTB":       (24, 119, 203),  # blue
        "BAY":       (221, 163, 12),  # yellow/ochre
        "TMB":       (0, 100, 170),
        "GSB":       (219, 63, 94),
        "PROMPTPAY": (19, 101, 191),
        "OTHER":     (50, 50, 50),
    }[slip.channel]
    d.rectangle([0, 0, CANVAS_W, 180], fill=header_colour)
    d.text((40, 54), slip.channel.replace("_", " "), font=title_font, fill="white")
    d.text((40, 118), "สลิปโอนเงิน / Transfer Slip", font=label_font, fill=(255, 255, 255, 220))

    # --- body card ---
    card = [32, 220, CANVAS_W - 32, CANVAS_H - 120]
    d.rounded_rectangle(card, radius=22, fill="white",
                        outline=(225, 228, 234), width=2)

    # Amount gets its own prominent row at the top of the card
    bboxes: dict[str, list[int]] = {}
    y = card[1] + 32
    rows = _slip_rows(slip)

    # Split: first row (amount) rendered large and centered
    amount_row = rows[0]
    d.text((card[0] + 32, y), amount_row.label, font=label_font, fill=(110, 115, 125))
    y += 34
    amount_x = card[0] + 32
    amount_text = amount_row.value
    d.text((amount_x, y), amount_text, font=value_bold, fill=(20, 24, 31))
    tb = d.textbbox((amount_x, y), amount_text, font=value_bold)
    bboxes["amount"] = [int(v) for v in tb]
    y = tb[3] + 34

    # Divider
    d.line([card[0] + 32, y, card[2] - 32, y], fill=(232, 236, 240), width=2)
    y += 28

    # Remaining rows
    for row in rows[1:]:
        d.text((card[0] + 32, y), row.label, font=label_font, fill=(110, 115, 125))
        y += 28
        font = value_bold if row.bold_value else value_font
        vx, vy = card[0] + 32, y
        d.text((vx, vy), row.value, font=font, fill=(20, 24, 31))
        tb = d.textbbox((vx, vy), row.value, font=font)
        bboxes[row.key] = [int(v) for v in tb]
        y = tb[3] + 32

    # --- footer watermark / disclaimer ---
    d.text((40, CANVAS_H - 80),
           "Synthetic slip — generated by thai-slip-copilot",
           font=label_font, fill=(180, 185, 195))

    return img, bboxes


def generate(out_dir: Path, count: int, seed: int) -> None:
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    for i in range(count):
        slip = sample_slip(rng)
        img, bboxes = render_slip(slip, rng)
        stem = f"slip_{i:06d}"
        img.save(img_dir / f"{stem}.png", optimize=True)
        payload = {
            "slip": json.loads(slip.model_dump_json()),
            "bboxes": bboxes,
            "canvas": [CANVAS_W, CANVAS_H],
        }
        (lbl_dir / f"{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
    print(f"wrote {count} slips → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render synthetic Thai bank-transfer slips.")
    ap.add_argument("--out", type=Path, default=Path("data/synth_v1"))
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out, args.count, args.seed)


if __name__ == "__main__":
    main()
