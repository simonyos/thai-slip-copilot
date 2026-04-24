"""End-to-end: image → detector → OCR → parsed fields.

Glue between ocr.run_ocr_on_slip and parsers.*. Returns a flat dict
of canonical values keyed by class. Policies for multi-instance classes
(e.g. two `name` boxes for sender + receiver):
  - `name`: top-y-first (sender) and bottom-y (receiver), keyed as
    `sender_name` / `receiver_name`.
  - `bank`: same idea — top-right (channel) and leftmost (sender) split.
  - everything else: first detection wins (they're single-instance on K+).
"""

from __future__ import annotations

from pathlib import Path

from thai_slip_copilot.ocr import FieldOCR, run_ocr_on_slip
from thai_slip_copilot.parsers import (
    parse_accnum,
    parse_amount_satang,
    parse_date,
    parse_name,
    parse_promptpay,
    parse_reference,
)


def _sort_by_y(fs: list[FieldOCR]) -> list[FieldOCR]:
    return sorted(fs, key=lambda f: (f.bbox_xyxy[1] + f.bbox_xyxy[3]) / 2)


def extract_slip(image_path: str | Path, detector) -> dict:
    """Run the full pipeline on one image, return parsed fields."""
    ocr_fields = run_ocr_on_slip(image_path, detector)
    by_class: dict[str, list[FieldOCR]] = {}
    for f in ocr_fields:
        by_class.setdefault(f.class_name, []).append(f)

    out: dict = {
        "sender_name":     None,
        "receiver_name":   None,
        "sender_accnum":   None,
        "amount_satang":   None,
        "timestamp":       None,
        "reference_id":    None,
        "promptpay":       None,
        "raw_ocr":         {},
    }

    names_sorted = _sort_by_y(by_class.get("name", []))
    if names_sorted:
        out["sender_name"] = parse_name(names_sorted[0].raw_text)
    if len(names_sorted) >= 2:
        out["receiver_name"] = parse_name(names_sorted[-1].raw_text)

    if (accnums := by_class.get("accnum")):
        out["sender_accnum"] = parse_accnum(accnums[0].raw_text)
    if (amounts := by_class.get("amount")):
        out["amount_satang"] = parse_amount_satang(amounts[0].raw_text)
    if (dates := by_class.get("date")):
        out["timestamp"] = parse_date(dates[0].raw_text)
    if (refs := by_class.get("reference")):
        out["reference_id"] = parse_reference(refs[0].raw_text)
    if (pps := by_class.get("promptpay")):
        out["promptpay"] = parse_promptpay(pps[0].raw_text)

    out["raw_ocr"] = {
        cls: [f.raw_text for f in fs]
        for cls, fs in by_class.items()
    }
    return out
