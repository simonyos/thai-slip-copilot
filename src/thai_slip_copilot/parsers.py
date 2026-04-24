"""Typed parsers for raw OCR text.

Each parser takes a noisy OCR string and returns either the parsed
value in the canonical schema form, or None if the text can't be
parsed confidently. Parsers are defensive about the known OCR error
patterns on K+ slips (tone-mark variance in Thai names, O/0 confusion
in reference digits, trailing "บาท"/"U" on amounts, prefix junk from
logo bleed on accnum).
"""

from __future__ import annotations

import re
from datetime import datetime

# --- date -----------------------------------------------------------

# K+ Thai-month abbreviations (with the trailing dot) → month number.
# The short forms are the only ones K+ uses; full names aren't needed.
_TH_MONTHS: dict[str, int] = {
    "ม.ค.":  1,  # มกราคม
    "ก.พ.":  2,  # กุมภาพันธ์
    "มี.ค.": 3,  # มีนาคม
    "เม.ย.": 4,  # เมษายน
    "พ.ค.":  5,  # พฤษภาคม
    "มิ.ย.": 6,  # มิถุนายน
    "ก.ค.":  7,  # กรกฎาคม
    "ส.ค.":  8,  # สิงหาคม
    "ก.ย.":  9,  # กันยายน
    "ต.ค.":  10,  # ตุลาคม
    "พ.ย.":  11,  # พฤศจิกายน
    "ธ.ค.":  12,  # ธันวาคม
}

# "4 ส.ค. 68 11:55 น." — day, Thai-month abbrev, 2-digit Buddhist year, HH:MM, "น."
_DATE_RE = re.compile(
    r"(?P<day>\d{1,2})\s+"
    r"(?P<month>[ก-๙]+\.[ก-๙]+\.(?:[ก-๙]+\.)?)\s+"  # "ส.ค." or "มี.ค." etc
    r"(?P<year>\d{2})\s+"
    r"(?P<hh>\d{1,2}):(?P<mm>\d{2})"
)


def parse_date(raw: str) -> str | None:
    """Parse a K+ timestamp like '4 ส.ค. 68 11:55 น.' into an ISO string
    with a +07:00 (Bangkok) offset. Buddhist year is converted to
    Gregorian (BE − 543). Returns None if the shape doesn't match."""
    m = _DATE_RE.search(raw)
    if not m:
        return None
    month_key = m.group("month")
    if month_key not in _TH_MONTHS:
        return None
    day = int(m.group("day"))
    month = _TH_MONTHS[month_key]
    # OCR yields "68" → BE 2568 → CE 2025 (BE year = CE year + 543).
    year = (2500 + int(m.group("year"))) - 543
    hh = int(m.group("hh"))
    mm = int(m.group("mm"))
    try:
        dt = datetime(year, month, day, hh, mm)
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")


# --- amount ---------------------------------------------------------

# Amount OCR looks like "105.00 บาท", "105.00 U" (Eng-only reader reads
# บ as U), "1,172.50 บาท". We match a plain decimal with optional
# thousands-separator commas.
_AMOUNT_RE = re.compile(r"(?P<int>\d{1,3}(?:,\d{3})*|\d+)(?:\.(?P<frac>\d{1,2}))?")


def parse_amount_satang(raw: str) -> int | None:
    """Return amount in satang (1 THB = 100 satang). '105.00' → 10500."""
    m = _AMOUNT_RE.search(raw)
    if not m:
        return None
    int_part = m.group("int").replace(",", "")
    frac_part = (m.group("frac") or "").ljust(2, "0")[:2]
    try:
        return int(int_part) * 100 + int(frac_part)
    except ValueError:
        return None


# --- reference ------------------------------------------------------

# K+ reference format: 20 chars, alphanumeric, uppercase. Examples:
#   015216115552AQR06997
#   015330083642319492 (18 — top-up has a different length)
#   015279164956BPP02269
# So we don't fix the length, but strip whitespace and normalize the
# OCR's typical O↔0 confusion.
_REF_RE = re.compile(r"[A-Z0-9]+")


def parse_reference(raw: str) -> str | None:
    """Extract the K+ reference code — a long run of A-Z/0-9."""
    cleaned = raw.upper().replace("O", "0")  # OCR misreads 0 as O in digit spans
    # But we want to keep genuine alpha chars (AQR / BPP / BTF / APM etc.)
    # so we only normalise O→0 *within* all-digit runs. Do it carefully:
    # split the reference into the first numeric block (12 digits), then
    # an alpha block, then a trailing numeric block.
    m = re.search(r"\d{8,14}[A-Z]{2,5}\d{3,8}", cleaned)
    if m:
        return m.group(0)
    # Fallback: pure digit reference (top-up slips sometimes)
    m2 = re.search(r"\d{15,20}", cleaned)
    if m2:
        return m2.group(0)
    # Last resort: longest run of [A-Z0-9]
    candidates = _REF_RE.findall(raw.upper())
    if not candidates:
        return None
    best = max(candidates, key=len)
    return best if len(best) >= 10 else None


# --- accnum / promptpay --------------------------------------------

# accnum format K+ bank account: "xxx-x-xNNNN-x" (fully masked except
# 4 digits). Case-insensitive x. The trailing -x is optional because
# EasyOCR sometimes crops the final character (especially on slightly-
# right-truncated crops).
_ACCNUM_RE = re.compile(
    r"[xX]{3}\s*-\s*[xX]\s*-\s*[xX](?P<last4>\d{4})(?:\s*-\s*[xX])?"
)

# promptpay phone variant: "xxx-xxx-NNNN"
_PP_PHONE_RE = re.compile(r"[xX]{3}\s*-\s*[xX]{3}\s*-\s*(?P<last4>\d{4})")

# promptpay merchant-ID variant: 15-digit numeric
_PP_MERCHANT_RE = re.compile(r"(?<!\d)(\d{15})(?!\d)")


def parse_accnum(raw: str) -> str | None:
    """Return the normalised masked account "xxx-x-xNNNN-x" if the OCR
    string contains it. Tolerates OCR noise around the pattern."""
    m = _ACCNUM_RE.search(raw)
    if m:
        return f"xxx-x-x{m.group('last4')}-x"
    return None


def parse_promptpay(raw: str) -> str | None:
    """Return either 'xxx-xxx-NNNN' (phone) or the 15-digit merchant ID."""
    m = _PP_PHONE_RE.search(raw)
    if m:
        return f"xxx-xxx-{m.group('last4')}"
    m2 = _PP_MERCHANT_RE.search(raw.replace(" ", ""))
    if m2:
        return m2.group(1)
    return None


# --- name -----------------------------------------------------------

# K+ prefix tokens seen at the start of sender/receiver names:
#   นาย, นาง, น.ส., นางสาว, ด.ช., ด.ญ., นาย., บจก., บริษัท
# We keep the name as-is (trimming only whitespace) — downstream LLM
# or exact-match eval handles the noise. EasyOCR on Thai tone marks
# introduces ~5-15% character error rate on our slips.
def parse_name(raw: str) -> str | None:
    """Minimal normalisation: strip + collapse whitespace. Returns None
    on empty."""
    s = re.sub(r"\s+", " ", raw).strip()
    return s or None
