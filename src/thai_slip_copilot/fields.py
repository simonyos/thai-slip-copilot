"""Unified field-detector class schema.

Two Roboflow source datasets label partially-overlapping sets of slip
regions. We merge them into one 8-class schema so a single YOLO field
detector can consume both. Source-specific classes that don't exist on
the other side just produce no bboxes on those images — YOLO handles
that gracefully as "this image has no instances of class X."

Mapping (unified_id: (unified_name, pipat_class, colamarc_class)):

    0: bank         ← pipat.Bank          (bank logo / channel indicator)
    1: name         ← pipat.Name          (account-holder name)
    2: amount       ← pipat.Price  | colamarc.amount
    3: qr           ← pipat.QR            (QR-code payload)
    4: date         ←                  colamarc.date
    5: accnum       ←                  colamarc.accnum
    6: promptpay    ←                  colamarc.pp1, colamarc.pp2
    7: reference    ←                  colamarc.transaction

Pipat's labels are slightly coarser (one "Name" class whether the name
belongs to sender or receiver); the downstream OCR + LLM stage infers
sender vs receiver from spatial ordering and label context in the
image, so we don't split it at the detector level.
"""

from __future__ import annotations

UNIFIED_CLASSES: tuple[str, ...] = (
    "bank",
    "name",
    "amount",
    "qr",
    "date",
    "accnum",
    "promptpay",
    "reference",
)

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

# Per-source class-id → unified-class-id remap tables.
PIPAT_REMAP: dict[int, int] = {
    0: CLASS_TO_ID["bank"],    # Bank
    1: CLASS_TO_ID["name"],    # Name
    2: CLASS_TO_ID["amount"],  # Price
    3: CLASS_TO_ID["qr"],      # QR
}

COLAMARC_REMAP: dict[int, int] = {
    0: CLASS_TO_ID["accnum"],     # accnum
    1: CLASS_TO_ID["amount"],     # amount
    2: CLASS_TO_ID["date"],       # date
    3: CLASS_TO_ID["promptpay"],  # pp1
    4: CLASS_TO_ID["promptpay"],  # pp2 (merge with pp1)
    5: CLASS_TO_ID["reference"],  # transaction
}

N_CLASSES = len(UNIFIED_CLASSES)
assert N_CLASSES == 8
