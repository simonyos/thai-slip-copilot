"""Per-crop OCR over detector v2.1 outputs.

Design: the field detector localises ~8 regions per slip. For each
detected box, we crop the image with a small padding margin (OCR
engines dislike tight bboxes), run EasyOCR in Thai+English mode, and
concatenate the recognised lines back into one `raw_text` string per
detection. Downstream parsers (`date` → ISO, `amount` → satang, etc.)
consume these `raw_text` strings.

Why EasyOCR: maintained, solid Thai support, single pip install, CUDA
+ MPS + CPU all work. We only need the `th,en` model pair — total
download ~180 MB on first call, cached in `~/.EasyOCR/`.

Loader is lazily-constructed and process-global so repeated calls in
a script don't reload the ~1.5 GB of model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from thai_slip_copilot.fields import UNIFIED_CLASSES

if TYPE_CHECKING:
    import easyocr


@dataclass
class FieldOCR:
    class_name: str
    detector_conf: float
    bbox_xyxy: tuple[float, float, float, float]  # pixel coords in source image
    raw_text: str
    ocr_conf: float  # mean recogniser conf across lines in the crop


def _pick_ocr_device() -> bool:
    """EasyOCR accepts a boolean `gpu=` — True tries CUDA/MPS, False CPU.
    On Mac MPS, EasyOCR's dbnet detector sometimes falls back to CPU
    internally; that's fine, it still runs."""
    import torch

    return torch.cuda.is_available() or torch.backends.mps.is_available()


@lru_cache(maxsize=2)
def get_reader(langs: tuple[str, ...] = ("th", "en")) -> "easyocr.Reader":
    import easyocr

    gpu = _pick_ocr_device()
    return easyocr.Reader(list(langs), gpu=gpu, verbose=False)


# Per-class crop geometry: (pad_left, pad_right, pad_top, pad_bottom) as
# fractions of box width/height. The template detector boxes we trained
# are systematically too narrow on the x-axis for text-line classes
# (accnum/reference run right off the edge of the label). Rather than
# retrain, we widen the crop at OCR time.
#
# Also: pure-digit classes (amount/reference/accnum/promptpay) run
# through an English-only reader — the Thai-enabled reader hallucinates
# Thai chars on Latin alphanumeric strings.
_PAD_POLICY: dict[str, tuple[float, float, float, float]] = {
    # class           L     R     T     B
    "date":          (0.20, 0.15, 0.20, 0.20),
    "name":          (0.05, 0.15, 0.15, 0.20),
    "amount":        (0.05, 0.30, 0.25, 0.25),
    "accnum":        (0.05, 0.65, 0.30, 0.30),  # box captures left ~60% of text
    "reference":     (0.15, 0.65, 0.40, 0.40),  # trained box truncates text right
    "promptpay":     (0.05, 0.30, 0.25, 0.25),
    "bank":          (0.10, 0.10, 0.10, 0.10),  # logos — we skip OCR anyway
    "qr":            (0.00, 0.00, 0.00, 0.00),
}

_DIGIT_CLASSES = {"amount", "reference", "accnum", "promptpay"}


def crop_with_policy(img, xyxy: tuple[float, float, float, float], cls_name: str):
    W, H = img.size
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    pl, pr, pt, pb = _PAD_POLICY.get(cls_name, (0.05, 0.05, 0.05, 0.05))
    px1 = max(0, int(x1 - bw * pl))
    py1 = max(0, int(y1 - bh * pt))
    px2 = min(W, int(x2 + bw * pr))
    py2 = min(H, int(y2 + bh * pb))
    return img.crop((px1, py1, px2, py2))


def run_ocr_on_slip(
    image_path: str | Path,
    detector,  # ultralytics YOLO model
    conf_threshold: float = 0.25,
    skip_classes: tuple[str, ...] = ("qr", "bank"),
) -> list[FieldOCR]:
    """Run detector + EasyOCR on one slip image.

    Returns a list of FieldOCR, one per detection the detector fired
    above `conf_threshold`. `qr` and `bank` are skipped by default:
    QR payloads need a dedicated decoder, and `bank` just localises a
    logo (no text to extract).
    """
    import numpy as np
    from PIL import Image

    reader_th = get_reader(("th", "en"))
    reader_en = get_reader(("en",))
    img = Image.open(image_path).convert("RGB")

    det_results = list(
        detector.predict(source=str(image_path), conf=conf_threshold, verbose=False)
    )
    if not det_results:
        return []
    r = det_results[0]

    out: list[FieldOCR] = []
    for box in r.boxes:
        cls_id = int(box.cls.item())
        cls_name = UNIFIED_CLASSES[cls_id]
        if cls_name in skip_classes:
            continue
        det_conf = float(box.conf.item())
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())  # type: ignore[assignment]
        crop = crop_with_policy(img, xyxy, cls_name)  # type: ignore[arg-type]

        reader = reader_en if cls_name in _DIGIT_CLASSES else reader_th
        results = reader.readtext(np.array(crop))
        if not results:
            text, mean_conf = "", 0.0
        else:
            text = " ".join(line[1] for line in results)
            mean_conf = float(
                sum(line[2] for line in results) / max(len(results), 1)
            )
        out.append(
            FieldOCR(
                class_name=cls_name,
                detector_conf=det_conf,
                bbox_xyxy=xyxy,  # type: ignore[arg-type]
                raw_text=text.strip(),
                ocr_conf=mean_conf,
            )
        )
    return out
