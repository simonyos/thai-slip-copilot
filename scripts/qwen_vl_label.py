"""Label K+ slip images with Qwen 2.5-VL.

Goal: a second-opinion / oracle labeler for the gold set. Given an
image, Qwen returns a structured JSON with every field's value AND
its bbox in pixel coords. We then diff against our detector+EasyOCR
pipeline to flag rows that need human verification.

Usage:
    uv run python scripts/qwen_vl_label.py \
        --images /path/to/phone_slips_raw \
        --out data/eval/qwen_predictions.jsonl \
        --model Qwen/Qwen2.5-VL-3B-Instruct \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


PROMPT = """You are given an image of a Thai bank transfer slip from the K+ app.
Extract the following fields and return them as a single JSON object.
For each field, include both the text value and its bounding box in the
image in pixel coordinates as [x1, y1, x2, y2].

Fields:
- "timestamp": the date and time of the transfer (keep the original Thai
  format exactly as shown, e.g. "4 ส.ค. 68 11:55 น.")
- "amount_thb": the transfer amount in THB as a string (e.g. "105.00",
  "4,000.00")
- "sender_name": sender's full name as printed
- "sender_accnum": sender's masked account (e.g. "xxx-x-x7829-x")
- "receiver_name": receiver's full name as printed
- "receiver_accnum": receiver's masked bank account if shown (same
  format as sender_accnum) OR null if the receiver uses PromptPay /
  merchant ID instead
- "promptpay": receiver's PromptPay phone (e.g. "xxx-xxx-9804") or a
  15-digit merchant PromptPay ID, OR null if the receiver uses a bank
  account instead
- "reference_id": the transaction reference code

If a field is genuinely not visible on the slip, set both its value
and bbox to null.

Respond with ONLY the JSON object, no prose, no markdown fences.
"""


def load_model_and_processor(model_id: str, quant: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs: dict = {"device_map": {"": 0}}  # force everything onto GPU 0
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif quant == "awq":
        # AWQ models are pre-quantized; the weights on disk ARE int4.
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16

    print(f"[vlm] loading {model_id}  quant={quant}…")
    t0 = time.time()
    # AutoModelForImageTextToText dispatches to the right class
    # (Qwen2_5_VL / Gemma4 / etc) based on the config.architectures entry.
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"[vlm] loaded in {time.time() - t0:.1f}s")
    return model, processor


def extract_fields(
    model, processor, image_path: Path,
    max_pixels: int | None, grayscale: bool,
) -> dict:
    from PIL import Image, ImageOps

    img = Image.open(image_path).convert("RGB")
    if grayscale:
        # Strip colour — sometimes helps (sometimes hurts) OCR fidelity
        # on slips with heavy background tinting. See smoke comparisons.
        img = ImageOps.grayscale(img).convert("RGB")
    if max_pixels:
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            img = img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    # Standard HF flow works for both Qwen 2.5-VL and Gemma 4 — the
    # chat template + processor handles image tokenisation internally.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
    )
    # Trim the prompt prefix
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {"raw": raw, "parsed": _try_parse_json(raw)}


def _try_parse_json(raw: str):
    """Qwen sometimes wraps output in ```json ... ``` — be lenient."""
    # Strip markdown fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    # Try to find the first balanced {...} block
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i, ch in enumerate(raw[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--quant", default="fp16", choices=("fp16", "4bit", "awq"),
                    help="fp16, 4bit (bitsandbytes), or awq (pre-quantized checkpoint)")
    ap.add_argument("--max-pixels", type=int, default=786432,
                    help="Cap vision-encoder input resolution (pixels). "
                         "768×1024 ≈ 786432. 0 disables.")
    ap.add_argument("--grayscale", action="store_true",
                    help="Collapse the input image to grayscale before "
                         "feeding to the vision encoder. Strips the K+ "
                         "teal tint; helps with fine Thai text fidelity.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images_dir = Path(args.images)
    paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.limit:
        paths = paths[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(args.model, args.quant)

    with out_path.open("w") as f:
        for i, p in enumerate(paths, 1):
            t0 = time.time()
            result = extract_fields(
                model, processor, p,
                max_pixels=args.max_pixels or None,
                grayscale=args.grayscale,
            )
            dt = time.time() - t0
            rec = {
                "image": p.name,
                "model": args.model,
                "raw": result["raw"],
                "parsed": result["parsed"],
                "seconds": round(dt, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            parsed_ok = "✓" if result["parsed"] is not None else "✗"
            print(f"[{i}/{len(paths)}] {p.name}  {dt:.1f}s  json={parsed_ok}")

    print(f"\nwrote {len(paths)} predictions → {out_path}")


if __name__ == "__main__":
    main()
