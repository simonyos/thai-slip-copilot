# Thai Slip Copilot — Mobile-Deployable Slip Parser

End-to-end pipeline that reads a Thai bank transfer slip (PromptPay / SCB / KBank /
BBL / KTB / Krungsri …) and returns a strict structured record:

```json
{
  "amount_satang": 150000,
  "currency": "THB",
  "timestamp": "2026-04-12T13:42:00+07:00",
  "reference_id": "202604121342A1234",
  "sender":   { "name": "MR. SIMON Y.",  "account_masked": "xxx-x-x1234-x", "bank": "SCB" },
  "receiver": { "name": "MRS. NAPA K.",  "account_masked": "xxx-x-x5678-x", "bank": "KBANK" },
  "channel": "SCB",
  "memo": "ค่าอาหาร",
  "category": "food"
}
```

The project pairs a lightweight OCR stage with a **fine-tuned small LLM**
(Gemma 3 270M primary, Qwen 2.5-1.5B as an ablation baseline) whose only
job is OCR-text → JSON structured extraction + category classification.

## Why

Thai fintech and SME tools routinely need to parse transfer slips — for
reconciliation, expense tracking, KYC. Existing solutions are either
(a) hand-crafted regex that breaks across bank template updates, or
(b) closed commercial APIs at ฿1–5 per call. A fine-tuned sub-500M LLM
that runs *on-device* sidesteps both: no per-call cost, no PII leaving
the phone, works offline.

## Mobile-first by design

Model sizing is the headline architectural decision:

| Model | fp16 | INT4 | Fits on |
|---|---:|---:|---|
| **Gemma 3 270M** *(primary)* | ~540 MB | ~170 MB | any phone from 2021+ |
| Qwen 2.5-1.5B *(ablation)* | ~3 GB | ~1 GB | flagship only |

Gemma 3 270M runs comfortably on-device via MediaPipe LLM Inference
(Android), Core ML / MLX (iOS), or `llama.rn` cross-platform. The
fine-tuned model weights ship with the app; no server round-trip.

## Plan

| Weekend | Deliverable |
|---|---|
| 1 | Synthetic slip renderer — 5+ bank templates, per-field bboxes, structured ground truth JSON |
| 2 | Field detector — scrape real Thai slips from Roboflow, merge into 8-class schema, train YOLOv8n, eval on own K+ phone slips, iterate |
| 3 | OCR stage — Thai text recognition on the detected field crops (Tesseract Thai / EasyOCR baseline, text → typed field value) |
| 4 | Fine-tune Gemma 3 270M on synth `OCR text → Slip JSON` pairs; eval exact-field accuracy on real-slip gold set |
| 5 | Ablation: Gemma 3 270M vs Qwen 2.5-1.5B vs regex-template baseline; category classifier head |
| 6 | Streamlit demo + FastAPI endpoint + HF Space deploy; GGUF export for mobile |
| 7 | Mobile demo — React Native app with `llama.rn`, on-device inference; record the demo |

Each weekend produces something shippable on its own; the final resume
bullet is the mobile demo + the rigorous FT vs baseline ablation.

## Schema

The structured output is defined once in
[`src/thai_slip_copilot/schema.py`](src/thai_slip_copilot/schema.py) and
is the single source of truth for:

- the synth renderer's ground truth JSON,
- the fine-tune target format,
- the FastAPI response schema,
- the Streamlit demo's display columns.

See the schema file for field-level documentation (PII handling, currency
unit conventions, timezone policy).

## Privacy

Real slips contain PII — full account numbers, names, phone numbers.
The repository never commits raw real images (`.gitignore` excludes
`data/real/` and all `*.jpg`/`*.png` under `data/`). The hand-verify
workflow redacts account numbers to `xxx-x-xNNNN-x` masking before
storing any label JSONL. The fine-tuned model operates on slip images
that never leave the device.

## Setup

```bash
# 1. Install
make setup

# 2. Generate 10 sample slips to eyeball the renderer
make sample

# 3. Generate 5,000 synth slips (~5 min on CPU)
make synth

# 4. Fine-tune Gemma 3 270M (~1 hour on a 3060 Ti)
make finetune

# 5. Run the Streamlit demo
make demo
```

## Project status

- ✅ Scaffold — schema, repo structure, README, mobile-deployment plan
- ✅ Weekend 1 — synth slip renderer
- ✅ Weekend 2 — field detector
  - Roboflow scrape (pipat + colamarc) merged into unified 8-class YOLO schema (674 train / 74 val)
  - YOLOv8n trained to 0.995 val mAP50 (detector v1) — but collapsed on real K+ phone slips for 4 of 8 classes due to label incompleteness across the merged sources
  - Self-labeled 377 own K+ phone slips via a fixed template + amount-anchored reference; retrained (detector v2.1) on the merged 1,014-image set
  - 7 of 8 classes at 91.5–100% coverage on real phone slips at 0.85+ confidence; `promptpay` deferred to a variant-aware labeling pass once OCR is wired up
- 🚧 Weekend 3 — OCR on the detected crops

## License

Code: MIT — see [LICENSE](LICENSE).
Model weights (fine-tuned Gemma 3 270M): subject to
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

## Related

Sibling project [`thai-plate-synth`](https://github.com/simonyos/thai-plate-synth) —
same synth-data-first discipline applied to Thai license-plate OCR. The
weekend-7 lesson from that project (_clean-bbox distillation hits
0.721 char-acc; post-processing can't recover a weak recognizer_)
directly shapes this project's architectural bet on fine-tuning over
post-hoc rules.
