"""Streamlit UI to hand-verify the 50-slip gold-transcription set.

Same flow as `thai-plate-synth/app/verify_labels.py` but adapted for
the multi-field slip schema: each row has 7+ fields to confirm, not a
single registration string.

Run:
    uv run streamlit run app/verify_slip_labels.py

Input: data/eval/gold_phone_v1.jsonl (pre-populated by
scripts/build_gold_set.py). The file is rewritten in-place on each
save, with the row's `verified` flag flipped to true and any
corrections persisted.

All edits stay on disk — the file is gitignored (PII).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import streamlit as st
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
GOLD_PATH = REPO / "data/eval/gold_phone_v1.jsonl"
IMAGES_DIR = Path("/Volumes/SM-EXT/Documents/Slip-orc-sample")

# Order matters: this is the visual field order in the form.
FIELDS: tuple[tuple[str, str, str], ...] = (
    # (key, label, help-hint)
    ("timestamp",       "Timestamp (ISO +07:00)", "e.g. 2025-08-04T11:55:00+07:00"),
    ("amount_satang",   "Amount (satang)",         "1 THB = 100 satang. ฿105.00 → 10500"),
    ("sender_name",     "Sender name",             "Exact text from the slip"),
    ("sender_accnum",   "Sender account",          "xxx-x-xNNNN-x format"),
    ("receiver_name",   "Receiver name",           "Exact text from the slip"),
    ("receiver_accnum", "Receiver account",        "xxx-x-xNNNN-x (bank account), else blank"),
    ("promptpay",       "PromptPay / merchant ID", "xxx-xxx-NNNN (phone) or 15-digit merchant, else blank"),
    ("reference_id",    "Reference ID",            "Verbatim, uppercase"),
)


def _load_rows() -> list[dict]:
    if not GOLD_PATH.is_file():
        return []
    return [
        json.loads(line)
        for line in GOLD_PATH.read_text().splitlines()
        if line.strip()
    ]


def _save_rows(rows: list[dict]) -> None:
    GOLD_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )


def _first_unverified(rows: list[dict]) -> int:
    for j, r in enumerate(rows):
        if not r.get("verified"):
            return j
    return 0


def _coerce(key: str, val: str) -> object:
    """Turn form strings back into the right types. Empty → None."""
    val = val.strip()
    if not val:
        return None
    if key == "amount_satang":
        try:
            return int(val.replace(",", "").replace(".", ""))
        except ValueError:
            return val
    return val


def main() -> None:
    st.set_page_config(page_title="Verify slip labels", layout="wide")
    st.title("Verify slip labels (K+ phone set)")

    rows = _load_rows()
    if not rows:
        st.error(f"No rows found at {GOLD_PATH}. Run `scripts/build_gold_set.py` first.")
        return

    if "idx" not in st.session_state:
        st.session_state.idx = _first_unverified(rows)

    total = len(rows)
    done = sum(1 for r in rows if r.get("verified"))
    i = st.session_state.idx
    rec = rows[i]
    img_path = IMAGES_DIR / rec["image"]

    # Header / progress
    cols = st.columns([3, 1, 1, 1])
    cols[0].progress(done / total, text=f"{done} / {total} verified")
    if cols[1].button("⬅ Prev", disabled=i == 0, use_container_width=True):
        st.session_state.idx = max(0, i - 1)
        st.rerun()
    if cols[2].button("Next ➡", disabled=i >= total - 1, use_container_width=True):
        st.session_state.idx = min(total - 1, i + 1)
        st.rerun()
    if cols[3].button("Jump to unverified", use_container_width=True):
        st.session_state.idx = _first_unverified(rows)
        st.rerun()

    status = "✅ verified" if rec.get("verified") else "⏳ unverified"
    st.markdown(f"**Slip {i + 1} / {total}** — `{rec['image']}` — {status}")

    # Image + form side by side
    img_col, form_col = st.columns([3, 2])
    with img_col:
        if img_path.is_file():
            st.image(Image.open(img_path).convert("RGB"), use_container_width=True)
        else:
            st.error(f"Image not found: {img_path}")

    with form_col:
        st.markdown("### Fields")
        edits: dict[str, str] = {}
        for key, label, hint in FIELDS:
            current = rec.get(key)
            default = "" if current is None else str(current)
            edits[key] = st.text_input(
                label, value=default, key=f"{rec['image']}__{key}", help=hint
            )
        notes = st.text_area("Notes", value=rec.get("notes", "") or "",
                             key=f"{rec['image']}__notes", height=80)

        st.markdown("---")
        action_cols = st.columns(3)
        save_verified = action_cols[0].button(
            "✓ Save & verify", type="primary", use_container_width=True,
            help="Commit edits, flip verified=true, advance to next unverified",
        )
        save_draft = action_cols[1].button(
            "💾 Save draft", use_container_width=True,
            help="Commit edits but keep verified=false",
        )
        unverify = action_cols[2].button(
            "↶ Un-verify", use_container_width=True, disabled=not rec.get("verified"),
            help="Flip verified back to false",
        )

    if save_verified or save_draft or unverify:
        for key, _, _ in FIELDS:
            rec[key] = _coerce(key, edits[key])
        rec["notes"] = notes.strip()
        if save_verified:
            rec["verified"] = True
            rec["verified_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        elif unverify:
            rec["verified"] = False
            rec.pop("verified_at", None)
        rows[i] = rec
        _save_rows(rows)

        if save_verified:
            # advance to next unverified
            for j in range(i + 1, total):
                if not rows[j].get("verified"):
                    st.session_state.idx = j
                    st.rerun()
                    return
            st.success("All rows verified! Run the eval script next.")
        st.rerun()

    # Footer — running completeness by field
    st.markdown("---")
    st.markdown("### Verified-field completeness")
    verified_rows = [r for r in rows if r.get("verified")]
    if not verified_rows:
        st.caption("No verified rows yet.")
    else:
        counts = []
        for key, label, _ in FIELDS:
            n = sum(1 for r in verified_rows if r.get(key) is not None)
            counts.append((label, n, len(verified_rows)))
        st.table({"field": [c[0] for c in counts],
                  "non-null / verified": [f"{c[1]} / {c[2]}" for c in counts]})


if __name__ == "__main__":
    main()
