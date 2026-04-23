"""Random data sampler for synthetic Thai slip generation.

Generates plausible `Slip` records for the renderer to draw. Each field
follows the realistic distribution shape we observed when drafting the
schema — amounts biased to small transfers (food/transport), a realistic
mix of named vs masked accounts, PromptPay phone aliases for ~30% of
receivers, Thai/English name styles interleaved.

This is intentionally a separate module from the renderer so the same
sampler can feed the LLM-finetune dataset (`text-only` pipeline) without
pulling in PIL.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from thai_slip_copilot.schema import Bank, Category, Party, Slip

BANGKOK_TZ = timezone(timedelta(hours=7))

# Title prefix + given-name + family-name pool. Kept short so the
# sampler is deterministic under a seed; real breadth can come from
# an expanded word list later. Mix of Thai and transliterated-to-ASCII
# forms because slip UIs render both.
TH_TITLES = ("นาย", "นาง", "น.ส.")
EN_TITLES = ("MR.", "MRS.", "MS.")

TH_FIRSTS = (
    "สมชาย", "สมหญิง", "ปิยะ", "อรุณ", "นภา", "วิชัย", "ธนกฤต",
    "กนกวรรณ", "ปริยากร", "ปรีชา", "สมพร", "อภิสิทธิ์", "อรวรรณ",
    "นพรัตน์", "ศิริพร", "กิตติ", "พิมพ์ชนก", "จิรายุ", "ณัฐวดี",
)
TH_LASTS = (
    "ใจดี", "รักเรียน", "ศรีสุข", "สุขสวัสดิ์", "ทองดี", "บุญมา",
    "จันทร์เพ็ญ", "ประเสริฐ", "อินทร์แก้ว", "วงศ์สุวรรณ", "พรหมมา",
    "แสงทอง", "ศรีทอง", "ยิ้มเยือน", "เจริญสุข",
)
EN_FIRSTS = (
    "SOMCHAI", "PIYA", "NAPA", "SIMON", "NOPPADOL", "WICHAI", "KANOK",
    "PRIYA", "PREECHA", "ARWEE", "SOMPHON", "APHISIT", "SIRIPORN",
    "KITTI", "PIMCHANOK", "JIRAYU", "NATTAWADEE",
)
EN_LASTS = (
    "JAIDEE", "RAK-RIAN", "SRISUK", "SUKSAWAT", "TONGDEE", "BOONMAR",
    "CHANPHEN", "PRASERT", "INKAEW", "WONGSUWAN", "PROMMA",
    "SAENGTONG", "SRITHONG", "YIMYEN", "JAROENSUK",
)

BANKS: tuple[Bank, ...] = (
    "SCB", "KBANK", "BBL", "KTB", "BAY", "TMB", "GSB", "PROMPTPAY",
)

# Weights for the *channel* (the bank app that produced the slip):
# PromptPay interbank QR is the most common surface in the wild right now.
CHANNEL_WEIGHTS = (
    ("PROMPTPAY", 0.35),
    ("SCB", 0.18),
    ("KBANK", 0.17),
    ("BBL", 0.10),
    ("KTB", 0.08),
    ("BAY", 0.06),
    ("TMB", 0.04),
    ("GSB", 0.02),
)

# Memos biased toward common everyday transfer purposes.
MEMOS = (
    None, None, None,  # ~30% of slips have no memo
    "ค่าอาหาร", "ค่าข้าว", "อาหารกลางวัน", "ค่ากาแฟ",
    "ค่าเช่า", "ค่าน้ำค่าไฟ", "ค่าอินเทอร์เน็ต",
    "เงินเดือน", "โบนัส", "คืนเงิน", "แชร์ค่าแท็กซี่",
    "ชำระค่าสินค้า", "ช้อปปี้ออเดอร์", "Lazada order",
    "coffee ☕", "lunch", "rent march", "refund",
)

# Category priors given a memo text — kept simple for sampling, the
# downstream classifier will learn a better mapping.
_CATEGORY_BY_KEYWORD: tuple[tuple[str, Category], ...] = (
    ("อาหาร", "food"), ("ข้าว", "food"), ("lunch", "food"),
    ("กาแฟ", "food"), ("coffee", "food"),
    ("เช่า", "rent"), ("rent", "rent"),
    ("น้ำ", "utilities"), ("ไฟ", "utilities"),
    ("อินเทอร์เน็ต", "utilities"), ("internet", "utilities"),
    ("เงินเดือน", "salary"), ("โบนัส", "salary"),
    ("คืนเงิน", "refund"), ("refund", "refund"),
    ("ช้อป", "shopping"), ("lazada", "shopping"), ("shopee", "shopping"),
    ("แท็กซี่", "transport"), ("grab", "transport"),
)


def _weighted_choice(rng: random.Random, pairs: tuple[tuple[str, float], ...]) -> str:
    choices, weights = zip(*pairs, strict=True)
    return rng.choices(choices, weights=weights, k=1)[0]


def _masked_account(rng: random.Random) -> str:
    """Render a bank account number with Thai-slip-style masking."""
    #   xxx-x-xNNNN-x   (visible middle 4 digits)
    visible = "".join(str(rng.randint(0, 9)) for _ in range(4))
    return f"xxx-x-x{visible}-x"


def _masked_phone(rng: random.Random) -> str:
    """PromptPay phone alias with Thai-slip masking — last 4 digits visible."""
    tail = "".join(str(rng.randint(0, 9)) for _ in range(4))
    return f"xxx-xxx-{tail}"


def _party(rng: random.Random, use_phone_alias: bool = False) -> Party:
    en_style = rng.random() < 0.55  # Thai UIs default to ASCII for farang names
    if en_style:
        title = rng.choice(EN_TITLES)
        name = f"{title} {rng.choice(EN_FIRSTS)} {rng.choice(EN_LASTS)[0]}."
    else:
        title = rng.choice(TH_TITLES)
        name = f"{title}{rng.choice(TH_FIRSTS)} {rng.choice(TH_LASTS)}"

    if use_phone_alias:
        return Party(
            name=name,
            phone=_masked_phone(rng),
            bank="PROMPTPAY",
        )
    return Party(
        name=name,
        account_masked=_masked_account(rng),
        bank=rng.choice(BANKS),
    )


def _amount_satang(rng: random.Random) -> int:
    """Log-uniform amount distribution, ฿20 → ฿50,000, quantised to ฿1.00."""
    log_min, log_max = 1.3, 4.7  # log10(20)..log10(50000)
    baht = int(round(10 ** rng.uniform(log_min, log_max)))
    return baht * 100


def _timestamp(rng: random.Random) -> datetime:
    # Random moment in the last 180 days in Bangkok time
    now = datetime(2026, 4, 24, tzinfo=BANGKOK_TZ)
    delta_s = rng.randint(0, 180 * 86400)
    return now - timedelta(seconds=delta_s)


def _reference_id(rng: random.Random, ts: datetime) -> str:
    # Bank-style: YYYYMMDDHHmm + 4 hex chars. Different banks use
    # different formats; this is a plausible superset for synth.
    hex_tail = "".join(rng.choices("0123456789ABCDEF", k=4))
    return f"{ts.strftime('%Y%m%d%H%M')}{hex_tail}"


def _memo(rng: random.Random) -> str | None:
    return rng.choice(MEMOS)


def _category_from_memo(memo: str | None) -> Category | None:
    if not memo:
        return "transfer"
    memo_low = memo.lower()
    for kw, cat in _CATEGORY_BY_KEYWORD:
        if kw in memo_low:
            return cat
    return "other"


def sample_slip(rng: random.Random) -> Slip:
    """One random Slip record. Deterministic under a seeded `rng`."""
    channel = _weighted_choice(rng, CHANNEL_WEIGHTS)
    ts = _timestamp(rng)

    # Receivers on PromptPay are ~50% phone aliases; on bank apps ~15%.
    phone_alias_prob = 0.5 if channel == "PROMPTPAY" else 0.15
    receiver = _party(rng, use_phone_alias=rng.random() < phone_alias_prob)
    sender = _party(rng, use_phone_alias=False)
    if channel != "PROMPTPAY":
        sender.bank = channel  # sender bank matches the app that produced the slip

    memo = _memo(rng)

    return Slip(
        amount_satang=_amount_satang(rng),
        timestamp=ts,
        reference_id=_reference_id(rng, ts) if rng.random() < 0.85 else None,
        sender=sender,
        receiver=receiver,
        channel=channel,
        memo=memo,
        category=_category_from_memo(memo),
    )
