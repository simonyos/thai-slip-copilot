"""Parser unit tests.

Test strings come from actual EasyOCR output on the K+ phone slips —
the noisy prefixes and O/0 confusion are NOT synthetic, they're what
the model really emits. See scripts/ocr_smoke.py for the raw dumps
this was calibrated against.
"""

from __future__ import annotations

from thai_slip_copilot.parsers import (
    parse_accnum,
    parse_amount_satang,
    parse_date,
    parse_name,
    parse_promptpay,
    parse_reference,
)


class TestParseDate:
    def test_standard(self):
        assert parse_date("4 ส.ค. 68 11:55 น.") == "2025-08-04T11:55:00+07:00"

    def test_single_digit_hour(self):
        assert parse_date("6 ส.ค. 68 8:08 น.") == "2025-08-06T08:08:00+07:00"

    def test_december(self):
        assert parse_date("2 ธ.ค. 68 12:39 น.") == "2025-12-02T12:39:00+07:00"

    def test_march(self):
        # "มี.ค." — three-segment month abbrev
        assert parse_date("15 มี.ค. 69 09:30 น.") == "2026-03-15T09:30:00+07:00"

    def test_april(self):
        # "เม.ย." — three-segment with leading vowel
        assert parse_date("1 เม.ย. 69 17:00 น.") == "2026-04-01T17:00:00+07:00"

    def test_garbage_returns_none(self):
        assert parse_date("not a date") is None

    def test_invalid_month_returns_none(self):
        assert parse_date("4 ไม่.มี. 68 11:55 น.") is None


class TestParseAmountSatang:
    def test_plain(self):
        assert parse_amount_satang("105.00") == 10500

    def test_with_baht_suffix_ocr_as_U(self):
        assert parse_amount_satang("105.00 U") == 10500

    def test_with_baht(self):
        assert parse_amount_satang("105.00 บาท") == 10500

    def test_thousands_comma(self):
        assert parse_amount_satang("5,000.00 บาท") == 500000

    def test_integer_only(self):
        assert parse_amount_satang("50 บาท") == 5000

    def test_garbage(self):
        assert parse_amount_satang("abc") is None


class TestParseReference:
    def test_standard_20char(self):
        assert parse_reference("015216115552AQR06997") == "015216115552AQR06997"

    def test_with_ocr_prefix_junk(self):
        # The OCR usually emits exactly the reference, but defensively:
        assert parse_reference("... 015216115552AQR06997 ...") == "015216115552AQR06997"

    def test_o_to_zero_in_digit_block(self):
        # OCR read digit '0' as letter 'O' at the start
        assert parse_reference("O15279164956BPP02269") == "015279164956BPP02269"

    def test_pure_digit_topup(self):
        assert parse_reference("015330083642319492") == "015330083642319492"


class TestParseAccnum:
    def test_standard(self):
        assert parse_accnum("xxx-x-x7829-x") == "xxx-x-x7829-x"

    def test_with_ocr_prefix(self):
        assert parse_accnum("O.cisil8 XXX-X-X7829-X") == "xxx-x-x7829-x"

    def test_uppercase_x(self):
        assert parse_accnum("XXX-X-X4081-X") == "xxx-x-x4081-x"

    def test_no_match(self):
        assert parse_accnum("not an account") is None

    def test_truncated_trailing_x(self):
        # EasyOCR sometimes drops the trailing "-x" on right-truncated crops
        assert parse_accnum("XXX-X-X7829-") == "xxx-x-x7829-x"
        assert parse_accnum("0.7161713 Byd XXX-X-X7829-") == "xxx-x-x7829-x"


class TestParsePromptpay:
    def test_phone(self):
        assert parse_promptpay("xxx-xxx-9804") == "xxx-xxx-9804"

    def test_phone_with_junk(self):
        assert parse_promptpay("รหัสพร้อมเพย์ xxx-xxx-0018") == "xxx-xxx-0018"

    def test_merchant_id(self):
        assert parse_promptpay("202508042422984") == "202508042422984"

    def test_phone_beats_merchant(self):
        # If both appear, we want the phone pattern first (it's more
        # specific for personal PromptPay)
        assert parse_promptpay("xxx-xxx-1234 ... 999999999999999") == "xxx-xxx-1234"


class TestParseName:
    def test_strip_whitespace(self):
        assert parse_name("  นาย ซีโมน ย  ") == "นาย ซีโมน ย"

    def test_collapse_internal_spaces(self):
        assert parse_name("นาย   ซีโมน    ย") == "นาย ซีโมน ย"

    def test_empty_returns_none(self):
        assert parse_name("   ") is None
