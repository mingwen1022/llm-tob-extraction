"""Field value normalization.

Without this, "75,000" vs "75000" or "14/03/2015" vs "2015-03-14" get counted
as errors and F1 is artificially low. Normalization is the single biggest lever
on eval correctness.

Field *type* is inferred from the leaf field name (path), e.g. any path ending in
"price"/"amount"/"total" is treated as money; "date" as date.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime

# field-name heuristics ------------------------------------------------------ #
_MONEY_HINTS = ("price", "amount", "total", "subtotal", "tax", "change", "cash")
_DATE_HINTS = ("date", "time")


def _leaf(field_path: str) -> str:
    return field_path.split(".")[-1].lower()


def is_money_field(field_path: str) -> bool:
    leaf = _leaf(field_path)
    return any(h in leaf for h in _MONEY_HINTS)


def is_date_field(field_path: str) -> bool:
    leaf = _leaf(field_path)
    return any(h in leaf for h in _DATE_HINTS)


# value normalizers ---------------------------------------------------------- #
def _norm_width_space(s: str) -> str:
    # full-width -> half-width, collapse whitespace
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_money(s: str) -> str:
    # drop currency symbols / thousands separators / spaces; keep digits + . , -
    s = s.lower()
    s = re.sub(r"(rp|idr|usd|us\$|\$|¥|€|£|rmb|cny)", "", s)
    s = s.replace(" ", "")
    # remove thousands separators: 75,000 -> 75000 ; 75.000 -> 75000 (id locale)
    s = re.sub(r"(?<=\d)[,.](?=\d{3}\b)", "", s)
    s = s.strip(".,-")
    return s


def _norm_date(s: str) -> str:
    raw = _norm_width_space(s)
    fmts = (
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d",
        "%d.%m.%Y", "%m/%d/%Y", "%Y年%m月%d日", "%d %b %Y", "%b %d, %Y",
    )
    for f in fmts:
        try:
            return datetime.strptime(raw, f).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw.lower()  # fall back to cleaned string


def normalize(field_path: str, value) -> str | None:
    """Normalize a leaf value for comparison. None stays None (skipped in eval)."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    if is_money_field(field_path):
        return _norm_money(s)
    if is_date_field(field_path):
        return _norm_date(s)
    return _norm_width_space(s).lower()


if __name__ == "__main__":
    cases = [
        ("total.total_price", "364,100"),
        ("total.total_price", "364100"),
        ("sub_total.tax_price", "Rp 33.100"),
        ("date", "14/03/2015"),
        ("date", "2015-03-14"),
        ("menu.nm", "  Ice  Lemon Tea "),
    ]
    for f, v in cases:
        print(f"{f:24} {v!r:18} -> {normalize(f, v)!r}")
