"""Convert CORD-v2 -> training jsonl (chat) + eval jsonl.

For each sample:
  INPUT  = receipt text reconstructed from valid_line words (reading order)
  GOLD   = gt_parse filtered to our Receipt schema (menu/sub_total/total subset)

Outputs (under --out, default data/cord):
  {split}.train.jsonl : {"messages": [system, user, assistant]}   <- for Unsloth
  {split}.eval.jsonl  : {"system":..., "user":..., "gt": {...}}    <- for eval/inference

Run:
  python -m shared.convert_cord --out data/cord
  python -m shared.convert_cord --sample          # offline 1-sample smoke test
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from .schema import Receipt, build_system_prompt

SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}

# fields we keep (must match schema.py)
MENU_KEYS = ("nm", "cnt", "price")
SUBTOTAL_KEYS = ("subtotal_price", "tax_price", "service_price")
TOTAL_KEYS = ("total_price", "cashprice", "changeprice")


# --------------------------------------------------------------------------- #
# 1) reconstruct reading-order text from valid_line words
# --------------------------------------------------------------------------- #
def _word_geom(word: dict) -> tuple[float, float, float]:
    """Return (left_x, center_y, height) from the word quad."""
    q = word.get("quad", {})
    xs = [q.get(f"x{i}", 0) for i in (1, 2, 3, 4)]
    ys = [q.get(f"y{i}", 0) for i in (1, 2, 3, 4)]
    return (min(xs), (min(ys) + max(ys)) / 2.0, max(ys) - min(ys))


def reconstruct_text(valid_line: list) -> str:
    """Reconstruct reading-order text: flatten all words, bin into visual rows by
    y (tolerance from median word height), order rows top->bottom, words left->right.
    This recovers "name  qty  price" on one line instead of splitting columns."""
    words = []  # (x, y, height, text)
    for line in valid_line:
        for w in line.get("words", []):
            t = str(w.get("text", "")).strip()
            if not t:
                continue
            x, y, h = _word_geom(w)
            words.append((x, y, h, t))
    if not words:
        return ""

    heights = sorted(w[2] for w in words)
    med_h = heights[len(heights) // 2] or 1.0
    tol = med_h * 0.6  # same-row if vertical centers within ~0.6 line height

    words.sort(key=lambda w: (w[1], w[0]))  # by y then x
    rows: list[list] = []
    row_y = None
    for x, y, h, t in words:
        if row_y is None or abs(y - row_y) > tol:
            rows.append([])
            row_y = y
        rows[-1].append((x, t))
    out_lines = []
    for row in rows:
        row.sort(key=lambda z: z[0])  # left -> right
        out_lines.append(" ".join(t for _, t in row))
    return "\n".join(out_lines)


# --------------------------------------------------------------------------- #
# 2) filter gt_parse -> our schema
# --------------------------------------------------------------------------- #
def _coerce_str(v):
    """CORD sometimes stores a repeated field as a list, e.g. ['49.636','49.636'].
    Collapse to a single string (first if all equal, else space-joined)."""
    if isinstance(v, list):
        vals = [str(x).strip() for x in v if x not in (None, "")]
        if not vals:
            return None
        return vals[0] if len(set(vals)) == 1 else " ".join(vals)
    return v


def _pick(d: dict, keys) -> dict:
    out = {}
    for k in keys:
        v = _coerce_str(d.get(k))
        if v not in (None, ""):
            out[k] = v
    return out


def filter_gt(gt_parse: dict) -> dict:
    out: dict = {}

    menu = gt_parse.get("menu")
    if isinstance(menu, dict):
        menu = [menu]
    if isinstance(menu, list):
        items = [_pick(m, MENU_KEYS) for m in menu if isinstance(m, dict)]
        items = [m for m in items if m]
        if items:
            out["menu"] = items

    sub = gt_parse.get("sub_total")
    if isinstance(sub, dict):
        sub = _pick(sub, SUBTOTAL_KEYS)
        if sub:
            out["sub_total"] = sub

    tot = gt_parse.get("total")
    if isinstance(tot, dict):
        tot = _pick(tot, TOTAL_KEYS)
        if tot:
            out["total"] = tot

    # validate-compatible shape (won't raise; Receipt allows all-optional)
    return Receipt.model_validate(out).model_dump(exclude_none=True)


# --------------------------------------------------------------------------- #
# 3) build a record
# --------------------------------------------------------------------------- #
def build_record(ground_truth: dict) -> Optional[dict]:
    valid_line = ground_truth.get("valid_line", [])
    gt_parse = ground_truth.get("gt_parse", {})
    text = reconstruct_text(valid_line)
    gt = filter_gt(gt_parse)
    if not text or not gt:
        return None
    system = build_system_prompt("cord")
    assistant = json.dumps(gt, ensure_ascii=False)
    return {
        "system": system,
        "user": text,
        "gt": gt,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
            {"role": "assistant", "content": assistant},
        ],
    }


# --------------------------------------------------------------------------- #
# 4) drivers
# --------------------------------------------------------------------------- #
def write_split(records: list[dict], out_dir: str, split: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{split}.train.jsonl"), "w") as ft, \
         open(os.path.join(out_dir, f"{split}.eval.jsonl"), "w") as fe:
        for r in records:
            ft.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
            fe.write(json.dumps({"system": r["system"], "user": r["user"], "gt": r["gt"]},
                                ensure_ascii=False) + "\n")


def run(out_dir: str, limit: Optional[int]) -> None:
    from datasets import load_dataset  # heavy import; only when really converting

    ds = load_dataset("naver-clova-ix/cord-v2")
    for hf_split, name in SPLIT_MAP.items():
        records = []
        rows = ds[hf_split]
        # we only need ground_truth; drop image column to avoid Pillow decode
        drop = [c for c in rows.column_names if c != "ground_truth"]
        if drop:
            rows = rows.remove_columns(drop)
        if limit:
            rows = rows.select(range(min(limit, len(rows))))
        for row in rows:
            gtruth = json.loads(row["ground_truth"])
            rec = build_record(gtruth)
            if rec:
                records.append(rec)
        write_split(records, out_dir, name)
        print(f"[{name}] wrote {len(records)} records -> {out_dir}/{name}.*.jsonl")


# --------------------------------------------------------------------------- #
# offline smoke sample (no download) — the Nasi Campur Bali receipt
# --------------------------------------------------------------------------- #
SAMPLE_GROUND_TRUTH = {
    "valid_line": [
        {"words": [{"quad": {"x1": 10, "y1": 10, "x2": 80, "y2": 10, "x3": 80, "y3": 22, "x4": 10, "y4": 22}, "text": "Nasi Campur Bali"},
                   {"quad": {"x1": 200, "y1": 10, "x2": 230, "y2": 10, "x3": 230, "y3": 22, "x4": 200, "y4": 22}, "text": "1 x"},
                   {"quad": {"x1": 300, "y1": 10, "x2": 360, "y2": 10, "x3": 360, "y3": 22, "x4": 300, "y4": 22}, "text": "75,000"}],
         "category": "menu.nm"},
        {"words": [{"quad": {"x1": 10, "y1": 30, "x2": 80, "y2": 30, "x3": 80, "y3": 42, "x4": 10, "y4": 42}, "text": "Ice Lemon Tea"},
                   {"quad": {"x1": 200, "y1": 30, "x2": 230, "y2": 30, "x3": 230, "y3": 42, "x4": 200, "y4": 42}, "text": "1 x"},
                   {"quad": {"x1": 300, "y1": 30, "x2": 360, "y2": 30, "x3": 360, "y3": 42, "x4": 300, "y4": 42}, "text": "24,000"}],
         "category": "menu.nm"},
        {"words": [{"quad": {"x1": 10, "y1": 60, "x2": 90, "y2": 60, "x3": 90, "y3": 72, "x4": 10, "y4": 72}, "text": "Subtotal"},
                   {"quad": {"x1": 300, "y1": 60, "x2": 360, "y2": 60, "x3": 360, "y3": 72, "x4": 300, "y4": 72}, "text": "99,000"}],
         "category": "sub_total.subtotal_price"},
        {"words": [{"quad": {"x1": 10, "y1": 80, "x2": 90, "y2": 80, "x3": 90, "y3": 92, "x4": 10, "y4": 92}, "text": "TOTAL"},
                   {"quad": {"x1": 300, "y1": 80, "x2": 370, "y2": 80, "x3": 370, "y3": 92, "x4": 300, "y4": 92}, "text": "108,900"}],
         "category": "total.total_price"},
    ],
    "gt_parse": {
        "menu": [
            {"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"},
            {"nm": "Ice Lemon Tea", "cnt": "1 x", "price": "24,000"},
        ],
        "sub_total": {"subtotal_price": "99,000", "tax_price": "9,900"},
        "total": {"total_price": "108,900"},
    },
}


def sample() -> None:
    rec = build_record(SAMPLE_GROUND_TRUTH)
    print("=== reconstructed INPUT text ===")
    print(rec["user"])
    print("\n=== GOLD (filtered gt_parse) ===")
    print(json.dumps(rec["gt"], ensure_ascii=False, indent=2))
    print("\n=== one training line (messages) ===")
    print(json.dumps({"messages": rec["messages"]}, ensure_ascii=False)[:400] + " ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cord")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per split (debug)")
    ap.add_argument("--sample", action="store_true", help="offline 1-sample smoke test")
    args = ap.parse_args()
    if args.sample:
        sample()
    else:
        run(args.out, args.limit)


if __name__ == "__main__":
    main()
