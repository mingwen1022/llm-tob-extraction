"""Domain-agnostic evaluator for structured extraction.

Three layers of metrics (report all):
  A) JSON validity   : parse-level + schema-level (report with/without constraint)
  B) field-level F1  : flatten -> normalize -> (field,value) multiset -> TP/FP/FN
                       reported as micro / macro / per-field
  C) latency         : measured by the caller, attached to the report

Method follows the CORD/Donut field-F1 convention: leaf (key, value) pairs as a
multiset; lists use the leaf field name WITHOUT index so order doesn't matter.

Usage (programmatic):
    from shared.eval import evaluate, build_comparison_table
    res = evaluate(raw_outputs, golds, domain="cord")
    print(res["report"])
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Optional

from .json_utils import safe_load
from .normalize import normalize
from .schema import get_model


# --------------------------------------------------------------------------- #
# flatten: nested json -> list[(field_path, normalized_value)]
# --------------------------------------------------------------------------- #
def flatten(obj, prefix: str = "") -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            pairs += flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:                      # NOTE: no index -> order-independent
            pairs += flatten(v, prefix)
    else:
        nv = normalize(prefix, obj)
        if nv is not None:                 # skip None / empty leaves
            pairs.append((prefix, nv))
    return pairs


# --------------------------------------------------------------------------- #
# layer A: JSON validity
# --------------------------------------------------------------------------- #
def validity(raw, model) -> tuple[Optional[dict], str]:
    """Return (parsed_obj_or_None, status) where status in
    {valid, schema_fail, parse_fail}. Accepts raw str or already-parsed dict."""
    if isinstance(raw, dict):
        obj, status = raw, "valid"
    else:
        obj, status = safe_load(raw)
    if status != "valid":
        return None, "parse_fail"
    try:
        model.model_validate(obj)
        return obj, "valid"
    except Exception:
        return obj, "schema_fail"   # parseable but off-schema; still scorable


# --------------------------------------------------------------------------- #
# layer B: field-level P/R/F1
# --------------------------------------------------------------------------- #
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def evaluate(raw_outputs, golds, domain: str = "cord", latency_s: Optional[float] = None) -> dict:
    """raw_outputs: list[str|dict] model predictions; golds: list[dict] gold JSON."""
    assert len(raw_outputs) == len(golds), "predictions / golds length mismatch"
    model = get_model(domain)

    n = len(golds)
    status_counts = Counter()                       # parse/schema/valid
    field_counts = defaultdict(lambda: [0, 0, 0])   # field -> [tp, fp, fn]

    for raw, gold in zip(raw_outputs, golds):
        obj, status = validity(raw, model)
        status_counts[status] += 1

        pred_pairs = Counter(flatten(obj)) if obj is not None else Counter()
        gold_pairs = Counter(flatten(gold))

        tp = pred_pairs & gold_pairs   # multiset intersection
        fp = pred_pairs - gold_pairs
        fn = gold_pairs - pred_pairs
        for (field, _), c in tp.items():
            field_counts[field][0] += c
        for (field, _), c in fp.items():
            field_counts[field][1] += c
        for (field, _), c in fn.items():
            field_counts[field][2] += c

    # micro (sum everything)
    TP = sum(v[0] for v in field_counts.values())
    FP = sum(v[1] for v in field_counts.values())
    FN = sum(v[2] for v in field_counts.values())
    micro_p, micro_r, micro_f1 = _prf(TP, FP, FN)

    # per-field + macro
    per_field = {}
    for field, (tp, fp, fn) in sorted(field_counts.items()):
        per_field[field] = dict(zip(("p", "r", "f1"), _prf(tp, fp, fn)))
        per_field[field].update(tp=tp, fp=fp, fn=fn)
    macro_f1 = (sum(d["f1"] for d in per_field.values()) / len(per_field)) if per_field else 0.0

    json_valid_rate = status_counts["valid"] / n if n else 0.0
    parseable_rate = (status_counts["valid"] + status_counts["schema_fail"]) / n if n else 0.0
    worst = min(per_field, key=lambda k: per_field[k]["f1"]) if per_field else None

    summary = {
        "n": n,
        "json_parseable_rate": round(parseable_rate, 4),
        "json_valid_rate": round(json_valid_rate, 4),
        "micro_p": round(micro_p, 4),
        "micro_r": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "worst_field": worst,
        "latency_s": latency_s,
        "status_counts": dict(status_counts),
        "per_field": per_field,
    }
    summary["report"] = _format_report(summary)
    return summary


def _format_report(s: dict) -> str:
    lines = [
        f"n={s['n']}  JSON parseable={s['json_parseable_rate']:.1%}  "
        f"schema-valid={s['json_valid_rate']:.1%}",
        f"micro  P={s['micro_p']:.3f}  R={s['micro_r']:.3f}  F1={s['micro_f1']:.3f}",
        f"macro  F1={s['macro_f1']:.3f}   worst field: {s['worst_field']}",
        "per-field:",
    ]
    for field, d in s["per_field"].items():
        lines.append(
            f"  {field:28} P={d['p']:.3f} R={d['r']:.3f} F1={d['f1']:.3f} "
            f"(tp={d['tp']} fp={d['fp']} fn={d['fn']})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3-way comparison table (E0..E4) -> markdown
# --------------------------------------------------------------------------- #
def build_comparison_table(results: dict[str, dict]) -> str:
    """results: {system_name: summary_dict}. Renders the resume-ready table."""
    head = "| 模型/实验 | JSON合法率 | micro-F1 | macro-F1 | 最差字段 | 延迟(s) |"
    sep = "|---|---|---|---|---|---|"
    rows = [head, sep]
    for name, s in results.items():
        lat = f"{s['latency_s']:.3f}" if s.get("latency_s") else "-"
        rows.append(
            f"| {name} | {s['json_valid_rate']:.1%} | {s['micro_f1']:.3f} | "
            f"{s['macro_f1']:.3f} | {s['worst_field']} | {lat} |"
        )
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# CLI: evaluate a predictions file against a gold file
# --------------------------------------------------------------------------- #
def _load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="jsonl, each line {'output': '<raw model text>'}")
    ap.add_argument("--gold", required=True, help="jsonl, each line {'gt': {...}}")
    ap.add_argument("--domain", default="cord")
    ap.add_argument("--name", default="run")
    args = ap.parse_args()

    preds = [r["output"] for r in _load_jsonl(args.pred)]
    golds = [r["gt"] for r in _load_jsonl(args.gold)]
    res = evaluate(preds, golds, domain=args.domain)
    print(res["report"])
    print("\n" + build_comparison_table({args.name: res}))


if __name__ == "__main__":
    main()
