"""Convert DuEE-fin -> 训练 jsonl(chat) + 评测 jsonl.

只保留三类事件：质押 / 股份回购 / 股东减持。一篇公告可能含多个事件（events 列表）。
含目标事件的样本才保留（跳过负样例/其他类型）。DuEE-fin 官方 test 无 gold，
故从带 gold 的 train(7015) 里自己切 train/test。

原始格式(每行一条):
  {"text":..., "event_list":[{"event_type":..,"arguments":[{"role":..,"argument":..}]}], ...}

输出(data/duee_fin_cn):
  {split}.train.jsonl : {"messages":[system,user,assistant]}
  {split}.eval.jsonl  : {"system","user","gt"}

Run:
  python -m shared.convert_duee_fin
  python -m shared.convert_duee_fin --sample   # 看一条转换结果
"""
from __future__ import annotations

import argparse
import json
import os
import random

from .schema import DuEEFinDoc, build_system_prompt

RAW = "data/duee_fin/DuEE-fin/duee_fin_train.json/duee_fin_train.json"
TARGET_TYPES = {"质押", "股份回购", "股东减持"}

# 角色名 -> schema 字段名（含 "/" 的转 "_"，其余同名）
ROLE_MAP = {"质押股票/股份数量": "质押股票_股份数量",
            "交易股票/股份数量": "交易股票_股份数量"}
VALID_FIELDS = set(DuEEFinDoc.model_fields["events"].annotation.__args__[0].model_fields)


def event_to_dict(ev: dict) -> dict:
    """单个事件 -> 拍平字段 dict（只保留 schema 里定义的字段）。"""
    d = {"event_type": ev["event_type"]}
    for a in ev.get("arguments", []):
        role = ROLE_MAP.get(a["role"], a["role"])
        if role in VALID_FIELDS and a.get("argument") not in (None, ""):
            d[role] = a["argument"]   # 同一 role 多值时保留最后一个（DuEE-fin 少见）
    return d


def build_gt(doc: dict) -> dict | None:
    """原始篇章 -> {events:[...]}，只含目标三类事件。无目标事件返回 None。"""
    evs = [event_to_dict(e) for e in (doc.get("event_list") or [])
           if e.get("event_type") in TARGET_TYPES]
    if not evs:
        return None
    return DuEEFinDoc.model_validate({"events": evs}).model_dump(exclude_none=True)


def build_record(doc: dict) -> dict | None:
    gt = build_gt(doc)
    text = (doc.get("text") or "").strip()
    if gt is None or not text:
        return None
    system = build_system_prompt("duee_fin", rich=True, types=True)
    assistant = json.dumps(gt, ensure_ascii=False)
    return {
        "system": system, "user": text, "gt": gt,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
            {"role": "assistant", "content": assistant},
        ],
    }


def write_split(records: list[dict], out_dir: str, split: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{split}.train.jsonl"), "w") as ft, \
         open(os.path.join(out_dir, f"{split}.eval.jsonl"), "w") as fe:
        for r in records:
            ft.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
            fe.write(json.dumps({"system": r["system"], "user": r["user"], "gt": r["gt"]},
                                ensure_ascii=False) + "\n")


def run(out_dir: str, test_frac: float) -> None:
    docs = [json.loads(l) for l in open(RAW) if l.strip()]
    records = [r for r in (build_record(d) for d in docs) if r]
    random.Random(42).shuffle(records)
    n_test = int(len(records) * test_frac)
    splits = {"test": records[:n_test], "val": records[n_test:2 * n_test],
              "train": records[2 * n_test:]}
    for name, recs in splits.items():
        write_split(recs, out_dir, name)
        print(f"[{name}] {len(recs)} 条 -> {out_dir}/{name}.*.jsonl")
    # 简单统计
    from collections import Counter
    c = Counter(e["event_type"] for r in records for e in r["gt"]["events"])
    print("事件类型分布:", dict(c))


def sample() -> None:
    for l in open(RAW):
        d = json.loads(l)
        rec = build_record(d)
        if rec and len(rec["gt"]["events"]) >= 1:
            print("=== INPUT (前200字) ===")
            print(rec["user"][:200])
            print("\n=== GT ===")
            print(json.dumps(rec["gt"], ensure_ascii=False, indent=2))
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/duee_fin_cn")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    if args.sample:
        sample()
    else:
        run(args.out, args.test_frac)


if __name__ == "__main__":
    main()
