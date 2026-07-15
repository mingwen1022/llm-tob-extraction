"""Convert CCKS2021 金融风控反欺诈案例 -> 训练 jsonl(chat) + 评测 jsonl.

原始格式(每行一条): {"text_id":..,"text":..,"level1":..,"level2":..,"level3":..,
                     "attributes":[{"type":..,"entity":..,"start":..,"end":..}]}
官方 eval 集(ccks_task1_eval_data.txt)无 gold，故从带 gold 的 train(5000条) 自己切分。

输出(data/ccks_fraud_cn):
  {split}.train.jsonl : {"messages":[system,user,assistant]}
  {split}.eval.jsonl  : {"system","user","gt"}

Run:
  python -m shared.convert_ccks
  python -m shared.convert_ccks --sample
"""
from __future__ import annotations

import argparse
import json
import os
import random

from .schema import FraudCase, build_system_prompt

RAW = "data/ccks_fin/ccks_task1_train.txt"
VALID_FIELDS = set(FraudCase.model_fields) - {"event_type"}


def build_gt(doc: dict) -> dict | None:
    event_type = "/".join(x for x in [doc.get("level1"), doc.get("level2"), doc.get("level3")] if x)
    gt: dict = {"event_type": event_type} if event_type else {}
    for a in doc.get("attributes", []):
        t, entity = a.get("type"), a.get("entity")
        if t in VALID_FIELDS and entity not in (None, ""):
            gt[t] = entity  # 同字段多值保留最后一个(与 convert_duee_fin 一致)
    if len(gt) <= 1:  # 只有 event_type、没有任何要素 -> 跳过(信号太弱)
        return None
    return FraudCase.model_validate(gt).model_dump(exclude_none=True)


def build_record(doc: dict) -> dict | None:
    gt = build_gt(doc)
    text = (doc.get("text") or "").strip()
    if gt is None or not text:
        return None
    system = build_system_prompt("ccks_fraud", rich=True, types=True)
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

    from collections import Counter
    c = Counter()
    for r in records:
        for k in r["gt"]:
            if k != "event_type":
                c[k] += 1
    print("字段分布:", dict(c.most_common()))


def sample() -> None:
    for l in open(RAW):
        d = json.loads(l)
        rec = build_record(d)
        if rec and len(rec["gt"]) >= 4:
            print("=== INPUT (前200字) ===")
            print(rec["user"][:200])
            print("\n=== GT ===")
            print(json.dumps(rec["gt"], ensure_ascii=False, indent=2))
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ccks_fraud_cn")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    if args.sample:
        sample()
    else:
        run(args.out, args.test_frac)


if __name__ == "__main__":
    main()
