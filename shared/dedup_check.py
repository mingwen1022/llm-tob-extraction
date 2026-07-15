"""Train/test 数据泄漏核查（精确文本重复）。

背景：CORD-v2 官方 split 里发现 train/test 之间有 8 条完全相同的原文（上游数据集
自带的重复条目，非本项目转换脚本引入）。虽然实测对分数影响很小（0.950→0.945），
但每接入一个新数据集（Phase 2: NDA/发票/CCKS）都应该先跑一遍这个核查，避免带着
虚高的分数不自知。

用法：
  python -m shared.dedup_check --train data/cord/train.eval.jsonl --test data/cord/test.eval.jsonl
  python -m shared.dedup_check --train data/cord/train.eval.jsonl --test data/cord/test.eval.jsonl --val data/cord/val.eval.jsonl
"""
from __future__ import annotations

import argparse
import json


def load_texts(path: str) -> list[str]:
    return [json.loads(l)["user"] for l in open(path) if l.strip()]


def report_overlap(name_a: str, texts_a: list[str], name_b: str, texts_b: list[str]) -> list[str]:
    set_a, set_b = set(texts_a), set(texts_b)
    overlap = set_a & set_b
    pct = len(overlap) / len(set_b) * 100 if set_b else 0
    print(f"{name_a} ∩ {name_b}: {len(overlap)} 条精确重复 "
          f"({pct:.1f}% of {name_b}, n={len(set_b)})")
    return list(overlap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--show", type=int, default=1, help="打印几条重复样本")
    args = ap.parse_args()

    train = load_texts(args.train)
    test = load_texts(args.test)

    overlap_tt = report_overlap("train", train, "test", test)
    if args.val:
        val = load_texts(args.val)
        report_overlap("train", train, "val", val)
        report_overlap("val", val, "test", test)

    if overlap_tt and args.show:
        print(f"\n重复样本示例（前 {args.show} 条）：")
        for t in overlap_tt[: args.show]:
            print("---")
            print(t[:200])

    if overlap_tt:
        print(f"\n⚠️  建议：报告分数时用去除这 {len(overlap_tt)} 条重叠后的干净子集分数，"
              f"或从 test 中剔除这些行。")
    else:
        print("\n✅ 无精确文本重叠，train/test 干净。")


if __name__ == "__main__":
    main()
