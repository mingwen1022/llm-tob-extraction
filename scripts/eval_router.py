"""评测路由器：从各域 test 集抽样混合，测零样本分类准确率 + 混淆矩阵。

用法：
  uv run python scripts/eval_router.py --n-per-domain 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.router import build_router_prompt, parse_route_output
from shared.schema import SCHEMA_REGISTRY

HOST = "http://localhost:11434"

DOMAIN_FILES = {
    "cord": "data/cord/test.eval.jsonl",
    "duee_fin": "data/duee_fin_cn/test.eval.jsonl",
    "ccks_fraud": "data/ccks_fraud_cn/test.eval.jsonl",
}


def chat(model: str, messages: list[dict], timeout=60) -> str:
    payload = {"model": model, "messages": messages, "stream": False,
               "think": False, "options": {"temperature": 0}}
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--n-per-domain", type=int, default=30)
    args = ap.parse_args()

    router_system = build_router_prompt()
    samples = []  # (true_domain, text)
    for domain, path in DOMAIN_FILES.items():
        rows = [json.loads(l) for l in open(path) if l.strip()][: args.n_per_domain]
        for r in rows:
            samples.append((domain, r["user"]))

    print(f"路由 prompt:\n{router_system}\n")
    print(f"混合样本: {len(samples)} 条（每域最多 {args.n_per_domain} 条）\n")

    confusion = Counter()  # (true, pred) -> count
    errors = []
    t0 = time.time()
    for i, (true_domain, text) in enumerate(samples):
        messages = [
            {"role": "system", "content": router_system},
            {"role": "user", "content": text[:1500]},  # 路由不需要全文，截断提速
        ]
        try:
            raw = chat(args.model, messages)
            pred = parse_route_output(raw)
        except Exception as e:
            pred = None
            raw = f"__ERROR__ {e}"
        confusion[(true_domain, pred)] += 1
        if pred != true_domain:
            errors.append((true_domain, pred, raw, text[:60]))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}  ({(time.time()-t0)/(i+1):.2f}s/条)")

    print(f"\n耗时 {time.time()-t0:.1f}s\n")

    # 混淆矩阵
    domains = list(SCHEMA_REGISTRY.keys())
    print("=== 混淆矩阵 (行=真实, 列=预测) ===")
    header = "真实\\预测".ljust(12) + "".join(d.ljust(12) for d in domains) + "其他/失败"
    print(header)
    total_correct = total = 0
    for true_d in domains:
        row_total = sum(v for (t, p), v in confusion.items() if t == true_d)
        row = [true_d.ljust(12)]
        for pred_d in domains:
            c = confusion.get((true_d, pred_d), 0)
            row.append(str(c).ljust(12))
            if true_d == pred_d:
                total_correct += c
        other = row_total - sum(confusion.get((true_d, d), 0) for d in domains)
        row.append(str(other))
        total += row_total
        print("".join(row))

    acc = total_correct / total if total else 0
    print(f"\n路由准确率: {total_correct}/{total} = {acc:.1%}")

    if errors:
        print(f"\n=== 路由错误样本 ({len(errors)}条) ===")
        for true_d, pred_d, raw, text in errors[:10]:
            print(f"  真实={true_d} 预测={pred_d} 原始输出={raw!r}")
            print(f"    文本: {text}")


if __name__ == "__main__":
    main()
