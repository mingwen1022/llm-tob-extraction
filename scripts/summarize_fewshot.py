"""实验 B 汇总：把 runs/fewshot/ 下各 (模型 × shots) cell 打分并拼成对照表。

要堵的刀：「你给 API 喂了几个示例？」—— 原来的 E4 对照是零示例，而本地模型见过
800 条标注。0.945 vs 0.866 这个头条数字完全依赖这个对比是否公平。

口径：全部用 CORD 干净 92 条子集（--exclude-leaked-from），与既有 E4 完全一致。

成本：GMI 不通过 API 暴露单价，所以这里**只报 token 数**，不编价格。
真正可迁移的成本结论是 token 倍数（few-shot 把单条 input 推高多少倍），
这个数是自测的、与厂商报价无关。

用法：
  uv run python scripts/summarize_fewshot.py --out runs/e4_fewshot_summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.eval import evaluate

GOLD = "data/cord/test.eval.jsonl"
TRAIN = "data/cord/train.eval.jsonl"
LOCAL_LORA_F1 = 0.945          # E2 本地 4B LoRA，干净 92 条（runs/baseline_results.md）

# 各厂商**官方**挂牌价（USD / 百万 token），查询日期 2026-07-27。
# ⚠ 三点限制，用这些数字时必须一起讲：
#   1) 这是厂商官方直连价，本实验实际走的是 GMI 聚合平台，聚合价可能不同；
#   2) 大模型价格变动极快（DeepSeek 永久降 75%、GLM-5 上调、Qwen 挂 5 折促销），
#      下面标注了促销/永久，但结论只应看量级不应看小数点；
#   3) cache_in 是缓存命中价。few-shot 的示例前缀在所有调用间完全相同，
#      属于高度可缓存内容 —— 这一列直接决定 few-shot 的真实边际成本。
PRICING = {   # model: (input, output, cache_in, 备注)
    "google/gemini-3.5-flash":     (1.50, 9.00, 0.15,  "官方价"),
    "Qwen/Qwen3.7-Max":            (1.25, 3.75, 0.25,  "5折促销价(挂牌2.50/7.50)"),
    "deepseek-ai/DeepSeek-V4-Pro": (0.435, 0.87, 0.003625, "永久降价后"),
    "moonshotai/kimi-k3":          (3.00, 15.00, 0.30, "推理模式锁定，无非思考档"),
    "zai-org/GLM-5.2-FP8":         (1.40, 4.40, 0.26,  "官方价"),
    "MiniMaxAI/MiniMax-M3":        (0.30, 1.20, 0.06,  "≤512K输入档"),
    # 历史对照（本轮未重跑 few-shot）
    "moonshotai/Kimi-K2.6":        (None, None, None,  "上一代，本轮未测"),
    "claude-sonnet-5":             (None, None, None,  "本轮未测"),
}

# 历史 0-shot 全量 prompt 结果（同口径，直接复用原始预测文件重新打分，不硬编数字）
HISTORIC_0SHOT = {
    "google/gemini-3.5-flash": "runs/e4_gemini.jsonl",
    "Qwen/Qwen3.7-Max": "runs/e4_qwenmax.jsonl",
    "deepseek-ai/DeepSeek-V4-Pro": "runs/e4_deepseek.jsonl",
    "moonshotai/Kimi-K2.6": "runs/e4_kimi.jsonl",
    "claude-sonnet-5": "runs/e4_claude.jsonl",
}


def clean_idx() -> list[int]:
    train = {json.loads(l)["user"] for l in open(TRAIN) if l.strip()}
    gold = [json.loads(l) for l in open(GOLD) if l.strip()]
    return [i for i, g in enumerate(gold) if g["user"] not in train]


def score(path: str, keep: list[int]) -> dict | None:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    gold = [json.loads(l) for l in open(GOLD) if l.strip()]
    if len(rows) < len(gold):
        return None                                  # 未跑完
    preds = [rows[i]["output"] for i in keep]
    golds = [gold[i]["gt"] for i in keep]
    res = evaluate(preds, golds, domain="cord")
    tin = [rows[i].get("tok_in", 0) for i in keep]
    tout = [rows[i].get("tok_out", 0) for i in keep]
    res["tok_in"] = sum(tin) / len(tin) if any(tin) else None
    res["tok_out"] = sum(tout) / len(tout) if any(tout) else None
    res["n_err"] = sum(1 for i in keep if str(rows[i]["output"]).startswith("__ERROR__"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/e4_fewshot_summary.md")
    args = ap.parse_args()

    keep = clean_idx()
    cells: dict[tuple[str, int], dict] = {}

    for model, path in HISTORIC_0SHOT.items():
        if os.path.exists(path):
            r = score(path, keep)
            if r:
                cells[(model, 0)] = r

    for path in sorted(glob.glob("runs/fewshot/*.jsonl")):
        m = re.match(r"(.+)_s(\d+)\.jsonl$", os.path.basename(path))
        if not m:
            continue
        slug, shots = m.group(1), int(m.group(2))
        model = slug.replace("__", "/", 1).replace("_", ".")   # 反解 slug
        # slug 由 tr '/.' '__' 生成，反解不唯一 -> 用已知模型名匹配
        for known in ["google/gemini-3.5-flash", "Qwen/Qwen3.7-Max",
                      "deepseek-ai/DeepSeek-V4-Pro", "moonshotai/kimi-k3",
                      "zai-org/GLM-5.2-FP8", "MiniMaxAI/MiniMax-M3"]:
            if known.replace("/", "_").replace(".", "_") == slug:
                model = known
                break
        r = score(path, keep)
        if r:
            cells[(model, shots)] = r
        else:
            n = sum(1 for _ in open(path))
            print(f"[skip] {os.path.basename(path)} 未跑完 ({n}/100)")

    models = sorted({m for m, _ in cells})
    shots_all = sorted({s for _, s in cells})

    L: list[str] = []
    A = L.append
    A("# 实验 B · API few-shot 对照")
    A("")
    A(f"口径：CORD 干净 {len(keep)} 条子集（去掉官方 split 自带的 "
      f"{100-len(keep)} 条跨 split 重复），与既有 E4 完全一致。")
    A(f"对照基准：**本地 4B LoRA = {LOCAL_LORA_F1}**（同一子集）。")
    A("")
    A("要堵的刀：原 E4 对照里 API 是**零示例**，而本地模型见过 800 条标注。"
      "这一轮给 API 补上 in-context 示例，看差距还剩多少。")
    A("")

    A("## 主表：micro-F1")
    A("")
    A("| 模型 | " + " | ".join(f"{s}-shot" for s in shots_all) + " | 相对本地LoRA |")
    A("|---" * (len(shots_all) + 2) + "|")
    for mo in models:
        row = [mo]
        best = None
        for s in shots_all:
            c = cells.get((mo, s))
            if c:
                row.append(f"{c['micro_f1']:.3f}")
                best = c["micro_f1"] if best is None else max(best, c["micro_f1"])
            else:
                row.append("—")
        gap = f"{best - LOCAL_LORA_F1:+.3f}" if best is not None else "—"
        A("| " + " | ".join(row) + f" | {gap} |")
    A("")

    A("## schema 合法率")
    A("")
    A("| 模型 | " + " | ".join(f"{s}-shot" for s in shots_all) + " |")
    A("|---" * (len(shots_all) + 1) + "|")
    for mo in models:
        row = [mo]
        for s in shots_all:
            c = cells.get((mo, s))
            row.append(f"{c['json_valid_rate']:.0%}" if c else "—")
        A("| " + " | ".join(row) + " |")
    A("")

    A("## token 消耗与成本测算")
    A("")
    A("token 数是本实验实测；单价用**各厂商官方挂牌价**（查询日期 2026-07-27，见脚本内 PRICING）。")
    A("")
    A("三点限制必须一起讲：①实验实际走 GMI 聚合平台，聚合价与官方直连价可能不同；"
      "②大模型价格变动极快（DeepSeek 永久降 75%、GLM-5 上调、Qwen 现挂 5 折促销），"
      "**结论只看量级、不看小数点**；③下面「无缓存」是最坏情况，实际 few-shot 前缀完全相同、"
      "高度可缓存，真实成本落在两列之间。")
    A("")
    A("| 模型 | shots | 单条in tok | 单条out tok | 单价in/out(\\$/M) | 每千条·无缓存 | 每千条·前缀全命中缓存 |")
    A("|---|---|---|---|---|---|---|")
    for mo in models:
        p = PRICING.get(mo, (None, None, None, ""))
        for s in shots_all:
            c = cells.get((mo, s))
            if not c or c["tok_in"] is None:
                continue
            if p[0] is None:
                A(f"| {mo} | {s} | {c['tok_in']:.0f} | {c['tok_out']:.0f} | — | — | — |")
                continue
            pin, pout, pcache = p[0], p[1], p[2]
            # 无缓存：全部 input 按标准价
            cost = (c["tok_in"] * pin + c["tok_out"] * pout) / 1e6 * 1000
            # 全命中：few-shot 前缀按缓存价，其余按标准价。前缀量 = 该档 input - 0档 input
            base = cells.get((mo, 0))
            prefix = max(0.0, c["tok_in"] - (base["tok_in"] if base and base["tok_in"] else 0))
            cost_c = ((c["tok_in"] - prefix) * pin + prefix * pcache
                      + c["tok_out"] * pout) / 1e6 * 1000
            cc = f"\\${cost_c:.2f}" if s > 0 and prefix > 0 else "—"
            A(f"| {mo} | {s} | {c['tok_in']:.0f} | {c['tok_out']:.0f} "
              f"| {pin}/{pout} | \\${cost:.2f} | {cc} |")
    A("")

    # few-shot 相对 0-shot 的成本倍数
    A("### few-shot 的成本倍数（相对同模型 0-shot）")
    A("")
    A("| 模型 | " + " | ".join(f"{s}-shot" for s in shots_all if s > 0) + " |")
    A("|---" * (len([s for s in shots_all if s > 0]) + 1) + "|")
    for mo in models:
        p = PRICING.get(mo, (None,))
        base = cells.get((mo, 0))
        if p[0] is None or not base or not base["tok_in"]:
            continue
        b = (base["tok_in"] * p[0] + base["tok_out"] * p[1])
        row = [mo]
        for s in shots_all:
            if s == 0:
                continue
            c = cells.get((mo, s))
            if not c or c["tok_in"] is None or b == 0:
                row.append("—")
            else:
                row.append(f"{(c['tok_in']*p[0]+c['tok_out']*p[1])/b:.1f}×")
        A("| " + " | ".join(row) + " |")
    A("")
    A("这是喂给第 7.2 节盈亏平衡测算的关键输入：few-shot 买来的那几个点精度，"
      "代价是单条成本按倍数上升；而本地微调的边际推理成本不随调用量变化。"
      "两者的交叉点就是该不该自建的分界线。")
    A("")

    err = [(mo, s, c["n_err"]) for (mo, s), c in cells.items() if c["n_err"]]
    if err:
        A("## ⚠ 调用失败")
        A("")
        A("| 模型 | shots | 失败条数 |")
        A("|---|---|---|")
        for mo, s, n in sorted(err):
            A(f"| {mo} | {s} | {n} |")
        A("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
