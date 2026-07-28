"""实验 A：域外行为基线测量。

**这个实验不测「会不会出错」** —— 当前架构没有拒答能力，域外文档一定会被强行分到
三类之一并抽出东西，这个结论不用验证。它测的是：要给它加拒答，最便宜的那条路
（抽取后按字段填充率做后置判断）走不走得通、阈值该定在哪、域内会误伤多少。

产出是一条基准线。**本轮不实现任何拦截逻辑，不改现有脚本，只测现状。**

要回答的三个问题：
  Q1 域外文档抽出的 JSON，schema 合法率是多少（高 => 失败是静默的）
  Q2 域外与域内的字段填充率分布分不分得开
  Q3 若分得开，阈值定在哪、域内误伤多少

⚠ 阈值必须**按域**定，不能用全局阈值。三个域的域内填充率量纲差异极大
（实测中位：cord 0.778 / duee_fin 0.295 / ccks_fraud 0.444）——duee_fin 域内
中位比 cord 域内最小值还低，因为它的 schema 是三类事件字段的并集(22个)，
单个事件天然只能填其中一类。用一个全局阈值必然要么误杀 duee_fin，要么拦不住任何东西。
本脚本同时输出全局扫描（用于说明它为什么不可行）与按域扫描（实际可用的那个）。

用法：
  uv run python scripts/probe_ood.py --ood data/ood_probe/probe.jsonl \\
      --indomain-n 30 --out runs/ood_probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import typing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydantic import BaseModel

from run_orchestrator import Orchestrator, DOMAIN_FILES
from shared.json_utils import safe_load
from shared.schema import SCHEMA_REGISTRY, get_model

DOMAINS = list(SCHEMA_REGISTRY.keys())


# --------------------------------------------------------------------------- #
# 字段填充率
#
# 口径（域外与域内必须完全一致，否则 Q2 无意义）：
#   分母 = 固定叶子字段数 + Σ(列表实际元素数 × 每元素叶子字段数)
#   分子 = 上述位置中值非空(非 None/""/[]/{})的个数
#   列表型按**实际输出的元素**展开计数（清单原文要求）
#
# 边界情况：duee_fin 的 schema 只有 events 一个列表字段、没有固定叶子。
# 若模型输出 events=[]，分母为 0。此时记 fill_rate=0.0 —— 「什么都没抽出来」
# 在拒答判断里语义上就等价于填充率最低，这也正是希望被拦下的信号。
# --------------------------------------------------------------------------- #
def leaf_spec(model: type[BaseModel]) -> tuple[list[str], dict[str, list[str]]]:
    """返回 (固定叶子路径列表, {列表字段名: 每元素叶子字段名列表})"""
    fixed: list[str] = []
    per_elem: dict[str, list[str]] = {}
    for name, f in model.model_fields.items():
        ann = f.annotation
        inner = None
        for arg in (ann,) + typing.get_args(ann):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                inner = arg
                break
            for sub in typing.get_args(arg):
                if isinstance(sub, type) and issubclass(sub, BaseModel):
                    inner = sub
                    break
            if inner:
                break
        if inner is not None and typing.get_origin(ann) is list:
            per_elem[name] = list(inner.model_fields)
        elif inner is not None:
            fixed.extend(f"{name}.{k}" for k in inner.model_fields)
        else:
            fixed.append(name)
    return fixed, per_elem


def _nonempty(v) -> bool:
    return v not in (None, "", [], {})


def fill_rate(obj: dict, model: type[BaseModel]) -> tuple[int, int, float]:
    fixed, per_elem = leaf_spec(model)
    filled = total = 0
    for path in fixed:
        total += 1
        cur = obj
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if _nonempty(cur):
            filled += 1
    for lname, subfields in per_elem.items():
        items = obj.get(lname) if isinstance(obj, dict) else None
        if not isinstance(items, list):
            items = []
        for it in items:
            for sf in subfields:
                total += 1
                if _nonempty(it.get(sf) if isinstance(it, dict) else None):
                    filled += 1
    return filled, total, (filled / total if total else 0.0)


# --------------------------------------------------------------------------- #
# 跑一条
# --------------------------------------------------------------------------- #
def probe_one(orch: Orchestrator, text: str) -> dict:
    t0 = time.time()
    routed = orch.route(text)
    t_route = time.time() - t0
    rec = {"routed_to": routed, "route_s": round(t_route, 2)}
    if routed is None:                      # 当前实现下预期为 0 次
        rec.update(parse_ok=False, schema_valid=False, filled=0, total=0,
                   fill_rate=0.0, extract_s=0.0, raw=None)
        return rec
    t1 = time.time()
    raw = orch.extract(text, routed)
    rec["extract_s"] = round(time.time() - t1, 2)
    obj, status = safe_load(raw)
    rec["parse_ok"] = status == "valid" and obj is not None
    model = get_model(routed)
    if rec["parse_ok"]:
        try:
            model.model_validate(obj)
            rec["schema_valid"] = True
        except Exception:
            rec["schema_valid"] = False
        f, t, r = fill_rate(obj, model)
    else:
        rec["schema_valid"] = False
        f = t = 0
        r = 0.0
    rec.update(filled=f, total=t, fill_rate=round(r, 4), raw=raw[:2000])
    return rec


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def five_num(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) < 4:
        return f"min={min(vals):.3f} 中位={statistics.median(vals):.3f} max={max(vals):.3f} (n={len(vals)})"
    q = statistics.quantiles(vals, n=4)
    return (f"min={min(vals):.3f} p25={q[0]:.3f} 中位={q[1]:.3f} "
            f"p75={q[2]:.3f} max={max(vals):.3f}")


def build_summary(rows: list[dict]) -> str:
    ood = [r for r in rows if r["group"] == "ood"]
    ind = [r for r in rows if r["group"] == "indomain"]
    L: list[str] = []
    A = L.append

    A("# 实验 A · 域外行为基线测量")
    A("")
    A(f"域外 {len(ood)} 条（真实公开语料 "
      f"{sum(1 for r in ood if r.get('origin') == 'real')} 条 + 构造 "
      f"{sum(1 for r in ood if r.get('origin') == 'synthetic')} 条）"
      f"· 域内对照 {len(ind)} 条")
    A("")
    A("本轮只测现状，未实现任何拦截逻辑。填充率口径见 `scripts/probe_ood.py` 顶部注释，")
    A("域外与域内使用完全相同的口径。")
    A("")

    # ---- 表1 路由判定分布 ----
    A("## 表1 路由判定分布（域外 40 条被判成了什么）")
    A("")
    A("| 被判为 | 条数 | 占比 |")
    A("|---|---|---|")
    for d in DOMAINS + [None]:
        n = sum(1 for r in ood if r["routed_to"] == d)
        label = d if d else "未识别(routed_to=null)"
        A(f"| {label} | {n} | {n/len(ood):.1%} |")
    A("")
    n_null = sum(1 for r in ood if r["routed_to"] is None)
    A(f"**Q1 前置观察**：域外 {len(ood)} 条里 `routed_to=null` 共 **{n_null}** 条"
      f"（{n_null/len(ood):.1%}）—— 当前路由没有「以上都不是」选项，"
      f"{'与预期一致，域外文档全部被强行分入已注册域' if n_null == 0 else '出现了未识别，与预期不符，需复核'}。")
    A("")
    A("按来源细分：")
    A("")
    A("| 来源 | 语言 | 真实/构造 | n | 判为cord | 判为duee_fin | 判为ccks_fraud |")
    A("|---|---|---|---|---|---|---|")
    for src in sorted(set(r["source"] for r in ood)):
        sub = [r for r in ood if r["source"] == src]
        lang = "/".join(sorted(set(r["lang"] for r in sub)))
        org = "/".join(sorted(set(r["origin"] for r in sub)))
        cnt = {d: sum(1 for r in sub if r["routed_to"] == d) for d in DOMAINS}
        A(f"| {src} | {lang} | {org} | {len(sub)} | " +
          " | ".join(str(cnt[d]) for d in DOMAINS) + " |")
    A("")

    # ---- 表2 schema 合法率 ----
    A("## 表2 schema 合法率（Q1）")
    A("")
    A("| 组 | n | 可解析率 | schema 合法率 |")
    A("|---|---|---|---|")
    for name, grp in [("域外", ood), ("域内", ind)]:
        if not grp:
            continue
        A(f"| {name} | {len(grp)} | {sum(r['parse_ok'] for r in grp)/len(grp):.1%} "
          f"| {sum(r['schema_valid'] for r in grp)/len(grp):.1%} |")
    A("")
    ood_valid = sum(r["schema_valid"] for r in ood) / len(ood) if ood else 0
    verdict = ("≥80%，**Q1 成立：失败是静默的**——域外输入产出的是格式完全合法的 JSON，"
               "下游程序无法从形状上分辨对错。"
               if ood_valid >= 0.8 else
               "<80%，**Q1 不成立**：域外失败有相当比例是显性的（schema 校验就能挡下），"
               "第 7.1 节关于「静默失败」的论述需要按实测收敛。")
    A(f"**Q1 判定**：域外 schema 合法率 = **{ood_valid:.1%}**，{verdict}")
    A("")

    # ---- 表3 填充率分布 ----
    A("## 表3 字段填充率分布（Q2）")
    A("")
    A("### 3a 整体（域外 vs 域内）")
    A("")
    A("| 组 | n | 五数概括 |")
    A("|---|---|---|")
    A(f"| 域外 | {len(ood)} | {five_num([r['fill_rate'] for r in ood])} |")
    A(f"| 域内 | {len(ind)} | {five_num([r['fill_rate'] for r in ind])} |")
    A("")
    A("### 3b 按域拆开（这才是可用的口径）")
    A("")
    A("三个域的域内填充率量纲差异极大，混在一起看会得出错误结论。")
    A("")
    A("| 域 | 域内 n | 域内五数概括 | 域外(被判到该域) n | 域外五数概括 | 是否分得开 |")
    A("|---|---|---|---|---|---|")
    for d in DOMAINS:
        di = [r["fill_rate"] for r in ind if r["routed_to"] == d]
        do = [r["fill_rate"] for r in ood if r["routed_to"] == d]
        sep = "—"
        if len(di) >= 4 and len(do) >= 4:
            qi = statistics.quantiles(di, n=4)
            qo = statistics.quantiles(do, n=4)
            sep = "✅ 分得开" if qo[2] < qi[0] else "❌ 重叠"
        elif do and di:
            sep = "样本太少，看表4"
        A(f"| {d} | {len(di)} | {five_num(di)} | {len(do)} | {five_num(do)} | {sep} |")
    A("")
    A("判定规则：域外 p75 < 域内 p25 记为「分得开」。")
    A("")

    # ---- 表4 阈值扫描 ----
    A("## 表4 阈值扫描（Q3）")
    A("")
    A("规则：`fill_rate < 阈值` 判为域外并拒绝。")
    A("域外拦截率 = 被正确挡下的域外文档占比；域内误拒率 = 被错误挡下的域内文档占比。")
    A("")
    A("### 4a 全局单一阈值（用于说明它为什么不可行）")
    A("")
    A("| 阈值 | 域外拦截率 | 域内误拒率 |")
    A("|---|---|---|")
    ths = [round(0.10 + 0.05 * i, 2) for i in range(11)]
    for th in ths:
        oi = sum(1 for r in ood if r["fill_rate"] < th) / len(ood) if ood else 0
        ii = sum(1 for r in ind if r["fill_rate"] < th) / len(ind) if ind else 0
        A(f"| {th:.2f} | {oi:.1%} | {ii:.1%} |")
    A("")

    A("### 4b 按域阈值（实际可用的那个）")
    A("")
    for d in DOMAINS:
        di = [r["fill_rate"] for r in ind if r["routed_to"] == d]
        do = [r["fill_rate"] for r in ood if r["routed_to"] == d]
        A(f"**{d}**（域内 n={len(di)}，域外被判到此域 n={len(do)}）")
        A("")
        if not di:
            A("域内样本为 0，跳过。")
            A("")
            continue
        A("| 阈值 | 域外拦截率 | 域内误拒率 |")
        A("|---|---|---|")
        for th in ths:
            oi = (sum(1 for v in do if v < th) / len(do)) if do else float("nan")
            ii = sum(1 for v in di if v < th) / len(di)
            oi_s = "—" if not do else f"{oi:.1%}"
            A(f"| {th:.2f} | {oi_s} | {ii:.1%} |")
        A("")
        # 推荐阈值：域内误拒率 ≤2% 前提下最大化域外拦截率
        best = None
        for th in ths:
            ii = sum(1 for v in di if v < th) / len(di)
            if ii <= 0.02:
                oi = (sum(1 for v in do if v < th) / len(do)) if do else 0.0
                if best is None or oi > best[1]:
                    best = (th, oi, ii)
        if best:
            A(f"→ 「域内误拒率 ≤2%」约束下的最优阈值 **{best[0]:.2f}**："
              f"域外拦截率 **{best[1]:.1%}**，域内误拒率 {best[2]:.1%}")
        else:
            A("→ 在任何扫描阈值下域内误拒率都 >2%，该域无法用单一填充率阈值兜底。")
        A("")

    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", default="data/ood_probe/probe.jsonl")
    ap.add_argument("--indomain-n", type=int, default=30, help="每域取多少条域内对照")
    ap.add_argument("--out", default="runs/ood_probe.jsonl")
    ap.add_argument("--summary", default="runs/ood_probe_summary.md")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--limit", type=int, default=None, help="冒烟测试用：每组只跑前N条")
    args = ap.parse_args()

    ood_rows = [json.loads(l) for l in open(args.ood) if l.strip()]
    if args.limit:
        ood_rows = ood_rows[: args.limit]

    tasks: list[dict] = []
    for r in ood_rows:
        tasks.append({"id": r["id"], "group": "ood", "source": r["source"],
                      "lang": r["lang"], "origin": r["origin"],
                      "true_domain": None, "text": r["text"]})
    for d, path in DOMAIN_FILES.items():
        n = args.limit or args.indomain_n
        rows = [json.loads(l) for l in open(path) if l.strip()][:n]
        for i, r in enumerate(rows):
            tasks.append({"id": f"in_{d}_{i:03d}", "group": "indomain", "source": d,
                          "lang": "zh" if d != "cord" else "en", "origin": "real",
                          "true_domain": d, "text": r["user"]})

    print(f"[probe] 域外 {sum(1 for t in tasks if t['group']=='ood')} 条 + "
          f"域内 {sum(1 for t in tasks if t['group']=='indomain')} 条 = {len(tasks)} 条")
    orch = Orchestrator(args.base)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results: list[dict] = []
    t_start = time.time()
    with open(args.out, "w") as fout:
        for i, t in enumerate(tasks):
            rec = {k: t[k] for k in ("id", "group", "source", "lang", "origin", "true_domain")}
            rec.update(probe_one(orch, t["text"]))
            results.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t_start
                print(f"  {i+1}/{len(tasks)}  {el/(i+1):.1f}s/条  "
                      f"预计剩余 {(len(tasks)-i-1)*el/(i+1)/60:.1f} 分钟")

    print(f"\n[done] {len(results)} 条 -> {args.out}  用时 {(time.time()-t_start)/60:.1f} 分钟")
    summary = build_summary(results)
    with open(args.summary, "w") as f:
        f.write(summary + "\n")
    print(f"[done] 汇总 -> {args.summary}\n")
    print(summary)


if __name__ == "__main__":
    main()
