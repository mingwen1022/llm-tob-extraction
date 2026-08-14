"""生成评测明细站点：一个 index + 每轮一个明细页，自包含单文件 HTML。

与 doc/site/ 完全解耦——不共用 nav、不共用 style.css，可以整个目录单独拷走。

  doc/eval/index.html              入口：三轮对比 + 跳转
  doc/eval/round1_api_zeroshot.html  第一轮：6 模型裸跑
  doc/eval/round2_api_fewshot.html   第二轮：few-shot 梯度
  doc/eval/round3_cord.html          第三轮：CORD 基座 vs LoRA
  doc/eval/round3_duee.html          第三轮：DuEE-fin
  doc/eval/round3_ccks.html          第三轮：CCKS-fraud

每页顶部是该轮聚合指标，下面是逐条可展开卡片（含 TP/FP/FN 逐项拆解）。

Run:
  uv run python scripts/build_eval_site.py
"""
from __future__ import annotations

import html
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.eval import flatten, validity
from shared.schema import get_model

OUT = "doc/eval"

DOMAIN_GOLD = {
    "cord":       ("data/cord/test.eval.jsonl", "data/cord/train.eval.jsonl"),
    "duee_fin":   ("data/duee_fin_cn/test.eval.jsonl", "data/duee_fin_cn/train.eval.jsonl"),
    "ccks_fraud": ("data/ccks_fraud_cn/test.eval.jsonl", "data/ccks_fraud_cn/train.eval.jsonl"),
}

# 每轮：标题、说明、该轮包含哪些配置（展示名 -> 预测文件）
ROUNDS = [
    dict(slug="round1_api_zeroshot", domain="cord",
         title="第一轮 · API 裸跑（零示例）",
         desc="六个前沿/国产旗舰，完整 prompt，不给任何示例。同一批 CORD 干净 92 条。",
         configs=[
             ("Qwen3.7-Max",      "runs/fewshot/Qwen_Qwen3_7-Max_s0.jsonl"),
             ("Gemini-3.5-Flash", "runs/fewshot/google_gemini-3_5-flash_s0.jsonl"),
             ("MiniMax-M3",       "runs/fewshot/MiniMaxAI_MiniMax-M3_s0.jsonl"),
             ("GLM-5.2",          "runs/fewshot/zai-org_GLM-5_2-FP8_s0.jsonl"),
             ("Kimi-K3",          "runs/fewshot/moonshotai_kimi-k3_s0.jsonl"),
             ("DeepSeek-V4-Pro",  "runs/fewshot/deepseek-ai_DeepSeek-V4-Pro_s0.jsonl"),
             ("本地基座 4B",       "runs/e0_full.jsonl"),
         ]),
    dict(slug="round2_api_fewshot", domain="cord",
         title="第二轮 · API 补足示例（few-shot）",
         desc="给 API 补上 in-context 示例后重测。Gemini 与 MiniMax 跑满 0/4/8/16/32/64 六档，"
              "其余四家跑 0 与 16 两档。",
         # 用「模型 × shots」表达，页面会自动生成梯度矩阵
         matrix=dict(
             models=["Gemini-3.5-Flash", "MiniMax-M3", "Kimi-K3", "GLM-5.2",
                     "Qwen3.7-Max", "DeepSeek-V4-Pro"],
             shots=[0, 4, 8, 16, 32, 64],
             slug={"Gemini-3.5-Flash": "google_gemini-3_5-flash",
                   "MiniMax-M3": "MiniMaxAI_MiniMax-M3",
                   "Kimi-K3": "moonshotai_kimi-k3",
                   "GLM-5.2": "zai-org_GLM-5_2-FP8",
                   "Qwen3.7-Max": "Qwen_Qwen3_7-Max",
                   "DeepSeek-V4-Pro": "deepseek-ai_DeepSeek-V4-Pro"}),
         configs=[
             ("Gemini 0",  "runs/fewshot/google_gemini-3_5-flash_s0.jsonl"),
             ("Gemini 4",  "runs/fewshot/google_gemini-3_5-flash_s4.jsonl"),
             ("Gemini 8",  "runs/fewshot/google_gemini-3_5-flash_s8.jsonl"),
             ("Gemini 16", "runs/fewshot/google_gemini-3_5-flash_s16.jsonl"),
             ("Gemini 32", "runs/fewshot/google_gemini-3_5-flash_s32.jsonl"),
             ("Gemini 64", "runs/fewshot/google_gemini-3_5-flash_s64.jsonl"),
             ("MiniMax 0",  "runs/fewshot/MiniMaxAI_MiniMax-M3_s0.jsonl"),
             ("MiniMax 4",  "runs/fewshot/MiniMaxAI_MiniMax-M3_s4.jsonl"),
             ("MiniMax 8",  "runs/fewshot/MiniMaxAI_MiniMax-M3_s8.jsonl"),
             ("MiniMax 16", "runs/fewshot/MiniMaxAI_MiniMax-M3_s16.jsonl"),
             ("MiniMax 32", "runs/fewshot/MiniMaxAI_MiniMax-M3_s32.jsonl"),
             ("MiniMax 64", "runs/fewshot/MiniMaxAI_MiniMax-M3_s64.jsonl"),
             ("Kimi-K3 0",   "runs/fewshot/moonshotai_kimi-k3_s0.jsonl"),
             ("Kimi-K3 16",  "runs/fewshot/moonshotai_kimi-k3_s16.jsonl"),
             ("GLM-5.2 0",   "runs/fewshot/zai-org_GLM-5_2-FP8_s0.jsonl"),
             ("GLM-5.2 16",  "runs/fewshot/zai-org_GLM-5_2-FP8_s16.jsonl"),
             ("Qwen-Max 0",  "runs/fewshot/Qwen_Qwen3_7-Max_s0.jsonl"),
             ("Qwen-Max 16", "runs/fewshot/Qwen_Qwen3_7-Max_s16.jsonl"),
             ("DeepSeek 0",  "runs/fewshot/deepseek-ai_DeepSeek-V4-Pro_s0.jsonl"),
             ("DeepSeek 16", "runs/fewshot/deepseek-ai_DeepSeek-V4-Pro_s16.jsonl"),
         ]),
    dict(slug="round3_cord", domain="cord",
         title="第三轮 · 本地微调：CORD（英文收据）",
         desc="同一基座 Qwen3.5-4B，微调前后对比。基座用完整 prompt（含类型要求），口径一致。",
         configs=[("基座（完整prompt）", "runs/e0_full.jsonl"), ("LoRA 微调", "runs/e2.jsonl")]),
    dict(slug="round3_duee", domain="duee_fin",
         title="第三轮 · 本地微调：DuEE-fin（中文金融公告）",
         desc="事件抽取，schema 是 22 字段的并集，单个事件只填其中一类。",
         configs=[("基座", "runs/duee_e0.jsonl"), ("LoRA 微调", "runs/duee_e2.jsonl")]),
    dict(slug="round3_ccks", domain="ccks_fraud",
         title="第三轮 · 本地微调：CCKS-fraud（中文反欺诈）",
         desc="社交媒体吐槽体文本，噪声远高于规范文档。",
         configs=[("基座", "runs/ccks_e0.jsonl"), ("LoRA 微调", "runs/ccks_e2.jsonl")]),
]

CSS = """*{box-sizing:border-box}
body{font:14px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB",sans-serif;
 margin:0;padding:22px;max-width:1500px;margin-inline:auto;color:#1e1e1e;background:#fff}
a{color:#1c7ed6}
h1{font-size:22px;margin:0 0 3px} h2{font-size:16px;margin:26px 0 9px}
.sub{color:#868e96;font-size:13px;margin-bottom:16px}
.back{font-size:12.5px;color:#868e96;text-decoration:none;display:inline-block;margin-bottom:10px}
.back:hover{color:#1c7ed6}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 16px}
th,td{border:1px solid #e9ecef;padding:6px 9px;text-align:left}
th{background:#f8f9fa;font-weight:600}
td.n{font-family:ui-monospace,monospace}
tr.hl td{background:#ebfbee}
.c{border:1px solid #e9ecef;border-radius:9px;margin-bottom:9px;overflow:hidden}
.ch{padding:9px 12px;cursor:pointer;display:grid;gap:6px;align-items:center}
.ch:hover{background:#f8f9fa}
.q{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
/* 列头：与卡片标题同一套 grid，列宽由 --cols 决定 */
.chead{position:sticky;top:46px;z-index:8;background:#fff;display:grid;gap:6px;
 padding:7px 12px;margin:0 0 7px;border:1px solid #e9ecef;border-radius:9px;
 font-size:11px;color:#868e96;letter-spacing:.03em;align-items:end}
.chead .cn{font-weight:600;color:#495057;line-height:1.25;word-break:break-word}
.chead .unit{font-family:ui-monospace,monospace;font-size:10px;color:#adb5bd}
.f1{font-family:ui-monospace,monospace;font-size:12px;text-align:center;
 padding:2px 0;border-radius:5px;background:#f1f3f5;color:#495057}
.f1.ok{background:#d3f9d8;color:#2b8a3e} .f1.bad{background:#ffe3e3;color:#c92a2a}
.f1.mid{background:#fff3bf;color:#a06e00}
.p{font-size:11px;padding:2px 7px;border-radius:99px;background:#f1f3f5;color:#495057;
 font-family:ui-monospace,monospace;white-space:nowrap}
.p.bad{background:#ffe3e3;color:#c92a2a} .p.warn{background:#fff3bf;color:#a06e00}
.p.ok{background:#d3f9d8;color:#2b8a3e} .p.id{background:#e7f5ff;color:#1971c2}
.b{display:none;padding:0 12px 12px;border-top:1px solid #f1f3f5}
.c.open .b{display:block}
.k{font-size:11px;color:#adb5bd;text-transform:uppercase;letter-spacing:.05em;margin-top:10px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,monospace;
 background:#f8f9fa;padding:8px 10px;border-radius:6px;margin:3px 0 0;max-height:320px;overflow:auto}
.bar{position:sticky;top:0;background:#fff;padding:9px 0;border-bottom:1px solid #e9ecef;
 margin-bottom:12px;display:flex;gap:6px;flex-wrap:wrap;z-index:9;align-items:center}
button{font:13px inherit;padding:5px 10px;border:1px solid #ced4da;background:#fff;
 border-radius:6px;cursor:pointer}
button.on{background:#1e1e1e;color:#fff;border-color:#1e1e1e}
button .n{opacity:.55;margin-left:5px;font-size:11px}
input[type=search]{font:13px inherit;padding:5px 10px;border:1px solid #ced4da;border-radius:6px;
 min-width:200px;background:#fff;color:inherit}
.hit{color:#868e96;font-size:12px;margin:0 0 10px}
.sep{width:1px;height:20px;background:#dee2e6;margin:0 3px}
.cfg{display:grid;grid-template-columns:130px 1fr;gap:8px;align-items:start;margin-top:7px}
.cfgname{font-size:12px;font-weight:600;padding-top:7px}
.diff{display:flex;flex-wrap:wrap;gap:5px;margin-top:5px}
.d{font-size:11px;padding:2px 7px;border-radius:5px;font-family:ui-monospace,monospace}
.d.tp{background:#d3f9d8;color:#2b8a3e} .d.fp{background:#ffe3e3;color:#c92a2a}
.d.fn{background:#fff3bf;color:#a06e00}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:13px;margin:14px 0}
.card{display:block;border:1px solid #e9ecef;border-radius:10px;padding:15px 16px;
 text-decoration:none;color:inherit;transition:.15s}
.card:hover{border-color:#1c7ed6;transform:translateY(-2px)}
.card .t{font-weight:600;margin-bottom:3px} .card .d2{font-size:12.5px;color:#868e96;margin-bottom:9px}
.card .m{font-size:12px;font-family:ui-monospace,monospace;color:#495057}
.note{border-left:3px solid #ced4da;background:#f8f9fa;padding:10px 13px;border-radius:0 7px 7px 0;
 margin:14px 0;font-size:13px}
@media (prefers-color-scheme:dark){
 body{background:#141414;color:#e9ecef} a{color:#4dabf7}
 th{background:#1e1e1e} th,td{border-color:#343a40} tr.hl td{background:#1a2e1f}
 .c{border-color:#343a40} .ch:hover{background:#1e1e1e} .b{border-color:#2b2b2b}
 .bar{background:#141414;border-color:#343a40} pre{background:#1e1e1e}
 .p{background:#2b2b2b;color:#ced4da} .p.id{background:#1b3a52;color:#74c0fc}
 button{background:#1e1e1e;color:#e9ecef;border-color:#495057}
 button.on{background:#e9ecef;color:#141414}
 input[type=search]{background:#1e1e1e;border-color:#495057}
 .card{border-color:#343a40} .note{background:#1e1e1e;border-color:#495057}
 .d.tp{background:#193d24;color:#8ce99a} .d.fp{background:#3d1a1a;color:#ffa8a8}
 .d.fn{background:#3d3312;color:#ffd43b} .sep{background:#343a40}}
"""

JS = """
document.querySelectorAll('.ch').forEach(h=>h.onclick=e=>{
  if(e.target.tagName==='A') return;
  h.parentElement.classList.toggle('open');});
const cards=[...document.querySelectorAll('.c')];
const btns=[...document.querySelectorAll('.bar button')];
const box=document.querySelector('#q');
let filter='all';
function apply(){
  const kw=(box?box.value:'').trim().toLowerCase();
  let n=0;
  cards.forEach(c=>{
    const okTag = filter==='all' || c.dataset.tags.split(',').includes(filter);
    const okKw  = !kw || c.dataset.text.toLowerCase().includes(kw);
    const show  = okTag && okKw;
    c.style.display = show?'':'none';
    if(show) n++;
  });
  const h=document.querySelector('#hit');
  if(h) h.textContent = `显示 ${n} / ${cards.length} 条`;
}
btns.forEach(b=>b.onclick=()=>{
  filter=b.dataset.f;
  btns.forEach(x=>x.classList.toggle('on', x===b));
  apply();
});
if(box) box.oninput=apply;
apply();
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def clean_index(domain):
    gp, tp_ = DOMAIN_GOLD[domain]
    gold = load(gp)
    train = {r["user"] for r in load(tp_)}
    keep = [i for i, r in enumerate(gold) if r["user"] not in train]
    return gold, keep


def per_doc(pred_row, gold_row, model):
    """返回该条的 obj/状态/TP-FP-FN/F1 与差异明细。"""
    obj, status = validity(pred_row["output"], model)
    d = obj.model_dump(exclude_none=True) if hasattr(obj, "model_dump") else obj
    G = Counter(flatten(gold_row["gt"]))
    P = Counter(flatten(d)) if d else Counter()
    tp_items = sorted((G & P).elements())
    fp_items = sorted((P - G).elements())
    fn_items = sorted((G - P).elements())
    tp, fp, fn = len(tp_items), len(fp_items), len(fn_items)
    den = 2 * tp + fp + fn
    return dict(status=status, tp=tp, fp=fp, fn=fn,
                f1=(2 * tp / den if den else 1.0),
                tp_items=tp_items, fp_items=fp_items, fn_items=fn_items,
                raw=pred_row["output"],
                tok_in=pred_row.get("tok_in"), tok_out=pred_row.get("tok_out"))


def agg(stats):
    """把逐条结果汇总成该配置的指标。"""
    TP = sum(s["tp"] for s in stats); FP = sum(s["fp"] for s in stats); FN = sum(s["fn"] for s in stats)
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    docf1 = [s["f1"] for s in stats]
    return dict(P=p, R=r, micro=f1,
                docmacro=statistics.mean(docf1),
                schema=sum(1 for s in stats if s["status"] == "valid") / len(stats),
                perfect=sum(1 for s in docf1 if s >= 0.999) / len(docf1),
                zero=sum(1 for s in docf1 if s <= 0.001) / len(docf1),
                n=len(stats))


def diff_html(items, cls):
    if not items:
        return ""
    return "".join(f'<span class="d {cls}">{esc(k)}={esc(v)}</span>' for k, v in items[:14]) + \
           (f'<span class="d {cls}">…另 {len(items)-14} 项</span>' if len(items) > 14 else "")


def build_round(rd) -> dict:
    domain = rd["domain"]
    gold, keep = clean_index(domain)
    model = get_model(domain)
    names = [n for n, _ in rd["configs"]]

    # 逐配置跑一遍
    results = {}
    for name, path in rd["configs"]:
        rows = load(path)
        results[name] = [per_doc(rows[i], gold[i], model) for i in keep]
    aggs = {n: agg(results[n]) for n in names}

    # ---- 顶部聚合 ----
    best = max(aggs.values(), key=lambda a: a["micro"])["micro"]
    head = []

    # 有 matrix 配置的（第二轮）先出一张「模型 × shots」梯度矩阵
    if rd.get("matrix"):
        m = rd["matrix"]
        name_of = {}   # (model, shot) -> 配置展示名
        for n, path in rd["configs"]:
            for mo, sl in m["slug"].items():
                for s in m["shots"]:
                    if path.endswith(f"{sl}_s{s}.jsonl"):
                        name_of[(mo, s)] = n
        head.append("<h2>梯度矩阵 · micro-F1（行=模型，列=示例条数）</h2>")
        head.append("<table><tr><th>模型</th>" +
                    "".join(f"<th>{s}-shot</th>" for s in m["shots"]) +
                    "<th>0→最佳</th></tr>")
        for mo in m["models"]:
            cells, vals = [], []
            for s in m["shots"]:
                n = name_of.get((mo, s))
                if n is None:
                    cells.append('<td class="n" style="opacity:.35">—</td>')
                    continue
                v = aggs[n]["micro"]
                vals.append((s, v))
                top = ' style="background:#ebfbee;font-weight:600"' if v == max(
                    aggs[name_of[(mo, x)]]["micro"] for x in m["shots"] if (mo, x) in name_of) else ""
                cells.append(f'<td class="n"{top}>{v:.3f}</td>')
            delta = ""
            if len(vals) >= 2:
                z = dict(vals).get(0)
                if z is not None:
                    delta = f'+{max(v for _, v in vals) - z:.3f}'
            head.append(f'<tr><td>{esc(mo)}</td>{"".join(cells)}<td class="n">{delta}</td></tr>')
        head.append("</table>")
        head.append('<div class="note">绿底=该模型自己的最佳档。'
                    'Gemini 32→64 完全持平（0.927→0.927）确认<b>饱和</b>；'
                    'MiniMax 16 档后进入 0.907~0.922 的噪声带。'
                    '同样 16 个示例各家收益从 +0.036 到 +0.078 差一倍多，'
                    '<b>「给 API 加示例」的收益并不可预期</b>。</div>')
        head.append("<h2>各配置完整指标</h2>")

    head.append("<table><tr><th>配置</th><th>Precision</th><th>Recall</th><th>micro-F1</th>"
                "<th>macro·按文档</th><th>schema 合法率</th><th>完美率</th><th>全错率</th></tr>")
    for n in names:
        a = aggs[n]
        hl = ' class="hl"' if a["micro"] == best else ""
        head.append(
            f'<tr{hl}><td>{esc(n)}</td><td class="n">{a["P"]:.3f}</td><td class="n">{a["R"]:.3f}</td>'
            f'<td class="n"><b>{a["micro"]:.3f}</b></td><td class="n">{a["docmacro"]:.3f}</td>'
            f'<td class="n">{a["schema"]:.0%}</td><td class="n">{a["perfect"]:.0%}</td>'
            f'<td class="n">{a["zero"]:.0%}</td></tr>')
    head.append("</table>")

    # ---- 筛选按钮统计（以最后一个配置为主视角，通常是最好的那个）----
    main = names[-1]
    n_bad     = sum(1 for s in results[main] if s["f1"] < 0.999)
    n_zero    = sum(1 for s in results[main] if s["f1"] <= 0.001)
    n_schema  = sum(1 for s in results[main] if s["status"] != "valid")
    n_perfect = sum(1 for s in results[main] if s["f1"] >= 0.999)
    n_improve = 0
    if len(names) >= 2:
        n_improve = sum(1 for a, b in zip(results[names[0]], results[main]) if b["f1"] - a["f1"] > 0.01)
    n_regress = 0
    if len(names) >= 2:
        n_regress = sum(1 for a, b in zip(results[names[0]], results[main]) if a["f1"] - b["f1"] > 0.01)

    bar = [f'<div class="bar"><button class="on" data-f="all">全部<span class="n">{len(keep)}</span></button>',
           f'<button data-f="bad">「{esc(main)}」未满分<span class="n">{n_bad}</span></button>',
           f'<button data-f="zero">「{esc(main)}」F1=0<span class="n">{n_zero}</span></button>',
           f'<button data-f="schemafail">schema 不合法<span class="n">{n_schema}</span></button>',
           f'<button data-f="perfect">完美 (F1=1)<span class="n">{n_perfect}</span></button>']
    if len(names) >= 2:
        bar.append('<span class="sep"></span>')
        bar.append(f'<button data-f="improve">末配置优于首配置<span class="n">{n_improve}</span></button>')
        bar.append(f'<button data-f="regress">末配置劣于首配置<span class="n">{n_regress}</span></button>')
    bar.append('<input type="search" id="q" placeholder="搜原文 / 输出 / 字段…">')
    bar.append('</div><div class="hit" id="hit"></div>')

    # ---- 逐条卡片 ----
    cards = []
    for pos, i in enumerate(keep):
        g = gold[i]
        s_main = results[main][pos]
        tags = []
        if s_main["f1"] < 0.999: tags.append("bad")
        if s_main["f1"] <= 0.001: tags.append("zero")
        if s_main["status"] != "valid": tags.append("schemafail")
        if s_main["f1"] >= 0.999: tags.append("perfect")
        if len(names) >= 2:
            d = s_main["f1"] - results[names[0]][pos]["f1"]
            if d > 0.01: tags.append("improve")
            if d < -0.01: tags.append("regress")

        searchable = g["user"][:400] + " " + json.dumps(g["gt"], ensure_ascii=False)[:400] + \
                     " " + str(s_main["raw"])[:400]

        badges = [f'<span class="q">{esc(g["user"][:110])}</span>',
                  f'<span class="p id">#{i}</span>']
        for n in names:
            s = results[n][pos]
            cls = "ok" if s["f1"] >= 0.999 else ("bad" if s["f1"] <= 0.001 else
                                                 ("mid" if s["f1"] < 0.7 else ""))
            badges.append(f'<span class="f1 {cls}">{s["f1"]:.2f}</span>')

        body = [f'<div class="k">输入原文</div><pre>{esc(g["user"][:1500])}</pre>',
                f'<div class="k">GOLD</div><pre>{esc(json.dumps(g["gt"], ensure_ascii=False, indent=1))}</pre>',
                '<div class="k">各配置输出与差异</div>']
        for n in names:
            s = results[n][pos]
            tok = ""
            if s["tok_in"]:
                tok = f' · in {s["tok_in"]} / out {s["tok_out"]} tok'
            body.append(
                f'<div class="cfg"><div class="cfgname">{esc(n)}<br>'
                f'<span class="p {"ok" if s["f1"]>=0.999 else ("bad" if s["f1"]<=0.001 else "")}">'
                f'F1 {s["f1"]:.3f}</span></div><div>'
                f'<pre>{esc(str(s["raw"])[:1200])}</pre>'
                f'<div class="diff">'
                f'<span class="d tp">TP {s["tp"]}</span>'
                f'<span class="d fp">FP {s["fp"]}</span>'
                f'<span class="d fn">FN {s["fn"]}</span>'
                f'<span class="d">schema {"✓" if s["status"]=="valid" else "✗ "+s["status"]}</span>'
                f'<span class="d">{tok}</span></div>'
                f'<div class="diff">{diff_html(s["fp_items"],"fp")}{diff_html(s["fn_items"],"fn")}</div>'
                f'</div></div>')

        cards.append(
            f'<div class="c" data-tags="{",".join(tags)}" data-text="{esc(searchable)}">'
            f'<div class="ch">{"".join(badges)}</div>'
            f'<div class="b">{"".join(body)}</div></div>')

    # ---- 列头（与卡片标题共用同一套 grid）----
    colw = 62 if len(names) > 8 else 76
    grid = f"grid-template-columns:minmax(110px,1fr) 40px repeat({len(names)},{colw}px)"
    chead = (f'<div class="chead" style="{grid}">'
             f'<div>文档原文（点击展开）</div><div class="unit">#</div>'
             + "".join(f'<div class="cn">{esc(n)}</div>' for n in names)
             + '</div>')
    # 让卡片标题用同一套 grid
    cards = [c.replace('<div class="ch">', f'<div class="ch" style="{grid}">') for c in cards]

    page = (f'<!doctype html><meta charset=utf-8><title>{esc(rd["title"])}</title>'
            f'<style>{CSS}</style>'
            f'<a class="back" href="index.html">← 返回评测目录</a>'
            f'<h1>{esc(rd["title"])}</h1>'
            f'<div class="sub">{esc(rd["desc"])}　·　{len(keep)} 条（已去泄漏）</div>'
            f'{"".join(head)}'
            f'<h2>逐条明细</h2>'
            f'<div class="note"><b>怎么读</b>：下表每行是一份文档，'
            f'<b>各列数字 = 该配置在这一条上的 micro-F1</b>'
            f'（<span class="f1 ok">1.00</span> 全对 · '
            f'<span class="f1">0.7~1</span> 部分对 · '
            f'<span class="f1 mid">&lt;0.7</span> 差 · '
            f'<span class="f1 bad">0.00</span> 全错）。'
            f'点行展开看输入原文、GOLD、各配置输出，以及差异逐项拆解——'
            f'<span class="d fp">红=多抽/抽错(FP)</span> '
            f'<span class="d fn">黄=漏抽(FN)</span>，格式 <code>字段=归一化值</code>。</div>'
            f'{"".join(bar)}{chead}{"".join(cards)}'
            f'<script>{JS}</script>')
    return dict(html=page, aggs=aggs, names=names, n=len(keep))


def build_index(built):
    rows = []
    for rd in ROUNDS:
        b = built[rd["slug"]]
        best_name = max(b["names"], key=lambda n: b["aggs"][n]["micro"])
        a = b["aggs"][best_name]
        rows.append((rd, b, best_name, a))

    cards = "".join(
        f'<a class="card" href="{rd["slug"]}.html"><div class="t">{esc(rd["title"])}</div>'
        f'<div class="d2">{esc(rd["desc"][:60])}…</div>'
        f'<div class="m">{b["n"]} 条 · {len(b["names"])} 个配置<br>'
        f'最佳 {esc(best_name)} · micro-F1 <b>{a["micro"]:.3f}</b></div></a>'
        for rd, b, best_name, a in rows)

    # 跨轮对比表：每轮取其最佳配置
    cmp_rows = "".join(
        f'<tr><td><a href="{rd["slug"]}.html">{esc(rd["title"])}</a></td>'
        f'<td>{esc(best_name)}</td><td class="n">{a["P"]:.3f}</td><td class="n">{a["R"]:.3f}</td>'
        f'<td class="n"><b>{a["micro"]:.3f}</b></td><td class="n">{a["docmacro"]:.3f}</td>'
        f'<td class="n">{a["schema"]:.0%}</td><td class="n">{a["perfect"]:.0%}</td>'
        f'<td class="n">{a["zero"]:.0%}</td></tr>'
        for rd, b, best_name, a in rows)

    # CORD 三轮纵向对比（同一批 92 条，可比）
    c1 = built["round1_api_zeroshot"]["aggs"]
    c2 = built["round2_api_fewshot"]["aggs"]
    c3 = built["round3_cord"]["aggs"]
    ladder = [("API 裸跑最佳（Qwen3.7-Max）", c1["Qwen3.7-Max"]),
              ("API few-shot 最佳（Gemini 32-shot）", c2["Gemini 32"]),
              ("本地基座（完整 prompt，未微调）", c1["本地基座 4B"]),
              ("本地 LoRA 微调", c3["LoRA 微调"])]
    ladder_rows = "".join(
        f'<tr{" class=hl" if "LoRA" in n else ""}><td>{esc(n)}</td>'
        f'<td class="n">{a["P"]:.3f}</td><td class="n">{a["R"]:.3f}</td>'
        f'<td class="n"><b>{a["micro"]:.3f}</b></td><td class="n">{a["docmacro"]:.3f}</td>'
        f'<td class="n">{a["schema"]:.0%}</td><td class="n">{a["perfect"]:.0%}</td></tr>'
        for n, a in ladder)

    return (f'<!doctype html><meta charset=utf-8><title>评测明细 · 通用大模型 API vs 本地微调</title>'
            f'<style>{CSS}</style>'
            f'<h1>评测明细</h1>'
            f'<div class="sub">通用大模型 API vs 本地微调小模型 · 逐条可查 · '
            f'指标口径与结论见 <code>runs/model_selection_report.md</code></div>'

            f'<h2>CORD 三轮纵向对比（同一批干净 92 条，可直接比）</h2>'
            f'<table><tr><th>方案</th><th>Precision</th><th>Recall</th><th>micro-F1</th>'
            f'<th>macro·按文档</th><th>schema</th><th>完美率</th></tr>{ladder_rows}</table>'
            f'<div class="note">零示例时本地微调领先约 <b>8.5 个点</b>；给 API 补足示例后差距收敛到 '
            f'<b>1.8 个点</b>，而 API 自身重跑波动就有 <b>±1.4 个点</b>——'
            f'精度已不再是选型理由，剩下的是合规、延迟、成本结构与输出确定性。</div>'

            f'<h2>各轮入口</h2>{cards}'

            f'<h2>各轮最佳配置一览</h2>'
            f'<table><tr><th>轮次</th><th>最佳配置</th><th>P</th><th>R</th><th>micro-F1</th>'
            f'<th>macro·按文档</th><th>schema</th><th>完美率</th><th>全错率</th></tr>{cmp_rows}</table>'
            f'<div class="note">⚠ 第一、二轮只在 <b>CORD</b> 上做，第三轮才是三个域。'
            f'跨域的数字不要混着读——三个域的 schema、条数、难度都不同，'
            f'详见报告 §3.3「为什么不给三域合并总分」。</div>')


def main():
    os.makedirs(OUT, exist_ok=True)
    built = {}
    for rd in ROUNDS:
        print(f"[build] {rd['slug']} ...", flush=True)
        b = build_round(rd)
        built[rd["slug"]] = b
        p = os.path.join(OUT, rd["slug"] + ".html")
        with open(p, "w") as f:
            f.write(b["html"])
        print(f"        -> {p}  ({os.path.getsize(p)/1024:.0f} KB, {b['n']} 条)")
    p = os.path.join(OUT, "index.html")
    with open(p, "w") as f:
        f.write(build_index(built))
    print(f"[build] -> {p}")


if __name__ == "__main__":
    main()
