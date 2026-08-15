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
from shared.router import build_router_prompt
from shared.schema import build_system_prompt, get_model

OUT = "doc/eval"

DOMAIN_GOLD = {
    "cord":       ("data/cord/test.eval.jsonl", "data/cord/train.eval.jsonl"),
    "duee_fin":   ("data/duee_fin_cn/test.eval.jsonl", "data/duee_fin_cn/train.eval.jsonl"),
    "ccks_fraud": ("data/ccks_fraud_cn/test.eval.jsonl", "data/ccks_fraud_cn/train.eval.jsonl"),
}

# 每轮：标题、说明、该轮包含哪些配置（展示名 -> 预测文件）
ROUNDS = [
    dict(slug="round1_api_zeroshot", domain="cord",
         title="第一轮 · API 裸跑（零示例）· CORD 英文收据",
         desc="六个前沿/国产旗舰，完整 prompt，不给任何示例。同一批 CORD 干净 92 条。",
         shots_note="不给示例——messages 只有 system + 待抽文档两条。",
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
         title="第二轮 · API 补足示例（few-shot）· CORD 英文收据",
         desc="给 API 补上 in-context 示例后重测。Gemini 与 MiniMax 跑满 0/4/8/16/32/64 六档，"
              "其余四家跑 0 与 16 两档。本地基座 4B 也跑了 0/16/32 三档——"
              "同一个 4B、同一批示例、同一个 prompt，与第三轮的 LoRA 只差「示例进上下文还是进权重」。",
         shots_note="system 之后插入 N 对 user/assistant 伪造对话：user=训练集收据原文，"
                    "assistant=该条 gold JSON（原样 json.dumps，未清洗），最后才是待抽文档。"
                    "示例取自 train.eval.jsonl 的「前 N 条干净样本」——先滤掉 8 条与测试集重复的，"
                    "再取前 N，确定可复现。",
         # 用「模型 × shots」表达，页面会自动生成梯度矩阵
         matrix=dict(
             models=["Gemini-3.5-Flash", "MiniMax-M3", "Kimi-K3", "GLM-5.2",
                     "Qwen3.7-Max", "DeepSeek-V4-Pro", "本地基座 Qwen3.5-4B"],
             shots=[0, 4, 8, 16, 32, 64],
             slug={"Gemini-3.5-Flash": "google_gemini-3_5-flash",
                   "MiniMax-M3": "MiniMaxAI_MiniMax-M3",
                   "Kimi-K3": "moonshotai_kimi-k3",
                   "GLM-5.2": "zai-org_GLM-5_2-FP8",
                   "Qwen3.7-Max": "Qwen_Qwen3_7-Max",
                   "DeepSeek-V4-Pro": "deepseek-ai_DeepSeek-V4-Pro",
                   "本地基座 Qwen3.5-4B": "e0_full"}),
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
             ("本地基座 0",  "runs/e0_full.jsonl"),
             ("本地基座 16", "runs/e0_full_s16.jsonl"),
             ("本地基座 32", "runs/e0_full_s32.jsonl"),
         ]),
    dict(slug="round3_cord", domain="cord",
         title="第三轮 · 本地微调（LoRA）· CORD 英文收据",
         desc="同一基座 Qwen3.5-4B，四种用法横向对比：不给示例 / 给 16 条 / 给 32 条 / 把示例训进权重。"
              "prompt 与示例来源完全一致，唯一变量是示例放在上下文里还是放在权重里。",
         shots_note="基座三档的示例注入方式与第二轮完全相同（同一套取法与去污染）；"
                    "LoRA 那一档 messages 只有 system + 待抽文档，示例已在权重里。",
         baseline="基座 0-shot",
         stored_prompt=["LoRA 微调"],
         configs=[("基座 0-shot", "runs/e0_full.jsonl"),
                  ("基座 +16示例", "runs/e0_full_s16.jsonl"),
                  ("基座 +32示例", "runs/e0_full_s32.jsonl"),
                  ("LoRA 微调", "runs/e2.jsonl")]),
    dict(slug="round3_duee", domain="duee_fin",
         title="第三轮 · 本地微调（LoRA）· DuEE-fin 中文金融公告",
         desc="事件抽取，schema 是 22 字段的并集，单个事件只填其中一类。",
         baseline="基座",
         shots_note="两档都不给示例——messages 只有 system + 待抽文档两条。",
         stored_prompt=["LoRA 微调"],
         configs=[("基座", "runs/duee_e0.jsonl"), ("LoRA 微调", "runs/duee_e2.jsonl")]),
    dict(slug="round3_ccks", domain="ccks_fraud",
         title="第三轮 · 本地微调（LoRA）· CCKS-fraud 中文反欺诈",
         desc="社交媒体吐槽体文本，噪声远高于规范文档。",
         baseline="基座",
         shots_note="两档都不给示例——messages 只有 system + 待抽文档两条。",
         stored_prompt=["LoRA 微调"],
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
.ch{padding:9px 12px;cursor:pointer;display:grid;gap:6px;align-items:center;
 grid-template-columns:var(--gcols)}
.ch:hover{background:#f8f9fa}
.q{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
/* 列头：与卡片标题同一套 grid，列宽由 --cols 决定 */
.chead{position:sticky;top:46px;z-index:8;background:#fff;display:grid;gap:6px;
 grid-template-columns:var(--gcols);
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
.arrow{color:#adb5bd;margin:0 7px}
.scroll{overflow-x:auto;max-width:100%}
.sp{border:1px solid #e9ecef;border-radius:9px;margin:0 0 14px;font-size:13px}
.sp summary{padding:9px 12px;cursor:pointer;color:#495057;font-weight:600}
.sp summary:hover{background:#f8f9fa}
.spbody{padding:0 12px 12px;color:#495057}
.sprow{margin:6px 0 0 2px}
.spn{display:inline-block;width:17px;height:17px;line-height:17px;text-align:center;
 border-radius:99px;background:#e7f5ff;color:#1971c2;font-size:10.5px;margin-right:7px}
.spnote{margin-top:10px;padding:8px 10px;background:#f8f9fa;border-radius:6px;font-size:12.5px}
.scroll table{min-width:max-content}
td.ok{background:#d3f9d8;color:#2b8a3e} td.mid{background:#fff3bf;color:#a06e00}
td.bad{background:#ffe3e3;color:#c92a2a} td.dim{color:#ced4da}
tr.ghost td{background:#fff5f5}
.tag{margin-left:6px;font-size:10px;padding:1px 5px;border-radius:99px;
 background:#ffe3e3;color:#c92a2a;vertical-align:middle}
.tag.warn{background:#fff3bf;color:#a06e00}
.sw{display:inline-block;padding:0 6px;border-radius:4px;background:#f1f3f5;font-size:11.5px}
.sw.ok{background:#d3f9d8;color:#2b8a3e} .sw.mid{background:#fff3bf;color:#a06e00}
.sw.bad{background:#ffe3e3;color:#c92a2a}
.sub2{display:block;font-size:10px;opacity:.75;margin-top:1px}
.rch{grid-template-columns:minmax(200px,1fr) 46px 190px 190px 90px}
.gain{margin-left:7px;padding:1px 6px;border-radius:99px;background:#d3f9d8;color:#2b8a3e;font-size:11.5px}
.bar2{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:0 0 10px;
 padding:8px 10px;background:#f8f9fa;border-radius:8px}
.bar2 .lbl{font-size:11.5px;color:#868e96;margin-right:2px}
.colbtn{font:12.5px inherit;padding:4px 9px;border:1px solid #ced4da;background:#fff;
 border-radius:6px;cursor:pointer}
.colbtn.on{background:#1c7ed6;color:#fff;border-color:#1c7ed6}
.colbtn .n{margin-left:5px;font-size:10.5px;opacity:.6}
.bar2 .sep{width:1px;height:16px;background:#dee2e6;margin:0 4px}
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
 .bar2{background:#1e1e1e} .colbtn{background:#1e1e1e;color:#e9ecef;border-color:#495057}
 .colbtn.on{background:#1c7ed6;color:#fff;border-color:#1c7ed6}
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

// ---- 列筛选：按下标同时隐藏「列头 + 所有行」的单元格，并重算 grid ----
const NCOL=+document.body.dataset.ncol, COLW=+document.body.dataset.colw;
const heads=[...document.querySelectorAll('.chead .cn')];
function showCols(keep){                       // keep: 下标 Set
  heads.forEach(h=>{h.style.display = keep.has(+h.dataset.ci)?'':'none';});
  document.querySelectorAll('.ch .f1').forEach(f=>{
    f.style.display = keep.has(+f.dataset.ci)?'':'none';});
  document.body.style.setProperty('--gcols',
    `minmax(110px,1fr) 40px repeat(${keep.size},${COLW}px)`);
}
const colbtns=[...document.querySelectorAll('.colbtn')];
colbtns.forEach(b=>b.onclick=()=>{
  colbtns.forEach(x=>x.classList.toggle('on', x===b));
  let keep;
  if(b.dataset.all)        keep=new Set(heads.map(h=>+h.dataset.ci));
  else if(b.dataset.shot)  keep=new Set(heads.filter(h=>h.dataset.shot===b.dataset.shot)
                                             .map(h=>+h.dataset.ci));
  else                     keep=new Set(heads.filter(h=>h.dataset.model===b.dataset.model)
                                             .map(h=>+h.dataset.ci));
  showCols(keep);
});
// 初始：有列筛选栏的页面默认只显示 16-shot，其余页面显示全部
const on0=document.querySelector('.colbtn.on');
if(on0) on0.click(); else showCols(new Set(heads.map(h=>+h.dataset.ci)));

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
    # 字段维度：按字段路径把全部文档的 TP/FP/FN 汇总，每字段先算 F1 再对字段数取平均。
    # 与文档维度是同一批 TP/FP/FN 换个分桶方式——每字段一票，不管它出现 10 次还是 230 次。
    byf = {}
    for st in stats:
        for key, items in (("tp", st["tp_items"]), ("fp", st["fp_items"]), ("fn", st["fn_items"])):
            for k, _ in items:
                byf.setdefault(k, {"tp": 0, "fp": 0, "fn": 0})[key] += 1
    fieldf1 = [2 * b["tp"] / (2 * b["tp"] + b["fp"] + b["fn"])
               for b in byf.values() if 2 * b["tp"] + b["fp"] + b["fn"]]
    return dict(P=p, R=r, micro=f1,
                fieldmacro=statistics.mean(fieldf1) if fieldf1 else 0.0,
                nfield=len(byf),
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

    # 配置名 -> (模型全名, shots)。列头受 62px 宽度限制只能用短名（"Gemini 16"），
    # 但聚合表是普通表格，用全名（"Gemini-3.5-Flash · 16-shot"）更好读。
    colmeta = {}
    if rd.get("matrix"):
        m = rd["matrix"]
        for n, path in rd["configs"]:
            for mo, sl in m["slug"].items():
                for sh in m["shots"]:
                    if (path.endswith(f"{sl}_s{sh}.jsonl")
                            or (sh == 0 and path.endswith(f"{sl}.jsonl"))):
                        colmeta[n] = (mo, sh)

    def full_name(n):
        if n not in colmeta:
            return n
        mo, sh = colmeta[n]
        return f"{mo} · {sh}-shot" if sh else f"{mo} · 零示例"

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
                    if (path.endswith(f"{sl}_s{s}.jsonl")
                            or (s == 0 and path.endswith(f"{sl}.jsonl"))):
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

    # ---- system prompt：注意 LoRA 与基座未必用同一个 ----
    # LoRA 必须用它训练时那版 prompt（train/inference 一致），而基座为了公平用完整版。
    # CORD 上这两者不同（218 vs 479 字符）；另外两域的训练 prompt 本身就是完整版，故相同。
    full_p = build_system_prompt(domain, rich=True, types=True)
    stored_p = load(DOMAIN_GOLD[domain][0])[0]["system"]      # 训练/eval 文件里存的那版
    stored_names = [n for n in rd.get("stored_prompt", []) if n in names]
    split = bool(stored_names) and stored_p != full_p

    parts = ["<b>任务说明</b>（这是什么抽取器）",
             "<b>字段说明</b>（每个字段是什么意思，人类可读）",
             "<b>类型要求</b>（值一律输出 JSON 字符串，金额数量也要加引号）",
             "<b>输出约定</b>（缺失填 null、列表无项填 []、只输出 JSON）",
             "<b>compact schema</b>（一行字段图，由 Pydantic 模型自动生成）"]
    if split:
        others = [n for n in names if n not in stored_names]
        summary = (f'system prompt 构成（<b>两版</b>：{esc("、".join(others))} 用 {len(full_p)} 字符完整版，'
                   f'{esc("、".join(stored_names))} 用 {len(stored_p)} 字符训练版）')
        body = (f'<div class="spnote"><b>⚠ 这一页的配置没有共用同一个 prompt，这是刻意的。</b><br>'
                f'{esc("、".join(stored_names))} 是用 {len(stored_p)} 字符那版<b>训练</b>的，'
                f'推理必须跟着用同一版，否则就是 train/inference 不一致；'
                f'而基座侧为了不让它输在"话没说全"上，统一给了 {len(full_p)} 字符的完整版'
                f'（多出字段说明与类型要求）。<br>'
                f'<b>也就是说微调侧拿到的提示信息更少，这个对比是往对它不利的方向做的。</b></div>'
                f'<div class="k">完整版（{len(full_p)} 字符）· 五段拼成，'
                f'由 <code>build_system_prompt(domain, rich=True, types=True)</code> 生成</div>'
                + "".join(f'<div class="sprow"><span class="spn">{i+1}</span>{t}</div>'
                          for i, t in enumerate(parts))
                + f'<pre>{esc(full_p)}</pre>'
                f'<div class="k">训练版（{len(stored_p)} 字符）· 只有任务说明 + 输出约定 + schema，'
                f'没有字段说明和类型要求</div><pre>{esc(stored_p)}</pre>')
    else:
        summary = f'system prompt 构成（{len(full_p)} 字符，本页所有配置共用同一个）'
        body = ('五段拼成，由 <code>build_system_prompt(domain, rich=True, types=True)</code> 生成：'
                + "".join(f'<div class="sprow"><span class="spn">{i+1}</span>{t}</div>'
                          for i, t in enumerate(parts))
                + f'<div class="k">实际发出去的原文</div><pre>{esc(full_p)}</pre>')
    head.append(f'<details class="sp"><summary>{summary}</summary><div class="spbody">{body}'
                + (f'<div class="spnote">示例注入：{esc(rd["shots_note"])}</div>'
                   if rd.get("shots_note") else "")
                + '</div></details>')

    head.append("<table><tr><th>配置</th><th>Precision</th><th>Recall</th><th>micro-F1</th>"
                "<th>macro-F1（字段维度）</th><th>macro-F1（文档维度）</th>"
                "<th>完美率</th><th>全错率</th></tr>")
    for n in names:
        a = aggs[n]
        hl = ' class="hl"' if a["micro"] == best else ""
        head.append(
            f'<tr{hl}><td>{esc(full_name(n))}</td><td class="n">{a["P"]:.3f}</td>'
            f'<td class="n">{a["R"]:.3f}</td>'
            f'<td class="n"><b>{a["micro"]:.3f}</b></td>'
            f'<td class="n">{a["fieldmacro"]:.3f}</td><td class="n">{a["docmacro"]:.3f}</td>'
            f'<td class="n">{a["perfect"]:.0%}</td>'
            f'<td class="n">{a["zero"]:.0%}</td></tr>')
    head.append("</table>")

    # ---- per-field 明细：定位是哪个字段拖后腿 ----
    # macro-F1（字段维度）只报警「有字段偏科」，是哪个字段得看这张表。
    # 带上 gold 实例数，因为实例太少的字段 F1 不可信，不能拿来做优化决策。
    gold_n = Counter(k for i in keep for k, _ in flatten(gold[i]["gt"]))
    fbuckets = {}                       # 配置 -> {字段: [tp, fp, fn]}
    for n in names:
        b = {}
        for st in results[n]:
            for key, items in (("tp", 0), ("fp", 1), ("fn", 2)):
                for k, _ in st[key + "_items"]:
                    b.setdefault(k, [0, 0, 0])[items] += 1
        fbuckets[n] = b
    allf = set(gold_n) | {k for b in fbuckets.values() for k in b}
    # gold 实例多的在前；gold 里不存在的「幻影字段」沉底，但仍逐行列出——
    # 它们每个都占 macro-F1（字段维度）一票，合并展示就看不出票数了。
    order = sorted(allf, key=lambda k: (-gold_n.get(k, 0), k))
    ghost_keys = sorted(k for k in allf if not gold_n.get(k, 0))

    def fcell(b, k, ghost=False):
        v = b.get(k)
        if not v:
            return '<td class="n dim">—</td>'          # 该配置在这个字段上无任何输出
        tp, fp, fn = v
        den = 2 * tp + fp + fn
        f = 2 * tp / den if den else 1.0
        cls = "ok" if f >= 0.95 else ("bad" if f < 0.7 else ("mid" if f < 0.9 else ""))
        # 幻影字段 TP 恒为 0，单看 0.000 看不出所以然，把 FP 摆到台面上
        extra = f'<span class="sub2">FP={fp}</span>' if ghost else ""
        return f'<td class="n {cls}" title="TP={tp} FP={fp} FN={fn}">{f:.3f}{extra}</td>'

    head.append("<h2>各字段表现（per-field）</h2>")
    head.append('<div class="scroll"><table><tr><th>字段</th><th>gold 实例数</th>'
                + "".join(f"<th>{esc(n)}</th>" for n in names) + "</tr>")
    for k in order:
        gn = gold_n.get(k, 0)
        gh = gn == 0
        tag = ('<span class="tag">幻影</span>' if gh else
               '<span class="tag warn">样本少</span>' if gn < 30 else "")
        head.append(f'<tr{" class=ghost" if gh else ""}><td>{esc(k)}{tag}</td>'
                    f'<td class="n">{gn}</td>'
                    + "".join(fcell(fbuckets[n], k, gh) for n in names) + "</tr>")
    head.append("</table></div>")
    ghosts = ghost_keys
    head.append('<div class="note">单元格颜色：'
                '<span class="sw ok">≥0.95</span> <span class="sw">0.90~0.95</span> '
                '<span class="sw mid">0.70~0.90</span> <span class="sw bad">&lt;0.70</span>；'
                '悬停看该字段的 TP/FP/FN。<b>「样本少」(gold 实例 &lt; 30) 的字段 F1 不可靠</b>，'
                '别拿它做优化决策。'
                + (f'<br><b>幻影字段 = gold 里根本不存在的字段路径</b>，'
                   f'本页出现 {len(ghosts)} 个：<code>{esc(" / ".join(ghosts))}</code>'
                   f'（注意没有 <code>menu.</code> 前缀）。成因是模型输出了<b>裸列表</b> '
                   f'<code>[{{...}}]</code> 而不是 <code>{{"menu":[...]}}</code>，拍平后路径整个错位。'
                   f'它们 TP 恒为 0、F1 恒为 0，却同样占 macro-F1（字段维度）的票——'
                   f'<b>这是该指标被拉低的直接原因</b>，剔掉它们重算会明显更高。'
                   if ghosts else "")
                + '</div>')

    # ---- 筛选按钮统计（以最后一个配置为主视角，通常是最好的那个）----
    main = names[-1]
    n_bad     = sum(1 for s in results[main] if s["f1"] < 0.999)
    n_zero    = sum(1 for s in results[main] if s["f1"] <= 0.001)
    n_perfect = sum(1 for s in results[main] if s["f1"] >= 0.999)
    bar = [f'<div class="bar"><button class="on" data-f="all">全部<span class="n">{len(keep)}</span></button>',
           f'<button data-f="bad">「{esc(main)}」未满分<span class="n">{n_bad}</span></button>',
           f'<button data-f="zero">「{esc(main)}」F1=0<span class="n">{n_zero}</span></button>',
           f'<button data-f="perfect">完美 (F1=1)<span class="n">{n_perfect}</span></button>']
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
        if s_main["f1"] >= 0.999: tags.append("perfect")

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
                # 只提示真正影响打分的那一类：抠不出 JSON → 预测集为空 → F1 必为 0。
                # schema 不合法（类型不符）照常参与打分，不值得在卡片上单列。
                + (f'<span class="d fp">JSON 解析失败</span>' if s["status"] == "parse_fail" else "")
                + f'<span class="d">{tok}</span></div>'
                  f'<div class="diff">{diff_html(s["fp_items"],"fp")}{diff_html(s["fn_items"],"fn")}</div>'
                  f'</div></div>')

        cards.append(
            f'<div class="c" data-tags="{",".join(tags)}" data-text="{esc(searchable)}">'
            f'<div class="ch">{"".join(badges)}</div>'
            f'<div class="b">{"".join(body)}</div></div>')

    # ---- 列筛选（仅 matrix 轮次）：按示例数 / 按模型看梯度 ----
    # 每列带 data-shot / data-model，JS 切换时同步隐藏单元格并重算 grid
    colsel = ""
    default_shot = 16 if colmeta else None
    if colmeta:
        m = rd["matrix"]
        shots_avail = sorted({s for _, s in colmeta.values()})
        ladder_models = [mo for mo in m["models"]
                         if len({s for k, (mm, s) in colmeta.items() if mm == mo}) >= 3]
        colsel = ('<div class="bar2"><span class="lbl">列：按示例数</span>'
                  + "".join(f'<button class="colbtn{" on" if s == default_shot else ""}" '
                            f'data-shot="{s}">{s}-shot'
                            f'<span class="n">{sum(1 for _, x in colmeta.values() if x == s)}</span>'
                            f'</button>' for s in shots_avail)
                  + '<span class="sep"></span><span class="lbl">按模型看梯度</span>'
                  + "".join(f'<button class="colbtn" data-model="{esc(mo)}">{esc(mo)}</button>'
                            for mo in ladder_models)
                  + '<span class="sep"></span>'
                  + '<button class="colbtn" data-all="1">全部列</button></div>')

    # ---- 列头（与卡片标题共用同一套 grid，宽度由 CSS 变量 --gcols 驱动）----
    # 列用下标标识（data-ci），JS 按下标同时隐藏「列头 + 所有行」的对应单元格并重算 grid
    colw = 62 if len(names) > 8 else 76

    def attrs(idx, n):
        a = f' data-ci="{idx}"'
        if n in colmeta:
            mo, s = colmeta[n]
            a += f' data-shot="{s}" data-model="{esc(mo)}"'
        return a

    chead = ('<div class="chead">'
             '<div>文档原文（点击展开）</div><div class="unit">#</div>'
             + "".join(f'<div class="cn"{attrs(i, n)}>{esc(n)}</div>'
                       for i, n in enumerate(names))
             + '</div>')
    # 给每行的 f1 单元格按顺序补上 data-ci
    cards2 = []
    for c in cards:
        parts = c.split('<span class="f1')
        rebuilt = parts[0]
        for i, seg in enumerate(parts[1:]):
            rebuilt += f'<span data-ci="{i}" class="f1' + seg
        cards2.append(rebuilt)
    cards = cards2

    init_n = sum(1 for n in names if (not colmeta) or colmeta.get(n, (None, None))[1] == default_shot)
    page = (f'<!doctype html><meta charset=utf-8><title>{esc(rd["title"])}</title>'
            f'<style>{CSS}</style>'
            f'<body data-ncol="{len(names)}" data-colw="{colw}" '
            f'style="--gcols:minmax(110px,1fr) 40px repeat({init_n},{colw}px)">'
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
            f'{"".join(bar)}{colsel}{chead}{"".join(cards)}'
            f'<script>{JS}</script>')
    return dict(html=page, aggs=aggs, names=names, n=len(keep))


ROUTER_FILE = "runs/orchestrator_eval_3domain.jsonl"
ROUTER_LABEL = {"cord": "CORD 英文收据", "duee_fin": "DuEE-fin 中文金融公告",
                "ccks_fraud": "CCKS-fraud 中文反欺诈"}


def build_router():
    """路由页：零样本判类，不是抽取任务，所以指标是准确率/混淆矩阵而非 F1。"""
    rows = load(ROUTER_FILE)
    # 数据里没存原文，用 gold 反查回各域测试集（45/45 可匹配）
    idx = {}
    for dom, (gp, _) in DOMAIN_GOLD.items():
        for g in load(gp):
            idx[(dom, json.dumps(g["gt"], ensure_ascii=False, sort_keys=True))] = g["user"]

    doms = list(ROUTER_LABEL)
    cm = {t: Counter() for t in doms}
    for r in rows:
        cm[r["true_domain"]][r["domain"]] += 1
    n_ok = sum(1 for r in rows if r["domain"] == r["true_domain"])
    acc = n_ok / len(rows)
    route_s = sorted(r["latency"]["route_s"] for r in rows)
    extract_s = sorted(r["latency"]["extract_s"] for r in rows)
    med = lambda a: a[len(a) // 2]

    head = [f'<div class="sub">零样本判类 · {len(rows)} 条（每域 15）· '
            f'准确率 <b>{acc:.1%}</b>（{n_ok}/{len(rows)}）</div>']
    head.append('<div class="note">⚠ 这一页与前三轮<b>不是同一类任务</b>：'
                '前三轮评的是「抽取得准不准」（字段级 P/R/F1），'
                '这里评的是「文档分到哪个域」（分类准确率）。两者不可混读。'
                '<br>⚠ 这批数据由 <b>Ollama 服务的基座</b>跑出（2026-07-11）；'
                '当前代码已改为用 PEFT <code>disable_adapter()</code> 临时回退到基座、不依赖外部服务。'
                '同一个 Qwen3.5-4B、同一个 prompt，但服务栈不同，<b>未重跑验证</b>。</div>')

    rp = build_router_prompt()
    head.append(f'<details class="sp"><summary>路由 prompt（{len(rp)} 字符，'
                f'选项由 SCHEMA_REGISTRY 自动生成）</summary>'
                f'<div class="spbody"><div class="spnote">'
                f'路由不训练分类器，直接让基座零样本判类；域列表来自 schema 注册表，'
                f'加一个新域只需注册 schema，路由 prompt 自动包含它。'
                f'</div><div class="k">实际发出去的原文</div><pre>{esc(rp)}</pre></div></details>')

    head.append("<h2>混淆矩阵</h2>")
    head.append('<div class="scroll"><table><tr><th>真实 \\ 预测</th>'
                + "".join(f"<th>{esc(ROUTER_LABEL[d])}</th>" for d in doms)
                + "<th>该域准确率</th></tr>")
    for t in doms:
        cells = "".join(
            f'<td class="n {"ok" if p == t and cm[t][p] else ("bad" if cm[t][p] else "dim")}">'
            f'{cm[t][p] or "—"}</td>' for p in doms)
        tot = sum(cm[t].values())
        head.append(f'<tr><td>{esc(ROUTER_LABEL[t])}</td>{cells}'
                    f'<td class="n">{cm[t][t]/tot:.1%}</td></tr>')
    head.append("</table></div>")

    head.append("<h2>延迟拆解</h2>")
    head.append('<table><tr><th>环节</th><th>中位</th><th>p90</th><th>占端到端</th></tr>'
                f'<tr><td>路由（零样本判类）</td><td class="n">{med(route_s):.2f}s</td>'
                f'<td class="n">{route_s[int(len(route_s)*0.9)]:.2f}s</td>'
                f'<td class="n">{med(route_s)/(med(route_s)+med(extract_s)):.0%}</td></tr>'
                f'<tr><td>抽取（切 adapter 后）</td><td class="n">{med(extract_s):.2f}s</td>'
                f'<td class="n">{extract_s[int(len(extract_s)*0.9)]:.2f}s</td>'
                f'<td class="n">{med(extract_s)/(med(route_s)+med(extract_s)):.0%}</td></tr>'
                "</table>")
    share = med(route_s) / (med(route_s) + med(extract_s))
    head.append('<div class="note">Mac M1 MPS 未优化配置。'
                f'<b>路由只占端到端的 {share:.0%}，而且它复用同一个常驻基座</b>——'
                '不额外占显存、不需要第二个模型、不依赖外部服务。'
                '加一个新域的边际成本只是训一个 adapter，路由侧零改动。</div>')

    cards = []
    for i, r in enumerate(rows):
        txt = idx[(r["true_domain"], json.dumps(r["gold"], ensure_ascii=False, sort_keys=True))]
        ok = r["domain"] == r["true_domain"]
        cards.append(
            f'<div class="c" data-tags="{"ok" if ok else "err"}" data-text="{esc(txt[:400])}">'
            f'<div class="ch rch">'
            f'<div class="q">{esc(txt[:110])}</div>'
            f'<div class="p id">#{i}</div>'
            f'<div class="p">{esc(ROUTER_LABEL[r["true_domain"]])}</div>'
            f'<div class="p {"ok" if ok else "bad"}">{esc(ROUTER_LABEL[r["domain"]])}</div>'
            f'<div class="p">{r["latency"]["route_s"]:.2f}s</div></div>'
            f'<div class="b"><div class="k">输入原文</div><pre>{esc(txt[:1500])}</pre></div></div>')

    n_err = len(rows) - n_ok
    bar = ('<div class="bar">'
           f'<button class="on" data-f="all">全部<span class="n">{len(rows)}</span></button>'
           f'<button data-f="ok">判对<span class="n">{n_ok}</span></button>'
           f'<button data-f="err">判错<span class="n">{n_err}</span></button>'
           '<input type="search" id="q" placeholder="搜原文…"></div><div class="hit" id="hit"></div>')
    chead = ('<div class="chead rch"><div>文档原文（点击展开）</div><div class="unit">#</div>'
             '<div class="cn">真实域</div><div class="cn">路由判定</div>'
             '<div class="cn">路由耗时</div></div>')

    return (f'<!doctype html><meta charset=utf-8><title>路由 · 零样本文档判类</title>'
            f'<style>{CSS}</style><body>'
            f'<a class="back" href="index.html">← 返回评测目录</a>'
            f'<h1>路由 · 零样本文档判类</h1>'
            f'{"".join(head)}'
            f'<h2>逐条明细</h2>{bar}{chead}{"".join(cards)}'
            f'<script>{JS}</script>')


def build_index(built):
    rrows = load(ROUTER_FILE)
    router_n = len(rrows)
    router_acc = sum(1 for r in rrows if r["domain"] == r["true_domain"]) / router_n
    _rs = sorted(r["latency"]["route_s"] for r in rrows)
    _es = sorted(r["latency"]["extract_s"] for r in rrows)
    router_med = _rs[len(_rs) // 2]
    router_share = router_med / (router_med + _es[len(_es) // 2])

    rows = []
    for rd in ROUNDS:
        b = built[rd["slug"]]
        best_name = max(b["names"], key=lambda n: b["aggs"][n]["micro"])
        a = b["aggs"][best_name]
        rows.append((rd, b, best_name, a))

    def card_score(rd, b, best_name, a):
        """微调轮给「微调前 → 微调后」，API 轮只有最佳配置可报。"""
        base = rd.get("baseline")
        if base and base in b["aggs"]:
            bm = b["aggs"][base]["micro"]
            return (f'微调前 {esc(base)} · <b>{bm:.3f}</b>'
                    f'<span class="arrow">→</span>'
                    f'微调后 {esc(best_name)} · <b>{a["micro"]:.3f}</b>'
                    f'<span class="gain">+{a["micro"]-bm:.3f}</span>')
        return f'最佳 {esc(best_name)} · micro-F1 <b>{a["micro"]:.3f}</b>'

    cards = "".join(
        f'<a class="card" href="{rd["slug"]}.html"><div class="t">{esc(rd["title"])}</div>'
        f'<div class="d2">{esc(rd["desc"][:60])}…</div>'
        f'<div class="m">{b["n"]} 条 · {len(b["names"])} 个配置<br>'
        f'{card_score(rd, b, best_name, a)}</div></a>'
        for rd, b, best_name, a in rows)

    # CORD 三轮纵向对比（同一批 92 条，可比）
    c1 = built["round1_api_zeroshot"]["aggs"]
    c2 = built["round2_api_fewshot"]["aggs"]
    c3 = built["round3_cord"]["aggs"]
    ladder = [("API 裸跑最佳（Qwen3.7-Max）", c1["Qwen3.7-Max"]),
              ("API few-shot 最佳（Gemini 32-shot）", c2["Gemini 32"]),
              ("本地基座 4B（零示例）", c1["本地基座 4B"]),
              ("本地基座 4B + 32 示例", c3["基座 +32示例"]),
              ("本地 LoRA 微调", c3["LoRA 微调"])]

    ladder_rows = "".join(
        f'<tr{" class=hl" if "LoRA" in n else ""}><td>{esc(n)}</td>'
        f'<td class="n">{a["P"]:.3f}</td><td class="n">{a["R"]:.3f}</td>'
        f'<td class="n"><b>{a["micro"]:.3f}</b></td>'
        f'<td class="n">{a["fieldmacro"]:.3f}</td><td class="n">{a["docmacro"]:.3f}</td>'
        f'<td class="n">{a["perfect"]:.0%}</td><td class="n">{a["zero"]:.0%}</td></tr>'
        for n, a in ladder)

    return (f'<!doctype html><meta charset=utf-8><title>评测明细 · 通用大模型 API vs 本地微调</title>'
            f'<style>{CSS}</style>'
            f'<h1>评测明细</h1>'
            f'<div class="sub">通用大模型 API vs 本地微调小模型 · 逐条可查 · '
            f'指标口径与结论见 <code>runs/model_selection_report.md</code></div>'

            f'<h2>CORD 三轮纵向对比（同一批干净 92 条，可直接比）</h2>'
            f'<table><tr><th>方案</th><th>Precision</th><th>Recall</th><th>micro-F1</th>'
            f'<th>macro-F1（字段维度）</th><th>macro-F1（文档维度）</th>'
            f'<th>完美率</th><th>全错率</th></tr>{ladder_rows}</table>'
            f'<h2>各轮入口</h2>{cards}'
            f'<a class="card" href="router.html"><div class="t">附 · 路由：零样本文档判类</div>'
            f'<div class="d2">不训练分类器，直接让同一个基座判文档属于哪个域，再切到对应 adapter。…</div>'
            f'<div class="m">{router_n} 条 · 每域 15<br>'
            f'准确率 <b>{router_acc:.1%}</b><span class="arrow">·</span>'
            f'路由耗时中位 {router_med:.2f}s，占端到端 {router_share:.0%}</div></a>'
)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[build] router ...", flush=True)
    rp = os.path.join(OUT, "router.html")
    with open(rp, "w") as f:
        f.write(build_router())
    print(f"        -> {rp}  ({os.path.getsize(rp)//1024} KB)")
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
