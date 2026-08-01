"""多场景 toB 结构化抽取 · Orchestrator 产品 Demo

一个入口：贴任意文档文本 → 自动识别类型(路由) → 选对应 LoRA adapter → 抽取标准化 JSON。
同时展示本地 LoRA vs 直接调大模型 API 的选型对比（精度/成本/合规）。

用法：
  uv run python scripts/demo_app.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr

from run_orchestrator import Orchestrator, DOMAIN_FILES

DOMAIN_LABEL = {"cord": "🧾 票据/收据抽取", "duee_fin": "📈 金融公告事件抽取",
                "ccks_fraud": "🚨 金融风控/反欺诈案例抽取"}

N_EXAMPLES_PER_DOMAIN = 8

EXAMPLES: dict[str, list[str]] = {}
for domain, path in DOMAIN_FILES.items():
    try:
        with open(path) as f:
            rows = [json.loads(l) for _, l in zip(range(N_EXAMPLES_PER_DOMAIN), f)]
        EXAMPLES[domain] = [r["user"][:600] for r in rows] or [""]
    except Exception:
        EXAMPLES[domain] = [""]


def pick_example(domain: str) -> str:
    return random.choice(EXAMPLES.get(domain, [""]))


_orch: Orchestrator | None = None


def get_orch() -> Orchestrator:
    global _orch
    if _orch is None:
        _orch = Orchestrator()
    return _orch


def run_demo(text: str):
    if not text or not text.strip():
        return "—", "{}", "请输入或选择一个样例文本"
    orch = get_orch()
    t0 = time.time()
    result = orch.run(text.strip())
    dt = time.time() - t0

    domain = result.get("domain")
    if domain is None:
        return "❌ 未识别", "{}", f"路由失败: {result.get('error')}"

    label = DOMAIN_LABEL.get(domain, domain)
    try:
        parsed = json.loads(result["output"])
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        pretty = result["output"]

    lat = result.get("latency", {})
    info = (f"识别为「{label}」→ 路由到 adapter `{domain}` → 用对应 schema 约束抽取\n"
           f"延迟：路由 {lat.get('route_s','?')}s + 抽取 {lat.get('extract_s','?')}s "
           f"= 总计 {dt:.2f}s（{orch.device}，未做 vLLM 生产优化）")
    return label, pretty, info


with gr.Blocks(title="多场景 toB 结构化抽取 Orchestrator") as demo:
    gr.Markdown("# 🧩 多场景 toB 结构化抽取 · Orchestrator")
    gr.Markdown(
        "**一个入口，自动识别文档类型，路由到对应领域专家模型，输出标准化 JSON。**\n\n"
        "架构：`文本 → 路由(零样本判类) → 选 LoRA adapter + schema → 约束抽取 → 结构化输出`\n"
        "共享基座 Qwen3.5-4B + 多个领域 LoRA 插件（票据 / 金融公告 / 反欺诈），无需调用方指定文档类型。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="输入文档文本（任意来源：OCR文本 / 网页正文 / 公告全文）",
                             lines=10, placeholder="贴入小票文本或金融公告文本...")
            with gr.Row():
                btn_cord = gr.Button("📄 试试票据样例")
                btn_duee = gr.Button("📄 试试金融公告样例")
                btn_ccks = gr.Button("📄 试试反欺诈样例")
            submit = gr.Button("🚀 提取结构化数据", variant="primary")
        with gr.Column(scale=1):
            domain_out = gr.Textbox(label="识别的文档类型（路由结果）")
            json_out = gr.Code(label="抽取结果（标准化 JSON）", language="json")
            info_out = gr.Textbox(label="处理链路 & 延迟", lines=3)

    btn_cord.click(lambda: pick_example("cord"), outputs=inp)
    btn_duee.click(lambda: pick_example("duee_fin"), outputs=inp)
    btn_ccks.click(lambda: pick_example("ccks_fraud"), outputs=inp)
    submit.click(run_demo, inputs=inp, outputs=[domain_out, json_out, info_out])

if __name__ == "__main__":
    print("[demo] 首次运行会加载基座+全部adapter，请耐心等待...")
    if os.environ.get("SPACE_ID"):
        demo.launch()  # HF Spaces 环境自动处理 host/port
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860)
