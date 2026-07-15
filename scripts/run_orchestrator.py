"""端到端 Orchestrator：文档 → 路由 → 选 adapter+schema → 抽取 → 标准化 JSON。

架构：
  文本 → [路由(共享基座零样本，禁用adapter)] → domain
       → [共享基座 + PEFT 多adapter切换(set_adapter)] → 该域抽取
       → 标准化 JSON 输出

一个基座常驻，多个 adapter 按需切换 —— 本质和 vLLM 多 LoRA 服务是同一个概念，
这里用 PEFT 原生多 adapter 支持在本地跑通，验证整条链路。路由复用同一个已加载的
基座（用 PEFT 的 disable_adapter() 临时回退到基座权重），不依赖任何外部服务
（如 Ollama），这样才能部署到没有本地服务的容器环境（如 HF Spaces）。

用法：
  # 单条测试
  uv run python scripts/run_orchestrator.py --text "6月3日下午发布公告称,公司接到股东傅宇晨的通知..."

  # 端到端评测(混合两个域的 test 集)
  uv run python scripts/run_orchestrator.py --eval --n-per-domain 20 --out runs/orchestrator_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.router import build_router_prompt, parse_route_output
from shared.schema import SCHEMA_REGISTRY, build_system_prompt

ADAPTERS = {"cord": "adapters/cord", "duee_fin": "adapters/duee_fin", "ccks_fraud": "adapters/ccks_fraud"}
DOMAIN_FILES = {"cord": "data/cord/test.eval.jsonl", "duee_fin": "data/duee_fin_cn/test.eval.jsonl",
                "ccks_fraud": "data/ccks_fraud_cn/test.eval.jsonl"}

ROUTER_SYSTEM = build_router_prompt()


class Orchestrator:
    """加载一次基座 + 全部注册 adapter，之后 extract() 按域切换。"""

    def __init__(self, base: str = "Qwen/Qwen3.5-4B"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from peft import PeftModel

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        dtype = torch.float32 if self.device == "cpu" else "auto"  # bf16/fp16 on CPU is slow/unreliable
        print(f"[orchestrator] 加载基座 {base} (device={self.device}, dtype={dtype}) ...")
        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype, trust_remote_code=True)

        names = list(ADAPTERS.keys())
        first = names[0]
        print(f"[orchestrator] 加载 adapter '{first}' <- {ADAPTERS[first]}")
        self.model = PeftModel.from_pretrained(model, ADAPTERS[first], adapter_name=first).to(self.device)
        for name in names[1:]:
            print(f"[orchestrator] 加载 adapter '{name}' <- {ADAPTERS[name]}")
            self.model.load_adapter(ADAPTERS[name], adapter_name=name)
        self.model.eval()
        print(f"[orchestrator] 就绪，已加载 adapter: {names}")

    def route(self, text: str) -> str | None:
        """基座零样本判类，临时禁用所有 adapter（回退到冻结的基座权重），
        不依赖任何外部服务——复用同一个已加载的模型做路由。"""
        import torch
        messages = [{"role": "system", "content": ROUTER_SYSTEM},
                    {"role": "user", "content": text[:1500]}]
        try:
            prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad(), self.model.disable_adapter():
            out = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        raw = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_route_output(raw)

    def extract(self, text: str, domain: str, max_new_tokens: int = 1024) -> str:
        import torch
        self.model.set_adapter(domain)
        system = build_system_prompt(domain, rich=True, types=True)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        try:
            prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def run(self, text: str) -> dict:
        t0 = time.time()
        domain = self.route(text)
        t_route = time.time() - t0
        if domain is None:
            return {"domain": None, "output": None, "error": "路由失败/未识别域"}
        t1 = time.time()
        output = self.extract(text, domain)
        t_extract = time.time() - t1
        return {"domain": domain, "output": output,
                "latency": {"route_s": round(t_route, 2), "extract_s": round(t_extract, 2)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--text", default=None, help="单条测试文本")
    ap.add_argument("--eval", action="store_true", help="混合两个域test集做端到端评测")
    ap.add_argument("--n-per-domain", type=int, default=20)
    ap.add_argument("--out", default="runs/orchestrator_eval.jsonl")
    args = ap.parse_args()

    orch = Orchestrator(args.base)

    if args.text:
        result = orch.run(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.eval:
        samples = []  # (true_domain, text, gold)
        for domain, path in DOMAIN_FILES.items():
            rows = [json.loads(l) for l in open(path) if l.strip()][: args.n_per_domain]
            for r in rows:
                samples.append((domain, r["user"], r["gt"]))

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        n_correct_route = 0
        with open(args.out, "w") as fout:
            for i, (true_domain, text, gold) in enumerate(samples):
                result = orch.run(text)
                result["true_domain"] = true_domain
                result["gold"] = gold
                if result["domain"] == true_domain:
                    n_correct_route += 1
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(samples)}  路由准确率(截至目前)={n_correct_route/(i+1):.1%}")

        print(f"\n完成: {len(samples)} 条 -> {args.out}")
        print(f"路由准确率: {n_correct_route}/{len(samples)} = {n_correct_route/len(samples):.1%}")


if __name__ == "__main__":
    main()
