"""Run a model over an eval file and dump raw outputs (for shared.eval).

Covers experiments E0/E1/E2/E3:
  E0  base, free decode      : --base <id>
  E1  base, constrained      : --base <id> --constrained
  E2  base+LoRA, free        : --base <id> --adapter adapters/cord
  E3  base+LoRA, constrained : --base <id> --adapter adapters/cord --constrained

Runs on Mac (MPS) via transformers + peft. Constrained decoding via Outlines.

Example:
  python scripts/run_inference.py --base Qwen/Qwen3.5-4B \
      --adapter adapters/cord --constrained \
      --eval-file data/cord/test.eval.jsonl --out runs/e3.jsonl

Then:
  python -m shared.eval --pred runs/e3.jsonl --gold data/cord/test.eval.jsonl \
      --name "E3 LoRA·约束"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.schema import get_model


def load_eval(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF id, e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default=None, help="path to HF PEFT adapter (omit for base)")
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="cord")
    ap.add_argument("--constrained", action="store_true", help="JSON-Schema constrained decoding")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # GDN/新算子回退CPU
    import torch
    from transformers import AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}  base={args.base}  adapter={args.adapter}  constrained={args.constrained}")

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    # Qwen3.5 训练时是 VL 结构(语言主干在 .language_model 下)，必须用 VL 类加载，
    # 否则 LoRA 的 key 路径(.model.language_model.layers) 对不上、adapter 不会被应用。
    model = None
    last_err = None
    for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        try:
            import transformers
            cls = getattr(transformers, loader)
            model = cls.from_pretrained(args.base, dtype="auto", trust_remote_code=True)
            print(f"loaded via {loader}")
            break
        except Exception as e:
            last_err = e
            print(f"[skip] {loader}: {str(e)[:120]}")
    if model is None:
        raise RuntimeError(f"无法加载基座: {last_err}")
    model = model.to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).to(device)
        # 自检：确认 LoRA 真的挂上了（训练后 lora_B 应非零）
        import torch as _t
        bsum = sum(p.abs().sum().item() for n, p in model.named_parameters() if "lora_B" in n)
        nlora = sum(1 for n, _ in model.named_parameters() if "lora_B" in n)
        print(f"[check] lora_B 模块数={nlora}  |lora_B|合计={bsum:.4f}  "
              f"{'✅ adapter 已生效' if bsum > 0 else '❌ adapter 全为0/未挂上!'}")
    model.eval()

    rows = load_eval(args.eval_file, args.limit)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # constrained generator (built once)
    gen_json = None
    if args.constrained:
        import outlines
        om = outlines.models.Transformers(model, tok)
        gen_json = outlines.generate.json(om, get_model(args.domain))

    t0 = time.time()
    with open(args.out, "w") as fout:
        for i, r in enumerate(rows):
            messages = [
                {"role": "system", "content": r["system"]},
                {"role": "user", "content": r["user"]},
            ]
            try:
                prompt = tok.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if gen_json is not None:
                obj = gen_json(prompt)                      # returns a pydantic instance
                output = obj.model_dump_json()
            else:
                inputs = tok(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False)
                output = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            fout.write(json.dumps({"output": output}, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)}")

    dt = time.time() - t0
    print(f"done: {len(rows)} samples in {dt:.1f}s ({dt/max(len(rows),1):.2f}s/sample) -> {args.out}")


if __name__ == "__main__":
    main()
