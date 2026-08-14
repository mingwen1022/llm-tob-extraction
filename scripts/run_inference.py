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
    ap.add_argument("--full-prompt", action="store_true",
                    help="用完整 prompt(字段说明+类型要求)覆盖 eval 文件里存的 system")
    ap.add_argument("--shots", type=int, default=0,
                    help="in-context 示例条数。与 run_api.py --shots 同一套取法与污染核查，"
                         "以便「基座+示例」和「API+示例」严格可比")
    ap.add_argument("--shot-file", default=None,
                    help="few-shot 示例来源，默认取同域训练集")
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

    override_system = None
    if args.full_prompt:
        from shared.schema import build_system_prompt
        override_system = build_system_prompt(args.domain, rich=True, types=True)

    # ---- few-shot：与 run_api.py 完全同一套取法 ----
    # 取「前 N 条干净的」：训练集里可能混着测试集原文（CORD 官方 split 自带跨 split 重复），
    # 直接取前 N 条会把测试题当示例喂进去。跳过污染项继续往后取，确定且可复现。
    shot_msgs = []
    if args.shots > 0:
        shot_file = args.shot_file or args.eval_file.replace("test.eval", "train.eval")
        pool = load_eval(shot_file)
        test_texts = {r["user"] for r in rows}
        clean = [s for s in pool if s["user"] not in test_texts]
        n_skipped = len(pool) - len(clean)
        shot_rows = clean[: args.shots]
        if len(shot_rows) < args.shots:
            sys.exit(f"--shot-file 去污染后只剩 {len(clean)} 条，不足 {args.shots} 条")
        if n_skipped:
            print(f"[few-shot] 跳过 {n_skipped} 条与测试集重叠的样本")
        for s in shot_rows:
            shot_msgs.append({"role": "user", "content": s["user"]})
            shot_msgs.append({"role": "assistant",
                              "content": json.dumps(s["gt"], ensure_ascii=False)})
        print(f"[few-shot] {args.shots} 条示例 <- {shot_file}（已核查与测试集零重叠）")

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
                {"role": "system", "content": override_system or r["system"]},
                *shot_msgs,
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
            fout.flush()          # 长任务要能实时看进度/断点续跑
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)}")

    dt = time.time() - t0
    print(f"done: {len(rows)} samples in {dt:.1f}s ({dt/max(len(rows),1):.2f}s/sample) -> {args.out}")


if __name__ == "__main__":
    main()
