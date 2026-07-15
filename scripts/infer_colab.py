"""在 Colab(GPU) 上用训好的 adapter 跑 test 集，产出预测文件，下载回本地评测。

Colab 用法（与训练同一 session，或重开都行）：
  1) 上传 cord_adapter/（或解压 cord_adapter.zip）+ test.eval.jsonl
  2) !pip install -q unsloth        # 若新 session
  3) !python infer_colab.py --adapter cord_adapter --eval-file test.eval.jsonl --out e2_pred.jsonl
  4) 下载 e2_pred.jsonl 回本地

本地评测：
  uv run python -m shared.eval --pred e2_pred.jsonl --gold data/cord/test.eval.jsonl --name "E2 LoRA·自由"
"""
from __future__ import annotations

import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="cord_adapter", help="LoRA adapter 目录")
    ap.add_argument("--eval-file", default="test.eval.jsonl")
    ap.add_argument("--out", default="e2_pred.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from unsloth import FastModel

    # 直接加载 adapter 目录：unsloth 会自动读 base(unsloth/Qwen3.5-4B) + 套上 LoRA
    model, tok = FastModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=2048,
        load_in_4bit=False,
    )
    FastModel.for_inference(model)

    rows = [json.loads(l) for l in open(args.eval_file) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    def render(msgs):
        # 关思考模式（抽取不需要 CoT）；不支持该 kwarg 时回退
        try:
            return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors="pt")
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt")

    t0 = time.time()
    with open(args.out, "w") as fout:
        for i, r in enumerate(rows):
            msgs = [{"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["user"]}]
            ids = render(msgs).to("cuda")
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            fout.write(json.dumps({"output": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(rows)}  ({(time.time()-t0)/(i+1):.2f}s/sample)")

    dt = time.time() - t0
    print(f"done: {len(rows)} in {dt:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
