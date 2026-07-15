"""Run baselines via Ollama (Mac-friendly, no transformers/MPS needed).

E0  base, free decode : --model qwen3.5:4b
E1  base, constrained : --model qwen3.5:4b --constrained   (Ollama structured output)

Outputs {"output": "<raw text>"} per line, consumed by shared.eval.

Example:
  python scripts/run_ollama.py --model qwen3.5:4b \
      --eval-file data/cord/test.eval.jsonl --out runs/e0.jsonl
  python -m shared.eval --pred runs/e0.jsonl --gold data/cord/test.eval.jsonl --name "E0 基座·自由"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.schema import build_system_prompt, flat_json_schema, get_model

HOST = "http://localhost:11434"


def chat(model: str, messages: list[dict], fmt=None, think=False, timeout=300) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,          # extraction needs no CoT; off = much faster
        "options": {"temperature": 0},
    }
    if fmt is not None:
        payload["format"] = fmt   # JSON schema -> constrained/structured output
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--eval-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="cord")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--think", action="store_true", help="enable thinking mode (slow)")
    ap.add_argument("--rich", action="store_true", help="用带字段说明的 rich system prompt")
    ap.add_argument("--types", action="store_true", help="prompt 里加'值一律输出字符串'的类型说明")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    override_system = (build_system_prompt(args.domain, rich=args.rich, types=args.types)
                       if (args.rich or args.types) else None)

    rows = [json.loads(l) for l in open(args.eval_file) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    fmt = flat_json_schema(get_model(args.domain)) if args.constrained else None

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    with open(args.out, "w") as fout:
        for i, r in enumerate(rows):
            messages = [
                {"role": "system", "content": override_system or r["system"]},
                {"role": "user", "content": r["user"]},
            ]
            try:
                output = chat(args.model, messages, fmt=fmt, think=args.think)
            except Exception as e:
                output = f"__ERROR__ {e}"
            fout.write(json.dumps({"output": output}, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)}  ({(time.time()-t0)/(i+1):.2f}s/sample)")

    dt = time.time() - t0
    print(f"done: {len(rows)} samples in {dt:.1f}s ({dt/max(len(rows),1):.2f}s/sample) -> {args.out}")


if __name__ == "__main__":
    main()
