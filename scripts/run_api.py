"""E4 前沿 API 对照：调各家大模型在 CORD test 上跑，复用 shared.eval 打分。

OpenAI 兼容接口，默认指向 OpenRouter（一个 key 调所有模型）。换 --model 即换模型。
纯 API 调用，不需要 GPU。

准备：
  uv sync --extra baseline            # 装 openai SDK
  export OPENROUTER_API_KEY=sk-or-...

用法（OpenRouter）：
  uv run python scripts/run_api.py --model openai/gpt-4o \
      --eval-file data/cord/test.eval.jsonl --out runs/e4_gpt4o.jsonl
  uv run python scripts/run_api.py --model deepseek/deepseek-chat --out runs/e4_deepseek.jsonl
  uv run python scripts/run_api.py --model google/gemini-2.5-flash --out runs/e4_gemini.jsonl
  uv run python scripts/run_api.py --model qwen/qwen-max --out runs/e4_qwenmax.jsonl

也可直连别家（改 base-url + key 环境变量）：
  --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --model deepseek-chat

评测：
  uv run python -m shared.eval --pred runs/e4_gpt4o.jsonl --gold data/cord/test.eval.jsonl --name "E4 GPT-4o"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()  # 读项目根目录 .env（GMI_API_KEY 等）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="如 openai/gpt-4o、deepseek/deepseek-chat")
    ap.add_argument("--eval-file", default="data/cord/test.eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--json-mode", action="store_true", help="开启 response_format=json_object")
    ap.add_argument("--domain", default="cord")
    ap.add_argument("--full-prompt", action="store_true",
                    help="用完整 prompt(字段说明+类型要求)，公平对照，替代 eval 文件里的简版")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="部分模型(如 Kimi)只允许 temperature=1")
    ap.add_argument("--no-temperature", action="store_true",
                    help="不传 temperature(如 claude-sonnet-5 弃用了该参数)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=90, help="单请求超时(秒)")
    ap.add_argument("--retries", type=int, default=4, help="429/超时的重试次数(指数退避)")
    ap.add_argument("--shots", type=int, default=0,
                    help="in-context 示例条数(实验B)。取 --shot-file 的固定前N条，不随机，可复现")
    ap.add_argument("--shot-file", default="data/cord/train.eval.jsonl",
                    help="few-shot 示例来源(必须是训练集，不能与测试集重叠)")
    args = ap.parse_args()

    from openai import OpenAI

    key = os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"缺少 API key：请 export {args.api_key_env}=...")
    client = OpenAI(base_url=args.base_url, api_key=key)

    rows = [json.loads(l) for l in open(args.eval_file) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    override_system = None
    if args.full_prompt:
        from shared.schema import build_system_prompt
        override_system = build_system_prompt(args.domain, rich=True, types=True)

    # ---- few-shot 示例（实验B）----
    # 取固定前 N 条（不随机，保证可复现），拼成 user/assistant 交替的 in-context 示例。
    # 污染核查：示例绝不能出现在测试集里，否则 few-shot 等于泄题。
    shot_msgs: list[dict] = []
    if args.shots > 0:
        shot_rows = [json.loads(l) for l in open(args.shot_file) if l.strip()][: args.shots]
        if len(shot_rows) < args.shots:
            sys.exit(f"--shot-file 只有 {len(shot_rows)} 条，不足 {args.shots} 条")
        test_texts = {r["user"] for r in rows}
        leaked = [i for i, s in enumerate(shot_rows) if s["user"] in test_texts]
        if leaked:
            sys.exit(f"few-shot 污染：示例 {leaked} 出现在测试集里，换 --shot-file 或调整取法")
        for s in shot_rows:
            shot_msgs.append({"role": "user", "content": s["user"]})
            shot_msgs.append({"role": "assistant",
                              "content": json.dumps(s["gt"], ensure_ascii=False)})
        print(f"[few-shot] {args.shots} 条示例 <- {args.shot_file}（已核查与测试集零重叠）")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    extra = {"response_format": {"type": "json_object"}} if args.json_mode else {}
    # 部分模型(如 claude-sonnet-5)弃用了 temperature 参数 → --no-temperature 就不传
    if not args.no_temperature:
        extra["temperature"] = args.temperature

    t0 = time.time()
    tok_in = tok_out = 0
    with open(args.out, "w") as fout:
        for i, r in enumerate(rows):
            messages = [
                {"role": "system", "content": override_system or r["system"]},
                *shot_msgs,
                {"role": "user", "content": r["user"]},
            ]
            output, delay = None, 5
            row_in = row_out = 0
            for attempt in range(args.retries + 1):
                try:
                    resp = client.chat.completions.create(
                        model=args.model, messages=messages, timeout=args.timeout,
                        max_tokens=args.max_tokens, **extra,
                    )
                    output = resp.choices[0].message.content or ""
                    if resp.usage:
                        row_in = resp.usage.prompt_tokens or 0
                        row_out = resp.usage.completion_tokens or 0
                        tok_in += row_in
                        tok_out += row_out
                    break
                except Exception as e:
                    if attempt == args.retries:
                        output = f"__ERROR__ {e}"
                    else:
                        time.sleep(delay)          # 429/超时 → 退避重试
                        delay = min(delay * 2, 60)
            fout.write(json.dumps({"output": output, "tok_in": row_in, "tok_out": row_out},
                                  ensure_ascii=False) + "\n")
            fout.flush()          # 长任务要能实时看进度/断点续跑
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(rows)}  ({(time.time()-t0)/(i+1):.2f}s/条)")

    dt = time.time() - t0
    n = len(rows)
    print(f"done: {n} 条 · {dt:.1f}s · shots={args.shots} · "
          f"tokens in={tok_in} out={tok_out} -> {args.out}")
    print(f"  单条均值: in={tok_in/n:.0f} out={tok_out/n:.0f} tok  ({dt/n:.2f}s/条)")
    print(f"（成本 = tokens × 该模型单价，用 in/out token 数自行乘）")


if __name__ == "__main__":
    main()
