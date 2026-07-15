"""列出 GMI Cloud 上可用的模型（OpenAI 兼容 GET /v1/models），
这样不用手动去网页复制 model ID。

用法：
  # 需要 .env 里已填 GMI_API_KEY
  uv run python scripts/list_gmi_models.py
  uv run python scripts/list_gmi_models.py --filter deepseek     # 按关键词过滤
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://api.gmi-serving.com/v1")
    ap.add_argument("--api-key-env", default="GMI_API_KEY")
    ap.add_argument("--filter", default=None, help="按关键词过滤 model id（大小写不敏感）")
    args = ap.parse_args()

    key = os.environ.get(args.api_key_env)
    if not key:
        sys.exit(f"缺少 API key：请在 .env 里填 {args.api_key_env}=sk-...")

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=key)

    models = client.models.list()
    ids = sorted(m.id for m in models.data)
    if args.filter:
        ids = [i for i in ids if args.filter.lower() in i.lower()]

    print(f"共 {len(ids)} 个模型" + (f"（过滤: {args.filter}）" if args.filter else "") + "：")
    for i in ids:
        print(" ", i)


if __name__ == "__main__":
    main()
