"""Robust extraction of a JSON object from raw model output.

Models often wrap output in ```json ... ``` fences or add stray prose. We strip
that and parse the first balanced {...} block. Returns (obj, status):
  status in {"valid", "parse_fail"}.
"""
from __future__ import annotations

import json
import re
from typing import Optional, Tuple

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_block(text: str) -> Optional[str]:
    if text is None:
        return None
    # 1) prefer fenced block
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    # 2) take first balanced {...}
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def safe_load(text: str) -> Tuple[Optional[dict], str]:
    block = extract_json_block(text)
    if block is None:
        return None, "parse_fail"
    try:
        return json.loads(block), "valid"
    except json.JSONDecodeError:
        return None, "parse_fail"


if __name__ == "__main__":
    raw = '好的：\n```json\n{"total": {"total_price": "364,100"}}\n```\n完成'
    print(safe_load(raw))
    print(safe_load("not json here"))
