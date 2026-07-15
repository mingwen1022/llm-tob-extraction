"""路由器：判断输入文本属于哪个已注册域，从而选择对应的 adapter + schema。

用法（零样本，基座自己判类，不需要训练）：
  from shared.router import route
  domain = route(text, model, tokenizer)   # -> "cord" | "duee_fin"

设计取舍：没有用语言/关键词规则（虽然当前两个域碰巧一个英文一个中文，
规则会"作弊式"地接近100%），而是用基座零样本判类 —— 这样以后加同语言的新域
时路由逻辑不用重写，是真正可扩展的设计。
"""
from __future__ import annotations

from .schema import SCHEMA_REGISTRY

ROUTER_SYSTEM_TEMPLATE = """你是文档分类路由器。判断输入文本属于以下哪个类型，只输出类型的英文key，不要输出其他任何内容。

可选类型：
{options}

只输出 key（如 "{example_key}"），不要解释、不要标点、不要多余文字。"""


def build_router_prompt() -> str:
    lines = []
    for key, info in SCHEMA_REGISTRY.items():
        lines.append(f'  "{key}": {info["task"]}')
    example_key = next(iter(SCHEMA_REGISTRY))
    return ROUTER_SYSTEM_TEMPLATE.format(options="\n".join(lines), example_key=example_key)


def parse_route_output(raw: str) -> str | None:
    """从模型输出里提取合法的 domain key（做容错：可能带引号/多余文字）。"""
    raw = raw.strip().strip('"').strip("'").strip()
    # 优先精确匹配
    if raw in SCHEMA_REGISTRY:
        return raw
    # 容错：输出里包含某个 key
    for key in SCHEMA_REGISTRY:
        if key in raw:
            return key
    return None


if __name__ == "__main__":
    print(build_router_prompt())
