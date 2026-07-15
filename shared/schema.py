"""Domain schemas for the structured-extraction Agent.

One Schema is defined once and reused in three places:
  1. system prompt  -> tells the model the output shape
  2. constrained decoding (Outlines) -> forces valid JSON
  3. eval -> schema-level validity check

Phase 1 ships the CORD (receipt) schema. Phase 2 adds NDA / invoice / finance
by registering more (name -> pydantic model) entries in SCHEMA_REGISTRY.
"""
from __future__ import annotations

import json
from typing import Optional, Type

from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# CORD (receipt) — high-frequency subset. Start small, expand later.
# --------------------------------------------------------------------------- #
class MenuItem(BaseModel):
    nm: Optional[str] = None      # item name
    cnt: Optional[str] = None     # quantity
    price: Optional[str] = None   # line price


class SubTotal(BaseModel):
    subtotal_price: Optional[str] = None
    tax_price: Optional[str] = None
    service_price: Optional[str] = None


class Total(BaseModel):
    total_price: Optional[str] = None
    cashprice: Optional[str] = None
    changeprice: Optional[str] = None


class Receipt(BaseModel):
    menu: list[MenuItem] = []
    sub_total: Optional[SubTotal] = None
    total: Optional[Total] = None


# --------------------------------------------------------------------------- #
# DuEE-fin（公司公告事件抽取）— 质押/股份回购/股东减持 三类合并的统一 schema
# 一篇公告可能含多个事件 → events 是列表；每个事件用 21 字段扁平 schema，缺失填 null
# 注意：原始角色名含 "/" 的已转 "_"（Pydantic 字段名不能带 /）
# --------------------------------------------------------------------------- #
class CompanyEvent(BaseModel):
    event_type: Optional[str] = None          # 质押 / 股份回购 / 股东减持
    披露时间: Optional[str] = None
    事件时间: Optional[str] = None
    交易金额: Optional[str] = None
    每股交易价格: Optional[str] = None
    交易完成时间: Optional[str] = None
    # 质押专属
    质押方: Optional[str] = None
    质权方: Optional[str] = None
    质押物: Optional[str] = None
    质押物所属公司: Optional[str] = None
    质押股票_股份数量: Optional[str] = None
    质押物占总股比: Optional[str] = None
    质押物占持股比: Optional[str] = None
    # 回购专属
    回购方: Optional[str] = None
    回购股份数量: Optional[str] = None
    占公司总股本比例: Optional[str] = None
    回购完成时间: Optional[str] = None
    # 减持专属
    股票简称: Optional[str] = None
    减持方: Optional[str] = None
    交易股票_股份数量: Optional[str] = None
    减持部分占所持比例: Optional[str] = None
    减持部分占总股本比例: Optional[str] = None


class DuEEFinDoc(BaseModel):
    events: list[CompanyEvent] = []


# --------------------------------------------------------------------------- #
# CCKS 金融风控/反欺诈案例要素抽取 — 盗用风险/欺诈风险等，字段高度同质(11/13重叠)
# 用统一扁平 schema（同 CORD 套路，不像 DuEE-fin 按事件类型分字段）
# 取高频8要素 + event_type(level1/level2/level3拼接)；稀疏要素(<1%,如银行卡号)不纳入
# --------------------------------------------------------------------------- #
class FraudCase(BaseModel):
    event_type: Optional[str] = None       # level1/level2/level3 拼接
    资损金额: Optional[str] = None
    支付渠道: Optional[str] = None
    受害人: Optional[str] = None
    嫌疑人: Optional[str] = None
    案发时间: Optional[str] = None
    案发城市: Optional[str] = None
    涉案平台: Optional[str] = None
    受害人身份: Optional[str] = None


# --------------------------------------------------------------------------- #
# Registry: domain name -> (pydantic model, task description)
# --------------------------------------------------------------------------- #
SCHEMA_REGISTRY: dict[str, dict] = {
    "cord": {
        "model": Receipt,
        "task": "你是收据信息抽取器。从收据 OCR 文本中抽取结构化信息。",
        # 每字段的人类可读说明（rich prompt 用；JSON key 仍是 nm/cnt 等，与 gold 一致）
        "field_desc": (
            "字段说明：\n"
            "  menu: 菜品/商品行项列表，每项 {nm:品名, cnt:数量, price:该行金额}\n"
            "  sub_total: {subtotal_price:小计(税前合计), tax_price:税额, service_price:服务费}\n"
            "  total: {total_price:合计(应付总额), cashprice:实付现金, changeprice:找零}\n"
        ),
    },
    "duee_fin": {
        "model": DuEEFinDoc,
        "task": "你是金融公告事件抽取器。从上市公司公告文本中抽取金融事件及其要素。",
        "field_desc": (
            "只抽取以下三类事件（无相关事件则 events 填 []）：质押、股份回购、股东减持。\n"
            "events 是事件列表，每个事件 {event_type: 事件类型} + 该类要素字段（缺失填 null）：\n"
            "  通用: 披露时间, 事件时间, 交易金额, 每股交易价格, 交易完成时间\n"
            "  质押: 质押方, 质权方, 质押物, 质押物所属公司, 质押股票_股份数量, 质押物占总股比, 质押物占持股比\n"
            "  股份回购: 回购方, 回购股份数量, 占公司总股本比例, 回购完成时间\n"
            "  股东减持: 股票简称, 减持方, 交易股票_股份数量, 减持部分占所持比例, 减持部分占总股本比例\n"
        ),
    },
    "ccks_fraud": {
        "model": FraudCase,
        "task": "你是金融风控案例要素抽取器。从支付欺诈/账户盗用案例文本中抽取关键要素。",
        "field_desc": (
            "字段说明：\n"
            "  event_type: 案例类型(如 盗用风险/欺诈风险 及其细分)\n"
            "  资损金额: 案例中的资金损失金额\n"
            "  支付渠道: 涉及的支付方式/渠道\n"
            "  受害人: 受害人姓名\n"
            "  嫌疑人: 嫌疑人姓名/昵称\n"
            "  案发时间: 案件发生的时间\n"
            "  案发城市: 案件发生的城市\n"
            "  涉案平台: 涉及的平台/App名称\n"
            "  受害人身份: 受害人的身份信息(如学生/白领等)\n"
        ),
    },
}


def get_model(domain: str) -> Type[BaseModel]:
    return SCHEMA_REGISTRY[domain]["model"]


def compact_schema_str(model: Type[BaseModel]) -> str:
    """A short, human-readable field map for the system prompt.

    e.g. {menu:[{nm,cnt,price}], sub_total:{subtotal_price,tax_price,service_price}, total:{...}}
    """
    def render(m: Type[BaseModel]) -> str:
        parts = []
        for name, field in m.model_fields.items():
            ann = field.annotation
            inner = _inner_model(ann)
            if inner is not None and _is_list(ann):
                parts.append(f"{name}:[{{{render(inner)}}}]")
            elif inner is not None:
                parts.append(f"{name}:{{{render(inner)}}}")
            else:
                parts.append(name)
        return ",".join(parts)

    return "{" + render(model) + "}"


def _deref(node, defs):
    """Inline all $ref and drop $defs -> self-contained schema (Ollama/llama.cpp
    grammar backend does not support $ref)."""
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            return _deref(defs[name], defs)
        return {k: _deref(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_deref(x, defs) for x in node]
    return node


def flat_json_schema(model: Type[BaseModel]) -> dict:
    """Dereferenced JSON schema for constrained decoding (Ollama `format`)."""
    s = model.model_json_schema()
    return _deref(s, s.get("$defs", {}))


TYPE_NOTE = (
    "重要：所有字段值一律输出为 JSON 字符串（数字/数量/金额也要加引号，"
    '例如 price 输出 "11,000" 而不是 11000）。\n'
)


def build_system_prompt(domain: str, rich: bool = False, types: bool = False) -> str:
    """rich: 加每字段人类可读说明。types: 加"值一律输出字符串"的类型说明
    （用于公平测量 schema 合法率——之前 prompt 没说类型，对大模型不公平）。"""
    info = SCHEMA_REGISTRY[domain]
    schema = compact_schema_str(info["model"])
    desc = info.get("field_desc", "") if rich else ""
    type_note = TYPE_NOTE if types else ""
    return (
        f"{info['task']}\n"
        f"{desc}{type_note}"
        f"严格按以下 JSON Schema 输出，缺失字段填 null，列表无项时填 []，"
        f"只输出 JSON，不要任何多余文字或解释。\n"
        f"Schema: {schema}"
    )


# --------------------------------------------------------------------------- #
# helpers for introspecting Optional[...] / list[...] / nested BaseModel
# --------------------------------------------------------------------------- #
def _is_list(ann) -> bool:
    import typing
    origin = typing.get_origin(ann)
    return origin in (list, list)


def _inner_model(ann) -> Optional[Type[BaseModel]]:
    """Return the nested BaseModel inside Optional[X], list[X], X if any."""
    import typing

    def is_model(t) -> bool:
        return isinstance(t, type) and issubclass(t, BaseModel)

    if is_model(ann):
        return ann
    for arg in typing.get_args(ann):
        if is_model(arg):
            return arg
        sub = _inner_model(arg)
        if sub is not None:
            return sub
    return None


if __name__ == "__main__":
    # quick self-check
    print("system prompt for 'cord':\n")
    print(build_system_prompt("cord"))
    print("\nvalidate a sample:")
    sample = {
        "menu": [{"nm": "Ice Lemon Tea", "cnt": "1 x", "price": "24,000"}],
        "sub_total": {"subtotal_price": "331,000", "tax_price": "33,100"},
        "total": {"total_price": "364,100"},
    }
    print(json.dumps(Receipt.model_validate(sample).model_dump(), ensure_ascii=False))
