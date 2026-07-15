# CORD LoRA 实验计划（Phase 1）

> 多场景 toB 结构化抽取 Agent · 第一个垂直切片
> 任务：小票 OCR 文本 → 结构化 JSON
> 基座：**Qwen3.5-4B（锁定）** · 训练：**HF PEFT / Unsloth** · 评测：**transformers+PEFT** · 服务：**vLLM**

---

## 0. 实验目标与成功标准

| 项 | 内容 |
|---|---|
| **核心目标** | 跑通「Qwen3.5-4B + LoRA + 约束解码」在 CORD 上 `OCR文本→JSON` 的完整链路，产出三方对比 |
| **战略目标** | 沉淀一套**领域无关**的流水线（数据转换 / 训练 / 约束 / 评测），Phase 2 换 schema 即可复用 |
| **成功标准** | ① 微调后 micro-F1 比基座 **+15 个点以上**<br>② 约束解码下 JSON 合法率 **≥ 99%**<br>③ 逼近 GPT-4o（差距 ≤ 5 点）<br>④ 流水线可泛化（换数据集只需改 schema + 转换脚本） |

---

## 1. 工具链与分工（先记牢，避免走错路）

```
[训练]                          [评测]                      [demo]            [生产多LoRA服务]
Unsloth / HF PEFT  ──adapter──► transformers+PEFT(+Outlines) ── Ollama ──── vLLM
 (Colab/GCP, CUDA)              (Mac M1 Max, MPS)            (Mac)         (GCP, CUDA)
 产出 HF 格式 adapter            直接吃 adapter, 出 F1         fuse+GGUF      adapter 即插即用
```

| 环节 | 工具 | 机器 | 备注 |
|---|---|---|---|
| 数据转换 | Python | Mac | CORD → 训练 jsonl |
| 微调 | Unsloth（QLoRA 4bit） | Colab 免费 T4 | 出 HF PEFT adapter |
| 评测 | transformers + peft + outlines | Mac (MPS) | 核心产出（F1 数字） |
| demo | Gradio + Ollama | Mac | adapter 需 fuse + 转 GGUF |
| 多 LoRA 服务 | vLLM（`--enable-lora`） | GCP CUDA | Phase 3 起 |

> ⚠️ **vLLM / Ollama 是推理引擎，跟微调无关。** 微调（Unsloth）和推理分离；adapter 给 vLLM 即插即用，给 Ollama 需先 fuse + 转 GGUF。

---

## 2. 数据准备

**数据集**：`naver-clova-ix/cord-v2`（HuggingFace，800 train / 100 val / 100 test）

每条样本 `ground_truth`（JSON 字符串）里有两样我们要的：
- `gt_parse` → **GOLD JSON**（目标）
- `valid_line[*].words`（text + 四点坐标）→ 重建 **INPUT 文本**

### 转换流程（`shared/convert_cord.py`）

```
ground_truth
 ├── valid_line[].words ── 按 y 分行、行内按 x 排序 ── 拼成 receipt_text   (INPUT)
 └── gt_parse ──────────── 直接作为 target JSON                          (GOLD)
        │
        ▼
组装 {system(任务+schema), user(receipt_text), assistant(gt_parse字符串)}
        │
        ▼
data/{train,val,test}.jsonl   ← test 抽 30~50 条人工抽查
```

> CORD 是图片数据集，但 OCR 已给好（`valid_line`），本阶段**不需要自己跑 OCR**，只需把词按阅读顺序拼成文本。真实图像 OCR 留到 Phase 5。

---

## 3. 四种数据形态（同一张小票串起来）

### 3.1 INPUT（OCR 文本，喂给模型的 user）
```text
RUMAH MAKAN BALI
Jl. Raya Ubud No.8
--------------------------------
Nasi Campur Bali     1 x   75,000
Bbk Bengil Nasi      1 x  125,000
Ice Lemon Tea        1 x   24,000
--------------------------------
Subtotal                  331,000
Tax 10%                    33,100
TOTAL                     364,100
```

### 3.2 GOLD / true target（gt_parse，训练的 assistant + 评测金标准）
```json
{
  "menu": [
    {"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"},
    {"nm": "Bbk Bengil Nasi",  "cnt": "1 x", "price": "125,000"},
    {"nm": "Ice Lemon Tea",    "cnt": "1 x", "price": "24,000"}
  ],
  "sub_total": {"subtotal_price": "331,000", "tax_price": "33,100"},
  "total":     {"total_price": "364,100"}
}
```

### 3.3 训练样本（一行 jsonl = 一张小票）
```json
{"messages": [
  {"role": "system", "content": "你是收据信息抽取器。严格按给定 JSON Schema 输出，缺失字段填 null，只输出 JSON。\nSchema: {menu:[{nm,cnt,price}], sub_total:{subtotal_price,tax_price}, total:{total_price}}"},
  {"role": "user", "content": "<receipt_text>"},
  {"role": "assistant", "content": "<gt_parse 的 JSON 字符串>"}
]}
```

### 3.4 OUTPUT（推理时只给 system+user，模型生成 assistant）
模型原始输出（可能带 ```json``` 包裹）→ `extract_json_block` + `json.loads` → PRED JSON。

---

## 4. Schema（`shared/schema.py`，CORD 高频子集，先小后大）

```python
from pydantic import BaseModel

class MenuItem(BaseModel):
    nm: str | None = None
    cnt: str | None = None
    price: str | None = None

class SubTotal(BaseModel):
    subtotal_price: str | None = None
    tax_price: str | None = None
    service_price: str | None = None

class Total(BaseModel):
    total_price: str | None = None
    cashprice: str | None = None
    changeprice: str | None = None

class Receipt(BaseModel):
    menu: list[MenuItem] = []
    sub_total: SubTotal | None = None
    total: Total | None = None
```

> 这个 Schema 一处定义、三处复用：① system prompt 告诉模型格式 ② Outlines 约束解码 ③ eval 的 schema 校验。先用这 9~10 个核心字段跑通，再扩 CORD 全字段。

---

## 5. 训练配置（Colab 免费 T4 / Unsloth，出 HF PEFT adapter）

| 超参 | 值 | 说明 |
|---|---|---|
| 基座 | **Qwen3.5-4B**（锁定） | 全程不换 |
| 方法 | QLoRA（4bit） | T4 16G 够 |
| LoRA rank / alpha | **r=16 / α=32** | 全域统一 |
| target modules | 全部 linear（q,k,v,o,gate,up,down） | |
| dropout | 0.05 | |
| epochs | 3 | 800 条小数据 |
| lr / scheduler | 2e-4 / cosine，warmup 5% | |
| batch / grad accum | 2 × 8（有效 16） | |
| max_seq_len | 2048 | 小票短，够用 |
| optimizer | adamw_8bit | 省显存 |

**产出**：`adapters/cord/`（HF PEFT 格式，vLLM 可直接加载）。

> 📌 **开工前 smoke test（必做）**：先用 30~50 条数据训一个 tiny LoRA → 用**最新版 vLLM** 加载验证多 LoRA 能跑通（Qwen3.5-4B 是 dense，不踩 MoE 的 fused-expert 兼容坑）。确认后再批量训。vLLM / Unsloth 钉最新版。

---

## 6. 实验矩阵（每行产出一组数字）

### 主实验
| 编号 | 配置 | 目的 |
|---|---|---|
| **E0** | Qwen3.5-4B 基座 · 自由解码 | 对照基线（下限） |
| **E1** | Qwen3.5-4B 基座 · +约束解码 | 看约束对"没微调"的提升 |
| **E2** | Qwen3.5-4B +LoRA · 自由解码 | 看微调单独贡献 |
| **E3** ⭐ | Qwen3.5-4B +LoRA · +约束解码 | **主结果** |
| **E4** | GPT-4o zero-shot | 上界参考 |

### 消融（时间够再做，简历更扎实）
| 编号 | 变量 | 看什么 |
|---|---|---|
| A1 | 数据量 200 / 400 / 800 | 数据效率曲线 |
| A2 | LoRA rank 8 / 16 / 32 | rank 敏感性 |
| A3 | OCR 文本是否按阅读顺序排序 | 输入质量影响 |
| A4 | vs Donut（CORD 官方 baseline） | 端到端 vs 你的方案 |

---

## 7. 评测逻辑（`shared/eval.py`，领域无关）

### 三层指标（必须分开报）
```
模型输出(字符串)
   │ ① 解析 + schema 校验      →  指标A: JSON 合法率（解析级 + schema级，约束前后分开）
   ▼
预测 JSON
   │ ② 拍平叶子字段 + 归一化
   ▼
{字段路径: 归一化值} 多重集
   │ ③ 与 gold 同样处理后比对
   ▼
TP/FP/FN  →  指标B: 字段级 P/R/F1（micro / macro / per-field）
```

### 字段级 F1（CORD/Donut 法：(字段,值) 多重集匹配）
- **拍平**：嵌套 JSON → `(叶子字段, 值)` 列表；**列表用字段名当 key 不带下标**（顺序无关，靠多重集匹配）。
- **归一化**（命门）：金额去千分位/货币符号（`75,000`→`75000`）、日期统一 ISO（`14/03/2015`→`2015-03-14`）、去空白、大小写、全半角；`None` 叶子跳过。
- **计数**：`TP=多重集交集`，`FP=pred多出的`，`FN=gold有但没命中的`。**预测错的值同时计 FP+FN。**
- `Precision=TP/(TP+FP)`，`Recall=TP/(TP+FN)`，`F1=2PR/(P+R)`。
- 聚合：micro（整体）+ macro（按字段平均）+ per-field（定位最差字段）。

### 关键代码骨架
```python
from collections import Counter

def flatten(obj, prefix=""):
    pairs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            pairs += flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for v in obj:                       # 列表不带下标
            pairs += flatten(v, prefix)
    elif obj is not None:
        pairs.append((prefix, normalize(prefix, obj)))
    return pairs

def prf(pred_json, gold_json):
    pred, gold = Counter(flatten(pred_json)), Counter(flatten(gold_json))
    tp = sum((pred & gold).values())
    fp = sum((pred - gold).values())
    fn = sum((gold - pred).values())
    return tp, fp, fn
```

### 预期结果形态（示意）
| 实验 | JSON合法率 | micro-F1 | macro-F1 | 最差字段 |
|---|---|---|---|---|
| E0 基座·自由 | ~85% | ~0.70 | ~0.64 | menu.cnt |
| E1 基座·约束 | ~99% | ~0.73 | ~0.67 | menu.sub |
| E2 LoRA·自由 | ~94% | ~0.90 | ~0.87 | menu.cnt |
| **E3 LoRA·约束** | **~99.5%** | **~0.92** | **~0.89** | menu.sub |
| E4 GPT-4o | ~98% | ~0.94 | ~0.92 | – |

---

## 8. 执行步骤（按顺序）

| # | 步骤 | 机器 | 预计 |
|---|---|---|---|
| 1 | 环境 & repo 骨架（`shared/` + `adapters/`） | Mac | 0.5 天 |
| 2 | `convert_cord.py`：下载+转换+重建文本+人工抽查 | Mac | 1 天 |
| 3 | `schema.py` + `eval.py`，用 10 条假数据自测 | Mac | 0.5 天 |
| 4 | **smoke test**：tiny LoRA → vLLM 多 LoRA 加载验证 | Colab+GCP | 0.5 天 |
| 5 | E0 / E1 基线（基座，无需训练） | Mac | 0.5 天 |
| 6 | Colab 训 LoRA → `adapters/cord/`（E2/E3） | Colab | 0.5 天 |
| 7 | 接 Outlines 约束解码，跑 E3 | Mac | 0.5 天 |
| 8 | E4 GPT-4o + 汇总对比表 | Mac | 0.5 天 |
| 9 | Gradio demo + README + 简历文案 | Mac | 0.5 天 |
| — | 消融 A1–A4（视时间补） | — | — |

**总计 ≈ 4~5 天**，结束即有一个可展示的完整子项目。

---

## 9. 产出物

- [ ] `01-doc-extraction/` 或 `adapters/cord/`（HF PEFT adapter）
- [ ] 5 行实验对比表 + per-field 误差分析 +（可选）数据效率曲线
- [ ] Gradio demo：贴小票文本 → 出 JSON + 字段高亮
- [ ] HF 上传 adapter + model card
- [ ] README + 一句简历 bullet（带量化指标）
- [ ] 可复用流水线（Phase 2 换 schema 直接套）

**简历文案模板：**
> 基于 **Qwen3.5-4B + LoRA** 微调收据/合同抽取模型，结合 JSON-Schema 约束解码，字段级 micro-F1 由基座 **71% 提升至 92%**，JSON 合法率 **99.5%**；以 4B 小模型逼近 GPT-4o，单条推理成本 **↓ ~95%** 且支持私有化部署。完成数据合成、训练、评测、量化部署与 Demo 全流程。

---

## 10. 避坑清单

- [ ] test 集务必**人工抽查**几十条，别全信数据集标注
- [ ] 评测前确认 `normalize` 处理了金额逗号 / 日期 / 空白，否则 F1 虚低
- [ ] menu 是列表，eval 用**多重集、不带下标**
- [ ] JSON 合法率**约束前后分开报**，才体现约束解码价值
- [ ] 解析容错：先 `extract_json_block` 剥掉 ```json``` 包裹再 parse
- [ ] 预测错的值要同时计 FP+FN（别只罚一次，否则 precision 虚高）
- [ ] E0~E2 一定要做，没有对照组的 F1 在简历上没说服力
- [ ] Colab 会断连，训练中途存 checkpoint
- [ ] 评测用 transformers+PEFT，**别用 Ollama**（不好接约束解码/adapter）

---

## 11. 与后续 Phase 的衔接

- **Phase 2**：复用本流水线，只写各域「标注→JSON」转换 + 新 schema，训 NDA / 发票 / CCKS 金融 adapter（同基座 Qwen3.5-4B、同 r=16）。
- **Phase 3**：vLLM 加载 1 基座 + N adapter，加 Orchestrator 路由。
- **Phase 5**：Qwen3.5-4B 原生多模态，可复用同基座直接做 `图片→JSON`（路线 B），或前置 PaddleOCR（路线 A）。

---

*基座/格式锁定：Qwen3.5-4B + HF PEFT，中途不换。*
