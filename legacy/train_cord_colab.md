# CORD LoRA 训练（Colab / GCP）

> 训练在 **CUDA GPU** 上跑，产出 **HF PEFT 格式 adapter**（vLLM 多 LoRA 可直接加载）。
> Mac 不做训练，只做数据转换 + 评测 + demo。
> 实际训练逻辑见同目录 **`train_cord.py`**，本文档是运行步骤。

## ⚠️ 针对 Qwen3.5 的两个要点

1. **用 Unsloth 的 `FastModel`**（Qwen3.5 是多模态，旧的 `FastLanguageModel` 不适用）。
2. **不建议 4-bit QLoRA**（Qwen3.5 量化误差偏大）→ 默认 **bf16 LoRA**。
   - bf16 需要 **Ampere+ GPU（L4 / A100）**。
   - 免费 **T4 无原生 bf16**：要么选 L4/A100 运行时（Colab Pro / GCP），要么加 `--load-in-4bit` 兜底（质量略降）。

## 0. 本地先生成数据（Mac）

```bash
uv run python -m shared.convert_cord --out data/cord
# 需要上传到 Colab 的文件：
#   data/cord/train.train.jsonl   (800 条)
#   data/cord/val.train.jsonl     (100 条)
```

## 1. Colab：选运行时 + 装依赖 + 传数据

- 运行时：**优先选 L4 或 A100**（支持 bf16）。只有 T4 时用 `--load-in-4bit`。
- 上传 `train_cord.py` + `train.train.jsonl` + `val.train.jsonl`（或挂 Google Drive / 从 HF 拉）。

```python
!pip install -q unsloth
```

## 2. 训练（一行命令，超参已对齐实验计划）

```bash
# L4 / A100（推荐，bf16）
!python train_cord.py --train-file train.train.jsonl --val-file val.train.jsonl --out cord_adapter

# 只有免费 T4 时（4bit 兜底）
!python train_cord.py --train-file train.train.jsonl --val-file val.train.jsonl --out cord_adapter --load-in-4bit
```

可选：直接推到 HF Hub（省得手动下载）

```bash
!huggingface-cli login
!python train_cord.py ... --push-to-hub <user>/cord-qwen35-4b-lora
```

## 3. 取回 adapter 到本地

```
cord_adapter/
  adapter_config.json
  adapter_model.safetensors   # 几十 MB
```
下载到本地 `adapters/cord/`，或从 HF Hub 拉。

## 4. smoke test：vLLM 多 LoRA 加载验证（GCP CUDA，开工前必做）

```bash
pip install -U vllm
vllm serve Qwen/Qwen3.5-4B --enable-lora \
    --lora-modules cord=/path/to/cord_adapter --max-lora-rank 16
# POST /v1/chat/completions，model="cord"，确认能出 JSON
```

## 5. 回本地评测（Mac）

```bash
# E2（自由解码，本地 Ollama 跑基座+adapter 不便；用 transformers+peft）
uv sync --extra infer
uv run python scripts/run_inference.py --base Qwen/Qwen3.5-4B --adapter adapters/cord \
    --eval-file data/cord/test.eval.jsonl --out runs/e2.jsonl
uv run python -m shared.eval --pred runs/e2.jsonl --gold data/cord/test.eval.jsonl --name "E2 LoRA·自由"

# E3（约束解码）建议在 vLLM 侧用 guided_json 跑（Ollama 对 qwen3.5 的 schema 约束不可靠）
```

## 超参对照表（train_cord.py 默认值）

| 超参 | 值 | flag |
|---|---|---|
| 基座 | unsloth/Qwen3.5-4B | `--base` |
| 精度 | bf16 LoRA（4bit 可选） | `--load-in-4bit` |
| r / alpha / dropout | 16 / 32 / 0.05 | `--rank --alpha --dropout` |
| epochs | 3 | `--epochs` |
| lr / scheduler / warmup | 2e-4 / cosine / 5% | `--lr` |
| batch × grad_accum | 2 × 8（有效 16） | `--batch --grad-accum` |
| max_seq_len | 2048 | `--max-seq-len` |
| loss | 仅在 assistant 回答上 | （自动 train_on_responses_only） |
