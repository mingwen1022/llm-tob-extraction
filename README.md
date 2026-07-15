# 面向本地私有化的 to-B 文档抽取 · 模型选型与评测研究 + Orchestrator 产品

"用大模型 API 还是本地微调小模型？" 的选型 + 评测研究，包装成多场景私有化抽取产品 Orchestrator。
面向 AI 产品经理岗（to-B / fintech）。**当前**：3 个独立域（CORD 英文小票 / DuEE-fin 中文金融公告 /
CCKS-fraud 中文反欺诈案例）`文本 → JSON`，共享一个基座 + 3 个 LoRA adapter，零样本路由自动分发。

- 定位与方案：[`doc/plans/项目定位与方案.md`](doc/plans/项目定位与方案.md)（WHY）· 执行计划：[`doc/plans/PLAN.md`](doc/plans/PLAN.md)（WHAT/下一步）
- 展示站点：打开 [`doc/site/index.html`](doc/site/index.html)（总览 / 实施路线 / 评测指标 / 评测结果 / Qwen3.5架构 / 产品架构 / 产品Demo）
- 结果：三域 E0→E2 micro-F1 提升稳定在 **+0.49~0.54**（CORD 0.409→0.945，DuEE-fin 0.322→0.861，CCKS-fraud 0.223→0.714，schema合法率均→100%），完整数字见 [`doc/site/results.html`](doc/site/results.html) / [`runs/`](runs/)
- 产品 Demo：`uv run python scripts/demo_app.py` 本地起一个可交互的多域抽取 Gradio 界面
- 基座锁定：**Qwen3.5-4B** · 训练：**HF PEFT/Unsloth** · 评测：**transformers+PEFT** · 服务：**PEFT多adapter（本地）/ vLLM多LoRA（生产设计）**

## 目录结构

```
doc/
  plans/           # 方案与计划（定位 / PLAN / CORD 实验计划）
  site/            # HTML 展示站点（index 为入口，可互相跳转）
shared/
  schema.py        # Pydantic 领域 schema + system prompt（SCHEMA_REGISTRY 注册新域，当前3个）
  normalize.py     # 字段归一化（金额/日期/空白/全半角）
  json_utils.py    # 从模型输出中鲁棒提取 JSON（剥 ```json``` 包裹）
  eval.py          # 评测器：flatten → 多重集 → P/R/F1（micro/macro/per-field）+ JSON 合法率
  router.py        # 零样本路由 prompt 构建 + 输出解析（新增域自动纳入，无需重训分类器）
  convert_cord.py / convert_duee_fin.py / convert_ccks.py  # 各域 原始标注 → 训练/评测 jsonl
  dedup_check.py   # train/test 数据泄漏核查（新域接入前必跑）
scripts/
  run_inference.py     # E0-E3 本地推理（transformers+peft，可选 Outlines 约束）
  run_ollama.py        # Ollama 基线推理（Mac 本地）
  run_api.py            # 前沿/国产大模型 API 对照（E4）
  run_orchestrator.py   # 端到端 Orchestrator：路由（复用基座零样本，disable_adapter）+ 多adapter抽取
  demo_app.py            # Gradio 产品 demo（本地跑 -> http://127.0.0.1:7860）
  train_cord.py / cord_train_colab.ipynb  # Colab/Unsloth 训练（各域复用同一脚本，产出 HF adapter）
tests/test_eval.py     # 评测器离线单测（无需模型/下载）
adapters/{cord,duee_fin,ccks_fraud}/  # 训练好的 3 个 LoRA adapter（共享同一基座）
data/{cord,duee_fin_cn,ccks_fraud_cn}/  # 转换后的数据（各域 train/val/test split，已去重复核查）
runs/              # 推理输出 + 结果（各域 baseline/results.md）
legacy/            # 早期草稿方案（已归档，不再维护）
```

## 快速开始（用 uv 管理环境）

依赖分组：核心（数据+评测，默认装）/ `infer`（本地推理+约束解码）/ `baseline`（GPT-4o）/ `demo`（Gradio）。

```bash
# 安装核心依赖（建 .venv）
uv sync

# 1) 离线自测（验证评测器逻辑，无需下载）
uv run python -m tests.test_eval
uv run python -m shared.convert_cord --sample

# 2) 生成 CORD 数据（需联网下载 naver-clova-ix/cord-v2）
uv run python -m shared.convert_cord --out data/cord

# 3) 本地推理/评测需要重包，按需加装：
uv sync --extra infer        # transformers + peft + torch + outlines
uv sync --extra baseline     # openai (E4)
uv sync --extra demo         # gradio

# 4) 基线 E0/E1（无需训练）
uv run python scripts/run_inference.py --base Qwen/Qwen3.5-4B \
    --eval-file data/cord/test.eval.jsonl --out runs/e0.jsonl
uv run python -m shared.eval --pred runs/e0.jsonl --gold data/cord/test.eval.jsonl --name "E0 基座·自由"

# 5) 训练（见 scripts/train_cord_colab.md，在 Colab 上）→ 下载到 adapters/cord/
# 6) E2/E3 评测（见 train_cord_colab.md 第 7 节）
```

## 实验矩阵

| 编号 | 配置 | 命令要点 |
|---|---|---|
| E0 | 基座·自由 | `--base` |
| E1 | 基座·约束 | `--base --constrained` |
| E2 | LoRA·自由 | `--base --adapter` |
| E3 | LoRA·约束 ⭐ | `--base --adapter --constrained` |
| E4 | GPT-4o | 单独脚本（待加） |

## 扩展到新领域（Phase 2）

1. 在 `shared/schema.py` 的 `SCHEMA_REGISTRY` 注册新域（pydantic 模型 + task 描述）
2. 写 `shared/convert_<域>.py`（标注 → `{system,user,gt}` / messages）
3. 复用 `run_inference.py` + `shared.eval`，训练同基座同 r=16 的新 adapter
4. Phase 3：vLLM 多 adapter 热插拔 + Orchestrator 路由
```
