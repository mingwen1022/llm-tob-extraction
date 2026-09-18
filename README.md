# Multi-Domain Structured Extraction Orchestrator

### Local fine-tuned 4B vs. frontier APIs — turning a procurement decision into a measurable experiment

**English** · [简体中文](README.zh.md)

Every to-B document extraction project hits the same fork in the road: customer data cannot leave the network, so should you buy a frontier LLM API or fine-tune a small model in-house? That call usually gets made on intuition. This project makes it measurable — three domains that differ in language, document type and task structure, run through one 4B base model, one LoRA recipe and one evaluator, then packaged behind a service that exposes a single entry point.

**Full experiment design, three-domain results and the selection conclusion**: <https://mingwen.net/projects/extract-orchestrator.html>

> Every number, its measurement convention and the known limitations live on the detail page. This README does not restate them — **two copies of a number will eventually disagree**. What follows is how to run it and what the code looks like.

| Receipt (routed to `cord`) | Fraud case (routed to `ccks_fraud`) |
| --- | --- |
| ![receipt](docs/screenshots/demo-receipt.png) | ![fraud](docs/screenshots/demo-fraud.png) |

One entry point, one 4B base. The caller never says what kind of document this is: the system classifies it zero-shot, switches to the matching adapter, extracts under that domain's schema, and reports which path it took plus per-stage latency. The 10–25s in the screenshots is an unoptimised Mac dev setup; a vLLM deployment should be an order of magnitude faster.

## The conclusion, in one paragraph

After fine-tuning, the three domains reach micro-F1 **0.945 / 0.861 / 0.714** with schema validity at 100% across the board (against a same-convention full-prompt baseline: CORD +0.298, DuEE-fin +0.539, CCKS-fraud +0.491 — **not the uniform lift an earlier draft claimed. Fine-tuning gains most where the prompt cannot reach**).

But **the zero-shot comparison was not a fair one**, and that turned out to be true twice over. Give the APIs in-context examples and the local 4B's lead over the best API collapses from 8.5 points to **1.8** — while the APIs' own rerun-to-rerun variance is ±1.4. Then give the *base* model the same examples, and the 20-point "parameter count gap" shrinks to **5–6 points**.

What survives all of that is the ablation on a single model: same prompt, same examples, the only variable being where the examples go. **Zero-shot 0.647 → 32 examples in context 0.876 → examples trained into the weights 0.945**, with the share of documents extracted perfectly going 7% → 36% → 68% — and LoRA inference carries the same context length as zero-shot. **Accuracy is no longer the reason to pick one over the other.** What remains is data residency, cost structure and output determinism. Full write-up in [`runs/model_selection_report.md`](runs/model_selection_report.md) §4.1 and §7.

## Current state

| Area | Status |
| --- | --- |
| Three-domain conversion + leakage checks | ✅ CORD / DuEE-fin / CCKS-fraud; `dedup_check.py` is mandatory before onboarding a domain |
| Experiment matrix E0–E4 | ✅ base free / base constrained / LoRA free / LoRA constrained / six frontier and Chinese APIs |
| Few-shot gradient | ✅ 0/4/8/16/32/64 (full ladder for Gemini and MiniMax, 0 and 16 for the rest) |
| Orchestrator | ✅ zero-shot routing + three hot-swapped adapters, 98.9% routing accuracy across domains |
| Out-of-domain fallback | 🔶 OOD behaviour and fill-rate thresholds measured; the fallback itself is not implemented (detail page §7.1) |
| Production deployment | ❌ local transformers+PEFT dev setup; vLLM multi-LoRA exists as a design only |

- Positioning and approach: [`doc/plans/项目定位与方案.md`](doc/plans/项目定位与方案.md) (WHY) · Execution plan: [`doc/plans/PLAN.md`](doc/plans/PLAN.md) (WHAT / next)
- Raw results: [`runs/`](runs/) (per-domain `*_results.md`, `e4_fewshot_summary.md`, `ood_probe_summary.md`)
- Product demo: `uv run python scripts/demo_app.py` starts an interactive multi-domain Gradio UI locally
- Base model locked to **Qwen3.5-4B** · training **HF PEFT / Unsloth** · evaluation **transformers + PEFT** · serving **PEFT multi-adapter (local) / vLLM multi-LoRA (designed)**
- All datasets are public academic corpora (CORD / DuEE-fin / CCKS2021). No real customer documents.

## Layout

```
doc/
  plans/           # approach and plans (positioning / PLAN / CORD experiment plan)
  site/            # HTML write-up (index is the entry point, pages cross-link)
shared/
  schema.py        # Pydantic domain schemas + system prompts (register a domain in SCHEMA_REGISTRY; 3 today)
  normalize.py     # field normalisation (amounts / dates / whitespace / full-width)
  json_utils.py    # robust JSON extraction from model output (strips ```json fences)
  eval.py          # evaluator: flatten → multiset → P/R/F1 (micro/macro/per-field) + JSON validity
  router.py        # zero-shot routing prompt + output parsing (new domains are picked up automatically —
                   # no classifier to retrain)
  convert_cord.py / convert_duee_fin.py / convert_ccks.py  # per-domain raw annotations → train/eval jsonl
  dedup_check.py   # train/test leakage check (mandatory before onboarding a domain)
scripts/
  run_inference.py      # E0–E3 local inference (transformers+peft, optional Outlines constraints)
  run_ollama.py         # Ollama baseline (local Mac)
  run_api.py            # frontier / Chinese API comparison (E4)
  run_orchestrator.py   # end-to-end: routing (base model via disable_adapter) + multi-adapter extraction
  eval_router.py        # routing accuracy + confusion matrix (--backend peft mirrors the production path)
  demo_app.py           # Gradio demo (local -> http://127.0.0.1:7860)
  build_eval_site.py    # generates doc/eval/ — per-round detail pages with per-document TP/FP/FN
  train_cord.py / cord_train_colab.ipynb  # Colab/Unsloth training (one script for all domains → HF adapter)
tests/test_eval.py      # offline evaluator unit tests (no model, no downloads)
adapters/{cord,duee_fin,ccks_fraud}/    # three trained LoRA adapters, sharing one base
data/{cord,duee_fin_cn,ccks_fraud_cn}/  # converted data (train/val/test per domain, leakage-checked)
runs/              # inference outputs + results (per-domain baseline/results.md)
legacy/            # early drafts, archived and unmaintained
```

## Quick start (uv-managed environment)

Dependency groups: core (data + evaluation, installed by default) / `infer` (local inference + constrained decoding) / `baseline` (API comparison) / `demo` (Gradio).

```bash
# core dependencies (creates .venv)
uv sync

# 1) offline self-test (validates evaluator logic, no downloads)
uv run python -m tests.test_eval
uv run python -m shared.convert_cord --sample

# 2) build the CORD dataset (downloads naver-clova-ix/cord-v2)
uv run python -m shared.convert_cord --out data/cord

# 3) local inference/evaluation needs heavier packages — install as needed:
uv sync --extra infer        # transformers + peft + torch + outlines
uv sync --extra baseline     # API comparison (E4)
uv sync --extra demo         # gradio

# 4) baselines E0/E1 (no training required)
uv run python scripts/run_inference.py --base Qwen/Qwen3.5-4B \
    --eval-file data/cord/test.eval.jsonl --out runs/e0.jsonl
uv run python -m shared.eval --pred runs/e0.jsonl --gold data/cord/test.eval.jsonl --name "E0 base/free"

# 5) training (see scripts/train_cord_colab.md, runs on Colab) → download into adapters/cord/
# 6) E2/E3 evaluation (see train_cord_colab.md §7)
```

## Experiment matrix

| ID | Configuration | Command |
|---|---|---|
| E0 | base, free decoding | `--base` |
| E1 | base, constrained | `--base --constrained` |
| E2 | LoRA, free decoding | `--base --adapter` |
| E3 | LoRA, constrained ⭐ | `--base --adapter --constrained` |
| E4 | frontier / Chinese API comparison | `scripts/run_api.py` (Gemini / Qwen / Claude / DeepSeek / Kimi / GLM / MiniMax, incl. the few-shot ladder) |

## Adding a domain

1. Register it in `SCHEMA_REGISTRY` in `shared/schema.py` (Pydantic model + task description)
2. Write `shared/convert_<domain>.py` (annotations → `{system,user,gt}` / messages)
3. Reuse `run_inference.py` + `shared.eval`; train a new adapter on the same base at the same r=16
4. The router needs no changes — a new entry in the registry is picked up automatically and nothing gets retrained. Hot-swapping adapters under vLLM in production remains a design, not a measured result.
