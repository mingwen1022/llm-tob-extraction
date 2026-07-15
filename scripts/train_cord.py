"""CORD LoRA 训练脚本（在 Colab / GCP 上跑，CUDA）。

产出 HF PEFT 格式 adapter（可被 vLLM 多 LoRA 直接加载）。

Colab 用法：
  1) 上传 data/cord/train.train.jsonl 和 val.train.jsonl（或挂 Drive / 从 HF 拉）
  2) !pip install -q unsloth
  3) !python train_cord.py --train-file train.train.jsonl --val-file val.train.jsonl

关键点（针对 Qwen3.5）：
  - 用 Unsloth 统一的 FastModel 类（Qwen3.5 是多模态，FastLanguageModel 不适用）
  - Qwen3.5 不建议 4-bit QLoRA（量化误差偏大）→ 默认 bf16 LoRA
  - bf16 需要 Ampere+ GPU（L4/A100）。免费 T4 无原生 bf16，请加 --load-in-4bit 兜底
"""
from __future__ import annotations

import argparse
import json


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen3.5-4B")
    ap.add_argument("--train-file", default="train.train.jsonl")
    ap.add_argument("--val-file", default="val.train.jsonl")
    ap.add_argument("--out", default="cord_adapter", help="adapter 保存目录")
    # LoRA（与实验计划一致）
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    # 训练
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)  # 有效 batch = 16
    # 显存
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="低显存兜底（如免费 T4）。Qwen3.5 不推荐，会略降质量")
    # 可选：推送到 HF Hub
    ap.add_argument("--push-to-hub", default=None, help="如 'user/cord-qwen35-4b-lora'")
    return ap.parse_args()


def main():
    args = build_args()

    from unsloth import FastModel
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    # 1) 载入基座（Qwen3.5 → FastModel，默认 bf16 LoRA）
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    # 2) 套 LoRA（统一 r=16，目标全部 linear）
    model = FastModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3) 数据：messages → chat 文本
    def fmt(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    train_ds = load_dataset("json", data_files=args.train_file, split="train").map(fmt)
    eval_ds = None
    if args.val_file:
        try:
            eval_ds = load_dataset("json", data_files=args.val_file, split="train").map(fmt)
        except Exception:
            eval_ds = None

    print(f"train={len(train_ds)}  eval={len(eval_ds) if eval_ds else 0}")
    print("sample text head:\n", train_ds[0]["text"][:300])

    # 4) 训练
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            optim="adamw_8bit",
            max_seq_length=args.max_seq_len,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_ds else "no",
            output_dir="outputs",
            seed=42,
            report_to="none",
        ),
    )

    # 只在「助手回答(JSON)」上算 loss，屏蔽 system/user（更稳、更省）
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("[ok] train_on_responses_only enabled")
    except Exception as e:
        print(f"[warn] train_on_responses_only skipped: {e}")

    trainer.train()

    # 5) 保存 adapter（HF PEFT 格式，不 fuse）
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[done] adapter saved -> {args.out}/  (adapter_config.json + adapter_model.safetensors)")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f"[done] pushed -> {args.push_to_hub}")


if __name__ == "__main__":
    main()
