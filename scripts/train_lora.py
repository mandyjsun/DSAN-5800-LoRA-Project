import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig


def load_jsonl_dataset(path: str):
    return load_dataset("json", data_files=path, split="train")


def maybe_bnb_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if not args.use_qlora:
        return None
    nf4 = bool(args.bnb_4bit_nf4)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bool(args.bnb_double_quant),
        bnb_4bit_quant_type="nf4" if nf4 else "fp4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    target_modules = args.target_modules.split(",") if args.target_modules else [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def format_example(ex: Dict, tokenizer: AutoTokenizer) -> str:
    instruction = ex.get("instruction", "")
    user_input = ex.get("input", "")
    code = ex.get("output", "")
    messages = [
        {"role": "system", "content": "You are a Python coding assistant. Produce correct, clean, efficient Python."},
        {
            "role": "user",
            "content": instruction if not user_input else f"{instruction}\n\nInput: {user_input}",
        },
        {"role": "assistant", "content": code},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return f"System: You are a Python coding assistant.\nUser: {instruction}\nAssistant:\n{code}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA/QLoRA adapter with TRL SFTTrainer.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, default="")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--bnb-4bit-nf4", action="store_true")
    parser.add_argument("--bnb-double-quant", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--config", type=str, default="", help="Optional YAML config (overridden by CLI)")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and (getattr(args, k) in (None, "", 0, False)):
                setattr(args, k, v)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_cfg = maybe_bnb_config(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = build_lora_config(args)

    train_ds = load_jsonl_dataset(args.train_file)
    val_ds = load_jsonl_dataset(args.val_file)

    def formatting_func(example):
        # TRL expects a single formatted string or list of strings for each example
        return format_example(example, tokenizer)

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        bf16=args.bf16,
        fp16=args.fp16,
        max_length=args.max_seq_len,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_cfg,
        formatting_func=formatting_func,
        args=sft_cfg,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Save config used
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()


