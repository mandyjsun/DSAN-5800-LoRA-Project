import argparse
import json
import os
from typing import Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


def load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_prompt(tokenizer: AutoTokenizer, instruction: str, user_input: str = "") -> str:
    # Prefer model's chat template if available
    messages = [
        {"role": "system", "content": "You are a Python coding assistant. Write correct, clean, and efficient Python code. Output only valid Python code. Please incorporate comments in your code where appropriate."},
        {"role": "user", "content": instruction if not user_input else f"{instruction}\n\nInput: {user_input}"},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback plain prompt
        return f"System: You are a Python coding assistant.\nUser: {instruction}\nAssistant:"


def generate_codes(
    model_name: str,
    mbpp_file: str,
    out_file: str,
    max_new_tokens: int = 768,
    temperature: float = 0.2,
    top_p: float = 0.9,
    device: str = "cuda",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    offload_folder: str | None = None,
    limit: int = 0,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        offload_folder=offload_folder,
    )
    model.eval()

    tasks = load_jsonl(mbpp_file)
    outputs: List[Dict] = []

    total = len(tasks)
    if limit and limit > 0:
        tasks = tasks[:limit]
        total = len(tasks)

    for idx, ex in enumerate(tasks, start=1):
        instruction = ex.get("instruction", "")
        user_input = ex.get("input", "")
        prompt = build_prompt(tokenizer, instruction, user_input)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        # Attempt to keep only the assistant completion after the prompt
        completion = text[len(prompt) :].strip() if text.startswith(prompt) else text

        outputs.append(
            {
                "task_id": ex.get("task_id"),
                "instruction": instruction,
                "generated": completion,
            }
        )

        # Progress print
        if idx % 10 == 0 or idx == total:
            print(f"[baseline] Processed {idx}/{total}")

    write_jsonl(out_file, outputs)
    print(f"Wrote generations: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot baseline on MBPP.")
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--mbpp-file", type=str, required=True, help="Processed MBPP JSONL")
    parser.add_argument("--out-file", type=str, required=True, help="Where to write generations JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit loading (requires bitsandbytes)")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map for model dispatch (e.g., auto, cuda:0)")
    parser.add_argument("--offload-folder", type=str, default="", help="Folder for offloading when using accelerate")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of MBPP tasks for a quick sanity run")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_codes(
        model_name=args.model,
        mbpp_file=args.mbpp_file,
        out_file=args.out_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        load_in_4bit=not args.no_4bit and not args.load_in_8bit,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        offload_folder=(args.offload_folder or None),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()


