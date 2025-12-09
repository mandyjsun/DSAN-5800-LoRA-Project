import argparse
import ast
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re


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
    messages = [
        {"role": "system", "content": "You are a Python coding assistant. Produce correct, clean, efficient Python."},
        {"role": "user", "content": instruction if not user_input else f"{instruction}\n\nInput: {user_input}"},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: You are a Python coding assistant.\nUser: {instruction}\nAssistant:"


def extract_python_code(text: str) -> str:
    # If chat template artifacts are present (e.g., [INST] ... [/INST]), take the tail after the last close tag
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]
    # Extract first fenced code block if any
    if "```" in text:
        parts = text.split("```")
        # parts: [pre, maybe 'python\ncode', post...]
        if len(parts) >= 3:
            block = parts[1]
            # Drop optional language tag
            if block.lstrip().lower().startswith("python"):
                block = block.split("\n", 1)[1] if "\n" in block else ""
            return block.strip()
    # Otherwise, return raw text stripped
    return text.strip()


def load_model(
    base_model: str,
    lora_dir: Optional[str],
    device: str,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    offload_folder: Optional[str] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
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
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        offload_folder=offload_folder,
    )
    if lora_dir:
        model = PeftModel.from_pretrained(
            model,
            lora_dir,
            device_map=device_map,
            offload_folder=offload_folder,
        )
    model.eval()
    return model, tokenizer


def safe_syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def run_tests_on_code(code: str, tests: List[str]) -> Tuple[bool, List[str]]:
    # Execute code and then run each test assert string
    # WARNING: This uses exec; run only in trusted environments.
    g: Dict = {}
    try:
        exec(code, g, g)  # noqa: S102
    except Exception as e:
        return False, [f"Execution error: {type(e).__name__}: {e}"]
    errors: List[str] = []
    for t in tests:
        try:
            exec(t, g, g)  # noqa: S102
        except Exception as e:
            errors.append(f"Test failed: {t} -> {type(e).__name__}: {e}")
    return (len(errors) == 0), errors


def evaluate_mbpp(
    base_model: str,
    mbpp_file: str,
    out_file: str,
    lora_dir: Optional[str] = None,
    max_new_tokens: int = 768,
    temperature: float = 0.2,
    top_p: float = 0.9,
    device: str = "cuda",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    offload_folder: Optional[str] = None,
    limit: int = 0,
) -> None:
    model, tokenizer = load_model(
        base_model,
        lora_dir,
        device,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device_map=device_map,
        offload_folder=offload_folder,
    )
    tasks = load_jsonl(mbpp_file)
    total = len(tasks)
    if limit and limit > 0:
        tasks = tasks[:limit]
        total = len(tasks)
    results: List[Dict] = []
    num_syntax_ok = 0
    num_pass = 0

    for idx, ex in enumerate(tasks, start=1):
        instruction = ex.get("instruction", "")
        tests = ex.get("tests") or []
        prompt = build_prompt(tokenizer, instruction)
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
        raw_completion = text[len(prompt) :].strip() if text.startswith(prompt) else text
        completion = extract_python_code(raw_completion)
        syntax_ok = safe_syntax_ok(completion)
        if syntax_ok:
            num_syntax_ok += 1
        passed = False
        test_errors: List[str] = []
        if tests:
            passed, test_errors = run_tests_on_code(completion, tests)
            if passed:
                num_pass += 1
        results.append(
            {
                "task_id": ex.get("task_id"),
                "instruction": instruction,
                "generated": completion,
                "syntax_ok": syntax_ok,
                "passed": passed,
                "errors": test_errors,
            }
        )
        if idx % 10 == 0 or idx == total:
            print(f"[eval] Processed {idx}/{total}")

    summary = {
        "total": len(tasks),
        "syntax_ok": num_syntax_ok,
        "syntax_rate": num_syntax_ok / max(1, len(tasks)),
        "pass": num_pass,
        "pass_rate": num_pass / max(1, len(tasks)),
        "model": base_model,
        "lora_dir": lora_dir,
    }
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote evaluation results to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate (MBPP) functional correctness and syntax.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora-dir", type=str, default="")
    parser.add_argument("--mbpp-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit loading (requires bitsandbytes)")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map for model dispatch (e.g., auto, cuda:0)")
    parser.add_argument("--offload-folder", type=str, default="", help="Folder for CPU offload when using device_map=auto")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks for a quick sanity run")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_mbpp(
        base_model=args.model,
        lora_dir=args.lora_dir or None,
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


