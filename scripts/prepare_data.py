import argparse
import json
import os
import random
import re
import textwrap
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


def read_or_load_codealpaca(path_hint: str | None) -> List[Dict]:
    if path_hint and os.path.exists(path_hint):
        ds = load_dataset("json", data_files={"train": os.path.join(path_hint, "**/*.json")}, split="train")
    else:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    records = []
    for ex in ds:
        # CodeAlpaca entries vary: keys: 'instruction','input','output'
        inst = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if not inst or not out:
            continue
        records.append({"instruction": inst, "input": inp, "output": out})
    return records


def read_or_load_mbpp(path_hint: str | None) -> List[Dict]:
    # Prefer HF dataset to capture tests
    try:
        ds = load_dataset("mbpp", split="test")
    except Exception:
        # Fallback to jsonl in provided directory
        if not path_hint:
            raise
        ds = load_dataset("json", data_files=os.path.join(path_hint, "**/*.jsonl"), split="train")
    records = []
    # HF `mbpp` has fields like: task_id, text (instruction), code (reference), test_list (list of tests)
    for ex in ds:
        instruction = (ex.get("text") or ex.get("prompt") or "").strip()
        tests = ex.get("test_list") or ex.get("test_list_str") or ex.get("test_list_simple")
        if isinstance(tests, str):
            # Some variants provide a stringified list
            try:
                tests = json.loads(tests)
            except Exception:
                tests = [tests]
        if not isinstance(tests, list):
            tests = []
        # We do not include ground-truth solutions in target to avoid leakage
        records.append(
            {
                "instruction": instruction,
                "input": "",
                "output": "",
                "tests": tests,
                "task_id": ex.get("task_id"),
            }
        )
    return records


_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*|\s*```\s*$", re.IGNORECASE)


def strip_code_fences(code: str) -> str:
    lines = code.strip().splitlines()
    if not lines:
        return code.strip()
    # Remove leading/trailing triple backtick blocks
    if _FENCE_RE.match(lines[0]):
        lines = lines[1:]
    if lines and _FENCE_RE.match(lines[-1]):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_indentation(code: str) -> str:
    # Convert tabs to 4 spaces, dedent common margin
    code = code.replace("\t", "    ")
    code = textwrap.dedent(code)
    # Avoid trailing spaces
    code = "\n".join(line.rstrip() for line in code.splitlines())
    return code.strip() + ("\n" if code.strip() else "")


def is_probably_python(code: str) -> bool:
    code = code.strip()
    if not code:
        return False
    # Heuristics to catch obvious non-Python
    bad_markers = ["public static", "System.out.println", "#include <", "function(", "<?php", "</html>"]
    if any(marker in code for marker in bad_markers):
        return False
    try:
        import ast

        ast.parse(code)
        return True
    except Exception:
        # Not necessarily invalid, but we only keep syntactically valid snippets for training
        return False


def split_train_val(
    records: List[Dict], train_ratio: float, val_ratio: float, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    return train, val


def write_jsonl(path: str, items: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_training_examples(codealpaca: List[Dict]) -> List[Dict]:
    cleaned: List[Dict] = []
    for ex in codealpaca:
        inst = ex["instruction"]
        inp = ex.get("input", "")
        out = strip_code_fences(ex["output"])
        out = normalize_indentation(out)
        if not is_probably_python(out):
            continue
        cleaned.append({"instruction": inst, "input": inp, "output": out})
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for Mistral LoRA SFT.")
    parser.add_argument("--codealpaca-dir", type=str, default="", help="Optional path hint to CodeAlpaca data")
    parser.add_argument("--mbpp-dir", type=str, default="", help="Optional path hint to MBPP data")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for processed JSONL")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CodeAlpaca-20k...")
    alpaca = read_or_load_codealpaca(args.codealpaca_dir or None)
    print(f"Loaded {len(alpaca)} CodeAlpaca records.")

    print("Cleaning and filtering Python-only examples...")
    trainable = build_training_examples(alpaca)
    print(f"Kept {len(trainable)} Python-valid examples.")

    print("Splitting train/val...")
    train, val = split_train_val(trainable, args.train_ratio, args.val_ratio, args.seed)
    print(f"Train: {len(train)}; Val: {len(val)}")

    print("Loading MBPP for test set...")
    mbpp = read_or_load_mbpp(args.mbpp_dir or None)
    print(f"MBPP test tasks: {len(mbpp)}")

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(args.out_dir, "mbpp_test.jsonl"), mbpp)
    print(f"Wrote processed JSONL to {args.out_dir}")


if __name__ == "__main__":
    main()


