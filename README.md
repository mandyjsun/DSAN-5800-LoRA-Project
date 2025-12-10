## DSAN-5800 LoRA Project — Runbook (CUDA on RTX 4070 Super)

End-to-end steps to run data prep, baseline, training, and evaluation locally on your 4070S. You manage your own virtual environment; activate it before running commands.

### Requirements
- GPU: NVIDIA RTX 4070 Super (12GB)
- OS: Windows 10/11 or WSL2 (Ubuntu 22.04 recommended for bitsandbytes)
- Python: 3.13 recommended (cu118 wheels); 3.10+ also works
- NVIDIA driver: recent, supporting CUDA 11.8
- Disk space: ~25 GB

Verify GPU in Python:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Windows/WSL2 driver check:
```powershell
nvidia-smi
```

### Install dependencies

1) Install PyTorch with CUDA (example: CUDA 11.8)
```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

2) Install project packages
```bash
pip install -r requirements.txt
```
Notes:
- QLoRA uses bitsandbytes; WSL2 is recommended. On Windows native, bitsandbytes may fall back to CPU or fail. If so, see Troubleshooting or use standard LoRA.
- requirements.txt excludes torch so you can choose the matching CUDA build.

### Repo structure
- scripts/
  - prepare_data.py: normalize CodeAlpaca + MBPP into JSONL
  - baseline_infer.py: zero-shot baseline generations on MBPP
  - train_lora.py: LoRA/QLoRA SFT (TRL/PEFT)
  - eval_mbpp.py: MBPP generation + unit tests, pass@1 + syntax rate
- configs/
  - lora_r8.yaml, lora_r32.yaml: pre-tuned configs for 12GB VRAM
- notebooks/
  - demo.ipynb: quick inference + sanity tests

Suggested folders (created automatically by some scripts, but safe to create):
```bash
mkdir -p data/raw data/processed artifacts/checkpoints artifacts/merged artifacts/metrics
```

### Data preparation
```bash
python scripts/prepare_data.py \
  --out-dir data/processed \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
Outputs:
- data/processed/train.jsonl
- data/processed/val.jsonl
- data/processed/mbpp_test.jsonl (includes tests array per task)

### Baseline (zero-shot)
```bash
python scripts/baseline_infer.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --mbpp-file data/processed/mbpp_test.jsonl \
  --out-file artifacts/metrics/baseline_generations.jsonl
```

### Training (QLoRA on 12GB)
Start with rank 8:
```bash
python scripts/train_lora.py \
  --config configs/lora_r8.yaml \
  --train-file data/processed/train.jsonl \
  --val-file data/processed/val.jsonl \
  --output-dir artifacts/checkpoints/mistral7b-code-r8
```

Try rank 32:
```bash
python scripts/train_lora.py \
  --config configs/lora_r32.yaml \
  --train-file data/processed/train.jsonl \
  --val-file data/processed/val.jsonl \
  --output-dir artifacts/checkpoints/mistral7b-code-r32
```

Tips if OOM:
- Lower max_seq_len in config (e.g., 1024 → 768/512)
- Keep micro-batch size = 1; increase grad_accum
- Close other GPU apps

Use standard LoRA (no QLoRA) if bitsandbytes fails on Windows native (remove use_qlora and run fp16).

### Evaluation on MBPP
```bash
python scripts/eval_mbpp.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --lora-dir artifacts/checkpoints/mistral7b-code-r8 \
  --mbpp-file data/processed/mbpp_test.jsonl \
  --out-file artifacts/metrics/mistral7b-code-r8-mbpp_results.json
```
Outputs include:
- pass_rate (pass@1)
- syntax_rate
- per-task results

Security note: evaluation executes test strings with exec; use in trusted environments only.

### Results
See [analysis.md](./analysis.ipynb) 

