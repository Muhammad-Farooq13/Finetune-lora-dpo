# Fine-Tuning Qwen2.5 for Function Calling with LoRA + DPO

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-green)](https://github.com/huggingface/peft)
[![TRL](https://img.shields.io/badge/TRL-DPO-orange)](https://github.com/huggingface/trl)
[![CI](https://github.com/Muhammad-Farooq13/project4-finetune-lora-dpo/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/project4-finetune-lora-dpo/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Muhammad--Farooq13-181717?logo=github)](https://github.com/Muhammad-Farooq13)

> I got tired of base models hallucinating function names and mangling JSON arguments when used as tool-calling agents. This repo shows how much a targeted LoRA fine-tune + DPO preference step actually moves the needle — with concrete before/after numbers.

---

## The Problem

Base LLMs are bad at structured output for tool/function calling. Given a schema like:

```json
{"name": "get_weather", "parameters": {"location": "string", "units": "string"}}
```

...and a user message "What's the weather in Lahore?", a base model will:
- Return free-form text ~77% of the time instead of JSON
- Hallucinate function names not in the schema ~61% of the time
- Get parameter names/types wrong even when it does output JSON

SFT with LoRA fixes the structural problem. DPO then improves preference — teaching the model to *prefer* correct calls over plausible-looking-but-wrong ones.

---

## Results (Qwen2.5-1.5B-Instruct · glaive-function-calling-v2 · 1 000 test samples)

> **Note:** These numbers are from a single run on 91k training examples. Results will vary with different random seeds or dataset subsets — I've seen tool-match fluctuate ±2–3 pp between runs. The trends (SFT >> base, DPO adds on top) are consistent though.

| Metric | Base Model | + SFT (LoRA) | + DPO | Δ Base→DPO |
|--------|:----------:|:------------:|:-----:|:----------:|
| Valid JSON output (%) | 23.4 | 91.2 | **93.7** | +70.3 pp |
| Exact tool/function match (%) | 18.7 | 74.3 | **79.8** | +61.1 pp |
| Parameter extraction F1 | 0.214 | 0.714 | **0.771** | +0.557 |
| Hallucination rate (%) | 61.3 | 12.4 | **8.9** | −52.4 pp |
| ROUGE-L | 0.183 | 0.691 | **0.741** | +0.558 |

**Training cost:**
- SFT (LoRA r=16, 3 epochs, 91k samples, QLoRA 4-bit): 2h 34m on 4× A100 40GB, peak 18.3 GB VRAM
- DPO (beta=0.1, 1 epoch): 47m on the same hardware
- Trainable parameters: **13.1M / 1.54B total (0.85%)** — full fine-tune would need ~10× more memory

---

## What I Learned

The SFT step does almost all the heavy lifting (+55 pp on tool match). DPO adds a meaningful but smaller improvement (+5.5 pp) — it mainly helps with:
- Reducing hallucinations (calling functions not in the schema)
- Improving parameter quality even when the function name was already right

The DPO rejection strategy matters more than I expected. "Wrong function name" rejections were the most valuable — the model was already underfitting on hallucination avoidance after SFT, so seeing explicit (chosen: correct_fn, rejected: wrong_fn) pairs had the biggest impact. "Wrong parameters" rejections helped less; the model's parameter errors post-SFT were mostly about value types, and the token-level signal from DPO is noisier there.

I used `glaiveai/glaive-function-calling-v2` (112k examples, single and multi-turn, real function schemas) as the training set. It's better than synthetic data because the function schemas and conversations are diverse — the model sees ~3k unique function names, not just "get_weather" repeated.

One thing that tripped me up early: I initially forgot to set `padding_side="right"` on the tokenizer for SFT, which caused the model to attend to padding tokens on the left and produced garbage outputs for the first few hundred steps. If your SFT loss doesn't start dropping within the first 50 steps, check your padding configuration first.

---

## How It Works

```
Dataset (glaiveai/glaive-function-calling-v2)
    │
    ├── loader.py          Parse system prompt → extract function schemas
    │                      Parse chat → turns list
    │
    ├── formatter.py       Apply model-specific chat template
    │                      Embed function schemas in system prompt
    │
    ▼
Phase 1: SFT with LoRA/QLoRA
    ├── Base model: Qwen2.5-1.5B-Instruct (loaded in 4-bit NF4)
    ├── LoRA: r=16, α=32, target all projection layers
    ├── SFTTrainer (packing=True for efficiency)
    └── Checkpoint: checkpoints/sft/

    │
    ▼
Phase 2: DPO
    ├── preference_builder.py  Build (prompt, chosen, rejected) triplets
    │     Strategies: wrong_function | wrong_params | missing_params | hallucinate
    ├── DPOTrainer (beta=0.1, sigmoid loss)
    │   ref_model=None  ← TRL handles reference via LoRA adapter disable
    └── Checkpoint: checkpoints/dpo/

    │
    ▼
Evaluation
    ├── valid_json_rate, exact_tool_match, parameter_f1
    ├── hallucination_rate (called fn not in schema)
    └── ROUGE-L
```

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Base model | Qwen2.5-1.5B-Instruct | — |
| LoRA adapters | `peft` | ≥ 0.10 |
| QLoRA quantisation | `bitsandbytes` | ≥ 0.43 |
| SFT training | `trl` SFTTrainer | ≥ 0.8.6 |
| DPO training | `trl` DPOTrainer | ≥ 0.8.6 |
| Model loading | `transformers` | ≥ 4.40 |
| Dataset | `datasets` | ≥ 2.18 |
| Experiment tracking | `wandb` | ≥ 0.16 |

**Alternative base models** (swap `MODEL_NAME` in `.env`):
- `mistralai/Mistral-7B-Instruct-v0.2` — better quality, needs more VRAM
- `meta-llama/Llama-3.2-1B-Instruct` — open weights, similar size
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` — no gating, easiest to access

---

## Quick Start

### Prerequisites
- Python 3.11+
- NVIDIA GPU with ≥ 16 GB VRAM (for 4-bit QLoRA) — or use CPU for inference only
- HuggingFace account (free) to download the dataset + model

### Install

```bash
git clone https://github.com/Muhammad-Farooq13/project4-finetune-lora-dpo.git
cd project4-finetune-lora-dpo

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN and optionally WANDB_API_KEY
```

### Step 1 — Download & preprocess data

```bash
python scripts/download_data.py
# Full dataset: ~112k examples, ~5 min
# For quick experimentation:
python scripts/download_data.py --max-samples 5000
```

### Step 2 — SFT with LoRA

```bash
python scripts/train_sft.py
# Or with overrides:
python scripts/train_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --lr 2e-4
```

### Step 3 — DPO

```bash
python scripts/train_dpo.py --sft-checkpoint checkpoints/sft
```

### Step 4 — Evaluate all three models

```bash
python scripts/evaluate.py
# Outputs a table like the one at the top of this README
# Also writes experiments/results.json
```

### Step 5 — Run the demo

```bash
python scripts/demo.py --adapter checkpoints/dpo
# Interactive: type a natural language request, see the function call
```

---

## Configuration

All hyperparameters live in `src/config/training_config.py` as dataclasses. The key ones:

### LoRA

| Parameter | Default | Notes |
|-----------|---------|-------|
| `r` | `16` | Rank — higher = more capacity + more params |
| `lora_alpha` | `32` | Scaling factor (effective lr = alpha/r × lr) |
| `lora_dropout` | `0.05` | Regularisation |
| `target_modules` | all proj layers | q/k/v/o + gate/up/down |
| `use_rslora` | `False` | Rank-stabilised LoRA — useful for r>64 |

### SFT

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Change via env var |
| `num_train_epochs` | `3` | |
| `learning_rate` | `2e-4` | Works well for LoRA; lower for full FT |
| `per_device_train_batch_size` | `4` | Effective batch = 4 × 4 accum = 16 |
| `load_in_4bit` | `true` | QLoRA — disable for A100/H100 with BF16 |
| `packing` | `true` | Packs short sequences together, ~2× throughput |

### DPO

| Parameter | Default | Notes |
|-----------|---------|-------|
| `beta` | `0.1` | KL penalty — lower = more deviation from SFT policy |
| `loss_type` | `sigmoid` | Standard DPO; alternatives: `ipo`, `hinge`, `kto_pair` |
| `learning_rate` | `5e-5` | Much lower than SFT — fine adjustment |
| `num_train_epochs` | `1` | DPO overfits quickly; 1 epoch is usually enough |

---

## Dataset

**`glaiveai/glaive-function-calling-v2`** — [HuggingFace Hub](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

- 112,960 examples
- Single-turn and multi-turn conversations
- Diverse real function schemas (~3k unique function names)
- Mix of: correct function calls, multi-step calls, tool response → follow-up answer

The dataset format:
```python
{
  "system": "SYSTEM: You are a helpful assistant with access to the following functions...\n{\"name\": \"get_weather\", ...}",
  "chat": "USER: What's the weather in Lahore?\nASSISTANT: <functioncall> {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Lahore\"}} </s>\nFUNCTION RESPONSE: {\"temperature\": 32, \"condition\": \"sunny\"}\nASSISTANT: The weather in Lahore is 32°C and sunny. </s>"
}
```

`src/data/loader.py` parses this into structured turns and extracts function schemas.

---

## DPO Preference Construction

Since `glaive-function-calling-v2` only has correct completions, I build rejected pairs synthetically using four strategies:

| Strategy | What it does | % of pairs |
|----------|-------------|-----------|
| `wrong_function` | Swap function name for another from schema | 25% |
| `wrong_params` | Corrupt argument values (reverse strings, negate numbers) | 25% |
| `missing_params` | Drop half the required parameters | 25% |
| `hallucinate` | Call a function not in the schema | 25% |

This is done in `src/data/preference_builder.py`. The model sees "correct call is **chosen**, this broken call is **rejected**" for each of these failure modes.

---

## Project Structure

```
project4-finetune-lora-dpo/
├── src/
│   ├── config/
│   │   └── training_config.py     # All hyperparameters as dataclasses
│   ├── data/
│   │   ├── loader.py              # Download + parse glaive dataset
│   │   ├── formatter.py           # Chat template formatting
│   │   └── preference_builder.py  # Build DPO (chosen, rejected) pairs
│   ├── model/
│   │   ├── loader.py              # Load model + tokenizer (+ QLoRA)
│   │   └── lora_config.py         # Build + apply LoRA adapters
│   ├── training/
│   │   ├── sft_trainer.py         # SFTTrainer wrapper
│   │   └── dpo_trainer.py         # DPOTrainer wrapper
│   ├── evaluation/
│   │   ├── metrics.py             # valid_json, tool_match, param_f1 etc.
│   │   └── evaluator.py           # Run eval across base/SFT/DPO
│   └── inference/
│       └── pipeline.py            # Load adapter + run predictions
├── scripts/
│   ├── download_data.py
│   ├── train_sft.py
│   ├── train_dpo.py
│   ├── evaluate.py
│   └── demo.py
├── experiments/
│   └── results.json               # Stored benchmark results
├── data/
│   └── samples/
│       └── function_schemas.json  # Example schemas for demo/testing
└── tests/
    ├── test_metrics.py
    ├── test_formatter.py
    └── test_preference_builder.py
```

---

## Roadmap

Things I know need work:

- [ ] Multi-turn conversation handling — currently only the first assistant turn is supervised; tool responses and follow-up turns are ignored
- [ ] Only tested Qwen2.5-1.5B — Mistral-7B should give better absolute numbers but I haven't run it yet
- [ ] DPO rejection pairs are all synthetic — real human preference data (e.g., from RLHF Arena) would be better but harder to get for function calling specifically
- [ ] No vLLM integration — inference is plain HuggingFace `generate()` which is slow; would be straightforward to add
- [ ] `evaluator.py` runs models sequentially — could parallelize across GPUs
- [ ] No confidence score / logprob output from the inference pipeline

---

## License

MIT © 2026 Muhammad Farooq — see [LICENSE](LICENSE)

---

Built by [Muhammad Farooq](https://github.com/Muhammad-Farooq13) · [mfarooqshafee333@gmail.com](mailto:mfarooqshafee333@gmail.com)
