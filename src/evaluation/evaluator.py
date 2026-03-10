"""Run evaluation against base, SFT, and DPO model checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

import structlog
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import FunctionCallMetrics, evaluate_single

logger = structlog.get_logger(__name__)


def generate_predictions(
    model_path: str,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> list[str]:
    """Generate completions from a model checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    predictions: list[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        predictions.extend(decoded)

        if i % (batch_size * 10) == 0:
            logger.info("generating", done=i, total=len(prompts))

    del model
    torch.cuda.empty_cache()
    return predictions


def run_evaluation(
    model_path: str,
    tokenizer,
    test_dataset: Dataset,
    max_samples: int = 1000,
) -> dict:
    """Evaluate a model on the test set and return aggregated metrics."""
    n = min(max_samples, len(test_dataset))
    samples = test_dataset.select(range(n))

    prompts = samples["text"]
    references = [
        json.loads(fc)[0] if fc and json.loads(fc) else ""
        for fc in samples["function_calls"]
    ]
    references = [json.dumps(r) if isinstance(r, dict) else str(r) for r in references]
    allowed_lists = [
        [fn.get("name", "") for fn in json.loads(f)]
        for f in samples["functions"]
    ]

    t0 = time.perf_counter()
    predictions = generate_predictions(model_path, tokenizer, prompts)
    elapsed = time.perf_counter() - t0

    metrics = FunctionCallMetrics()
    for pred, ref, allowed in zip(predictions, references, allowed_lists):
        evaluate_single(pred, ref, allowed, metrics)

    result = metrics.to_dict()
    result["inference_time_s"] = round(elapsed, 1)
    result["throughput_sps"] = round(n / elapsed, 1)
    return result


def compare_models(
    base_model: str,
    sft_model: str,
    dpo_model: str,
    tokenizer_path: str,
    test_dataset: Dataset,
    output_path: str = "experiments/results.json",
    max_samples: int = 1000,
) -> dict:
    """Run evaluation on all three checkpoints and write a comparison report."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results: dict = {}
    for name, path in [("base", base_model), ("sft_lora", sft_model), ("dpo", dpo_model)]:
        logger.info("evaluating", model=name, path=path)
        results[name] = run_evaluation(path, tokenizer, test_dataset, max_samples)

    report = {
        "base_model": base_model,
        "dataset": "glaiveai/glaive-function-calling-v2",
        "test_samples": max_samples,
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("results_saved", path=str(out))

    _print_table(results)
    return report


def _print_table(results: dict) -> None:
    metrics = [
        "valid_json_rate",
        "exact_tool_match_rate",
        "parameter_f1",
        "hallucination_rate",
        "rouge_l",
    ]
    col_w = 13
    print("\n" + "=" * 72)
    header = f"{'Metric':<26} {'Base':>{col_w}} {'SFT+LoRA':>{col_w}} {'DPO':>{col_w}} {'Δ Base→DPO':>{col_w}}"
    print(header)
    print("-" * 72)
    for m in metrics:
        base_v = results.get("base", {}).get(m, 0.0)
        sft_v = results.get("sft_lora", {}).get(m, 0.0)
        dpo_v = results.get("dpo", {}).get(m, 0.0)
        delta = dpo_v - base_v
        print(
            f"{m:<26} {base_v:>{col_w}.3f} {sft_v:>{col_w}.3f} "
            f"{dpo_v:>{col_w}.3f} {delta:>+{col_w}.3f}"
        )
    print("=" * 72 + "\n")
