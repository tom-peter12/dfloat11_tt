"""Standard benchmark evaluation: MMLU, TruthfulQA, WikiText-2 perplexity, C4 perplexity.

Mirrors Table 2 of the DFloat11 paper.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger


def _run_lm_eval(
    model_id: str,
    tasks: List[str],
    limit: Optional[int] = None,
    device: str = "cpu",
) -> Dict:
    """Run lm-evaluation-harness and return parsed results."""
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_id},dtype=bfloat16",
        "--tasks", ",".join(tasks),
        "--device", device,
        "--output_path", "/tmp/lm_eval_output",
        "--log_samples",
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    logger.info(f"Running lm-eval: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            logger.error(f"lm-eval failed: {result.stderr[:500]}")
            return {"error": result.stderr[:500]}
        # Parse output JSON from /tmp/lm_eval_output
        import glob
        jsons = glob.glob("/tmp/lm_eval_output/**/*.json", recursive=True)
        if not jsons:
            return {"error": "no output json found"}
        with open(jsons[0]) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}


def _compute_perplexity(
    model_id: str,
    dataset: str,
    split: str = "test",
    max_samples: int = 2048,
    device: str = "cpu",
) -> float:
    """Compute word-level perplexity on WikiText-2 or C4."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import math

    logger.info(f"Computing perplexity on {dataset}/{split} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    if dataset == "wikitext2":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(data["text"][:max_samples])
    elif dataset == "c4":
        data = load_dataset("allenai/c4", "en", split=split, streaming=True)
        texts = []
        for i, item in enumerate(data):
            if i >= max_samples: break
            texts.append(item["text"])
        text = "\n\n".join(texts)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    encodings = tokenizer(text, return_tensors="pt")
    max_len = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    for begin_loc in range(0, seq_len, stride):
        end_loc   = min(begin_loc + max_len, seq_len)
        trg_len   = end_loc - max(begin_loc, begin_loc + max_len - stride)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    logger.info(f"Perplexity on {dataset}: {ppl:.3f}")
    return ppl


def run_benchmark_test(
    model_id: str,
    results_dir: Path,
    device: str = "cpu",
    limit: Optional[int] = None,
) -> Dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {"model_id": model_id, "test": "benchmarks"}

    # MMLU (5-shot)
    logger.info("[benchmarks] Running MMLU ...")
    mmlu_res = _run_lm_eval(model_id, ["mmlu"], limit=limit, device=device)
    results["mmlu"] = mmlu_res

    # TruthfulQA
    logger.info("[benchmarks] Running TruthfulQA ...")
    tqa_res = _run_lm_eval(model_id, ["truthfulqa_mc2"], limit=limit, device=device)
    results["truthfulqa"] = tqa_res

    # WikiText-2 perplexity
    try:
        results["wikitext2_ppl"] = _compute_perplexity(model_id, "wikitext2", device=device)
    except Exception as e:
        logger.warning(f"WikiText-2 perplexity failed: {e}")
        results["wikitext2_ppl"] = None

    # C4 perplexity
    try:
        results["c4_ppl"] = _compute_perplexity(model_id, "c4", device=device)
    except Exception as e:
        logger.warning(f"C4 perplexity failed: {e}")
        results["c4_ppl"] = None

    out_path = results_dir / "benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"[benchmarks] Saved to {out_path}")
    return results
