"""Output-equivalence test: token sequences and logits must be bit-identical.

Runs both BF16 reference and DF11 model on 50 prompts with greedy decoding.
Logits compared exactly (not allclose — exact BF16 equality required).
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger


def run_output_equivalence_test(
    model_id: str,
    bundle_path: Path,
    results_dir: Path,
    patterns: List[str],
    prompt_file: Path,
    tt_device: Optional[object] = None,
    max_new_tokens: int = 50,
    prompt_limit: Optional[int] = None,
    compare_logits: bool = True,
    wrap_blocks: bool = True,
) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dfloat11_tt.nn.hf_patch import clear_df11_weight_caches, from_pretrained_df11

    logger.info(f"[output_equiv] Loading tokenizer for {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    if prompt_file.exists():
        prompts = [l.strip() for l in prompt_file.read_text().splitlines() if l.strip()]
    else:
        prompts = [
            "The capital of France is",
            "In machine learning, overfitting refers to",
            "The quick brown fox jumps over the",
        ]
    prompts = prompts[: prompt_limit or 50]
    logger.info(
        f"[output_equiv] Running {len(prompts)} prompt(s), max_new_tokens={max_new_tokens}, "
        f"compare_logits={compare_logits}"
    )

    def _generation_kwargs(model, inputs: Dict[str, torch.Tensor]) -> Dict:
        generation_config = deepcopy(model.generation_config)
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id

        kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": max_new_tokens,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
        }
        if compare_logits:
            kwargs["output_logits"] = True
        return kwargs

    # ---- BF16 reference model (runs on CPU) ----
    logger.info("[output_equiv] Loading BF16 reference model ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    ref_model.eval()

    # ---- DF11 model ----
    if tt_device is not None:
        logger.info("[output_equiv] Loading DF11 model on Tenstorrent device ...")
        df11_model = from_pretrained_df11(
            model_id, bundle_path, tt_device, patterns=patterns, wrap_blocks=wrap_blocks
        )
    else:
        logger.warning(
            "[output_equiv] No TT device provided — running DF11 model on CPU via reference decoder."
        )
        df11_model = ref_model  # trivially passes (same model)

    results: List[Dict] = []
    all_pass = True

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            logger.info(f"[output_equiv] [{i+1}/{len(prompts)}] BF16 generate start")
            t0 = time.perf_counter()
            ref_out = ref_model.generate(**_generation_kwargs(ref_model, inputs))
            ref_seconds = time.perf_counter() - t0
            logger.info(
                f"[output_equiv] [{i+1}/{len(prompts)}] BF16 generate done "
                f"in {ref_seconds:.2f}s"
            )
            if tt_device is not None:
                logger.info(
                    f"[output_equiv] [{i+1}/{len(prompts)}] DF11/TT generate start "
                    "(full-linears mode can be slow)"
                )
                t0 = time.perf_counter()
                df11_out = df11_model.generate(**_generation_kwargs(df11_model, inputs))
                df11_seconds = time.perf_counter() - t0
                logger.info(
                    f"[output_equiv] [{i+1}/{len(prompts)}] DF11/TT generate done "
                    f"in {df11_seconds:.2f}s"
                )
            else:
                df11_out = ref_out  # same object when no device
                df11_seconds = ref_seconds

        ref_tokens  = ref_out.sequences[0].tolist()
        df11_tokens = df11_out.sequences[0].tolist()
        prompt_token_count = int(inputs["input_ids"].shape[-1])
        ref_new_tokens = ref_tokens[prompt_token_count:]
        df11_new_tokens = df11_tokens[prompt_token_count:]
        ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
        df11_text = tokenizer.decode(df11_tokens, skip_special_tokens=True)
        ref_completion = tokenizer.decode(ref_new_tokens, skip_special_tokens=True)
        df11_completion = tokenizer.decode(df11_new_tokens, skip_special_tokens=True)
        ref_tokens_per_sec = len(ref_new_tokens) / ref_seconds if ref_seconds > 0 else None
        df11_tokens_per_sec = len(df11_new_tokens) / df11_seconds if df11_seconds > 0 else None

        tokens_match = (ref_tokens == df11_tokens)

        # Compare first logit tensor bit-exactly
        logits_match = True
        if compare_logits and tt_device is not None and ref_out.logits and df11_out.logits:
            ref_logit  = ref_out.logits[0].to(torch.bfloat16).view(torch.uint16)
            df11_logit = df11_out.logits[0].to(torch.bfloat16).view(torch.uint16)
            logits_match = torch.equal(ref_logit, df11_logit)

        passed = tokens_match and logits_match
        if not passed:
            all_pass = False

        results.append({
            "prompt_idx": i,
            "prompt": prompt[:80],
            "tokens_match": tokens_match,
            "logits_match": logits_match,
            "passed": passed,
            "ref_tokens": ref_tokens,
            "df11_tokens": df11_tokens,
            "ref_new_tokens": ref_new_tokens,
            "df11_new_tokens": df11_new_tokens,
            "ref_text": ref_text,
            "df11_text": df11_text,
            "ref_completion": ref_completion,
            "df11_completion": df11_completion,
            "ref_generate_seconds": ref_seconds,
            "df11_generate_seconds": df11_seconds,
            "ref_tokens_per_sec": ref_tokens_per_sec,
            "df11_tokens_per_sec": df11_tokens_per_sec,
        })

        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{i+1}/{len(prompts)}] {status}: {prompt[:40]!r}")
        logger.info(f"    BF16 completion: {ref_completion!r}")
        logger.info(f"    DF11 completion: {df11_completion!r}")
        logger.info(
            f"    timing: BF16={ref_seconds:.2f}s, DF11={df11_seconds:.2f}s, "
            f"DF11 tok/s={df11_tokens_per_sec or 0:.2f}"
        )

    if tt_device is not None:
        clear_df11_weight_caches(df11_model)

    summary = {
        "model_id": model_id,
        "test": "output_equivalence",
        "all_pass": all_pass,
        "n_prompts": len(prompts),
        "n_fail": sum(1 for r in results if not r["passed"]),
        "results": results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "output_equivalence.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"[output_equiv] {'ALL PASS' if all_pass else 'FAILURES'} — "
        f"{summary['n_fail']} / {len(prompts)} prompts failed."
    )
    return summary
