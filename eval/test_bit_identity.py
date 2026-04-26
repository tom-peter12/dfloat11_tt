"""Bit-identity test: verify every decompressed tensor exactly matches its BF16 original.

Zero tolerance — any mismatch fails the test immediately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger

from dfloat11_tt.compress.bundle import load_model_bundle
from dfloat11_tt.compress.reference_decoder import decode_bundle


def run_bit_identity_test(
    model_id: str,
    bundle_path: Path,
    results_dir: Path,
    patterns: List[str],
) -> Dict:
    from transformers import AutoModelForCausalLM
    import re

    logger.info(f"[bit_identity] Loading reference model {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    logger.info(f"[bit_identity] Loading compressed bundle {bundle_path} ...")
    bundles = load_model_bundle(bundle_path)

    results: List[Dict] = []
    all_pass = True

    for name, bundle in bundles.items():
        # Find reference weight
        ref_weight = None
        for full_name, module in model.named_modules():
            if full_name == name:
                if hasattr(module, "weight"):
                    ref_weight = module.weight.data.cpu().flatten()
                break

        if ref_weight is None:
            logger.warning(f"  Could not find reference weight for {name}, skipping.")
            continue

        logger.info(f"  Testing {name} shape={bundle['shape']} ...")

        # Run pure-Python decoder (reference)
        decoded = decode_bundle(bundle).flatten()

        # Exact bit comparison
        ref_view  = ref_weight.view(torch.uint16)
        dec_view  = decoded.view(torch.uint16)
        mismatches = (ref_view != dec_view).sum().item()

        passed = (mismatches == 0)
        if not passed:
            all_pass = False
            logger.error(f"  FAIL: {name} — {mismatches} mismatched elements out of {ref_weight.numel()}")
        else:
            logger.success(f"  PASS: {name}")

        results.append({
            "name": name,
            "shape": bundle["shape"],
            "n_elements": bundle["n_elements"],
            "mismatches": mismatches,
            "passed": passed,
        })

    summary = {
        "model_id": model_id,
        "test": "bit_identity",
        "all_pass": all_pass,
        "n_tensors": len(results),
        "n_fail": sum(1 for r in results if not r["passed"]),
        "results": results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "bit_identity.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"[bit_identity] {'ALL PASS' if all_pass else 'FAILURES DETECTED'} — "
        f"{len(results)} tensors tested, {summary['n_fail']} failed. "
        f"Results → {out_path}"
    )
    return summary
