"""CLI: compress a HuggingFace model to DFloat11-TT format.

Usage:
    python -m dfloat11_tt.compress \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --out ./compressed/llama-3.1-8b.df11tt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger

from .bundle import save_model_bundle
from .compressor import compress_tensor
from .reference_decoder import decode_bundle


# Default patterns matching original DFloat11: all Linear and Embedding weights.
DEFAULT_PATTERNS: List[str] = [
    r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    r".*\.mlp\.(gate_proj|up_proj|down_proj)",
    r"model\.embed_tokens",
    r"lm_head",
]


def _should_compress(name: str, patterns: List[str]) -> bool:
    return any(re.fullmatch(p, name) for p in patterns)


def compress_model(
    model_name_or_path: str,
    out_path: Path,
    patterns: List[str] = DEFAULT_PATTERNS,
    check_correctness: bool = True,
    device: str = "cpu",
) -> None:
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading model {model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    compressed: Dict[str, Dict] = {}
    total_orig_bytes = 0
    total_comp_bytes = 0

    for name, module in model.named_modules():
        if not _should_compress(name, patterns):
            continue

        if isinstance(module, nn.Linear):
            weight = module.weight.data
        elif isinstance(module, nn.Embedding):
            weight = module.weight.data
        else:
            continue

        if weight.dtype != torch.bfloat16:
            logger.warning(f"Skipping {name}: dtype {weight.dtype} (expected bfloat16)")
            continue

        logger.info(f"Compressing {name} shape={list(weight.shape)} ...")
        bundle = compress_tensor(weight)

        if check_correctness:
            decoded = decode_bundle(bundle)
            original = weight.detach().cpu().flatten()
            if not torch.equal(decoded.flatten(), original):
                raise RuntimeError(
                    f"Bit-identity check FAILED for {name}. "
                    "The decoded tensor does not match the original."
                )
            logger.info(f"  ✓ bit-identity verified")

        orig_bytes = weight.numel() * 2
        comp_bytes = (
            bundle["encoded_exponent"].nbytes
            + bundle["sign_mantissa"].nbytes
            + bundle["gaps"].nbytes
            + bundle["output_positions"].nbytes
            + bundle["luts"].nbytes
        )
        ratio = comp_bytes / orig_bytes * 100
        logger.info(f"  compression: {orig_bytes/1e6:.1f} MB → {comp_bytes/1e6:.1f} MB ({ratio:.1f}%)")

        compressed[name] = bundle
        total_orig_bytes += orig_bytes
        total_comp_bytes += comp_bytes

    if not compressed:
        logger.error("No matching modules found. Check --patterns.")
        sys.exit(1)

    logger.info(f"Saving bundle to {out_path} ...")
    save_model_bundle(compressed, out_path)

    overall_ratio = total_comp_bytes / total_orig_bytes * 100
    logger.info(
        f"Done. Compressed {len(compressed)} tensors. "
        f"Overall: {total_orig_bytes/1e9:.3f} GB → {total_comp_bytes/1e9:.3f} GB "
        f"({overall_ratio:.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DFloat11-TT model compressor")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--out", required=True, type=Path, help="Output .df11tt bundle path")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=None,
        help="Regex patterns for module names to compress (default: all Linear + Embedding)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip bit-identity verification (faster but unsafe)",
    )
    parser.add_argument("--device", default="cpu", help="Device for model loading (cpu or cuda)")
    args = parser.parse_args()

    patterns = args.patterns if args.patterns else DEFAULT_PATTERNS
    compress_model(
        args.model,
        args.out,
        patterns=patterns,
        check_correctness=not args.no_check,
        device=args.device,
    )


if __name__ == "__main__":
    main()
