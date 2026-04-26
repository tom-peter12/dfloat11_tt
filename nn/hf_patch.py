"""HuggingFace Transformers integration: load a compressed DF11 model."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger

from .df11_embedding import DF11Embedding
from .df11_linear import DF11Linear
from .df11_block import DF11TransformerBlock


DEFAULT_PATTERNS: List[str] = [
    r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    r".*\.mlp\.(gate_proj|up_proj|down_proj)",
    r"model\.embed_tokens",
    r"lm_head",
]


def _should_compress(name: str, patterns: List[str]) -> bool:
    return any(re.fullmatch(p, name) for p in patterns)


def from_pretrained_df11(
    model_name_or_path: str,
    compressed_bundle_path: Union[str, Path],
    tt_device: Any,
    patterns: List[str] = DEFAULT_PATTERNS,
    wrap_blocks: bool = True,
    **hf_kwargs: Any,
) -> nn.Module:
    """Load a HuggingFace model with DFloat11-TT compressed weights.

    Loads the base BF16 model structure, then replaces every targeted
    nn.Linear with DF11Linear and every targeted nn.Embedding with
    DF11Embedding, both backed by compressed tensors.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        compressed_bundle_path: Path to .df11tt bundle file from the compressor.
        tt_device: Tenstorrent device handle (from ttnn.open_device or MeshDevice).
        patterns: Regex patterns for module names to swap.
        wrap_blocks: If True, wrap transformer blocks with DF11TransformerBlock
                     for batched decompression.
        **hf_kwargs: Extra kwargs forwarded to AutoModelForCausalLM.from_pretrained.

    Returns:
        The model with DF11Linear modules, ready for inference.
    """
    from transformers import AutoModelForCausalLM
    from dfloat11_tt.compress.bundle import load_model_bundle

    logger.info(f"Loading base model {model_name_or_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        **hf_kwargs,
    )
    model.eval()

    logger.info(f"Loading compressed bundle from {compressed_bundle_path} ...")
    bundles: Dict[str, Dict] = load_model_bundle(compressed_bundle_path)
    logger.info(f"Bundle contains {len(bundles)} compressed tensors.")

    replaced = 0
    for name, module in list(model.named_modules()):
        if not _should_compress(name, patterns):
            continue
        if name not in bundles:
            logger.warning(f"Module {name} matched pattern but not found in bundle — skipping.")
            continue

        bundle = bundles[name]
        shape = bundle["shape"]

        if isinstance(module, nn.Linear):
            df11 = DF11Linear(
                in_features  = module.in_features,
                out_features = module.out_features,
                bias         = module.bias is not None,
                device       = tt_device,
            )
            if module.bias is not None:
                import ttnn
                bias_tt = ttnn.from_torch(
                    module.bias.data.bfloat16(),
                    dtype=ttnn.bfloat16,
                    device=tt_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                df11.bias = bias_tt
            df11._module_name = name
            df11.load_bundle(bundle, tt_device)
            _set_submodule(model, name, df11)
            replaced += 1

        elif isinstance(module, nn.Embedding):
            df11 = DF11Embedding(
                num_embeddings = module.num_embeddings,
                embedding_dim  = module.embedding_dim,
                padding_idx    = module.padding_idx,
                device         = tt_device,
            )
            df11._module_name = name
            df11.load_bundle(bundle, tt_device)
            _set_submodule(model, name, df11)
            replaced += 1

    logger.info(f"Replaced {replaced} modules with DF11Linear.")

    if wrap_blocks:
        _wrap_transformer_blocks(model, tt_device)

    return model


def clear_df11_weight_caches(model: nn.Module) -> None:
    """Release cached decompressed weights from all DF11 modules."""
    cleared = 0
    for module in model.modules():
        if isinstance(module, (DF11Embedding, DF11Linear)):
            module.clear_weight_cache()
            cleared += 1
    if cleared:
        logger.info(f"Cleared DF11 weight caches for {cleared} modules.")


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a nested submodule by dotted name path."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _wrap_transformer_blocks(model: nn.Module, tt_device: Any) -> None:
    """Wrap transformer decoder layers with DF11TransformerBlock."""
    # Heuristic: look for lists/ModuleLists named 'layers' or 'h' (GPT-2) or 'blocks'
    for attr_name in ("layers", "h", "blocks", "decoder_layers"):
        container = None
        for part in ("model", "transformer", ""):
            root = getattr(model, part, None) if part else model
            if root is not None and hasattr(root, attr_name):
                container = getattr(root, attr_name)
                break
        if container is not None and hasattr(container, "__iter__"):
            for i, block in enumerate(container):
                wrapped = DF11TransformerBlock(block, tt_device)
                container[i] = wrapped
            logger.info(f"Wrapped {len(container)} transformer blocks with DF11TransformerBlock.")
            return
    logger.info("Could not auto-detect transformer block container; skipping block wrapping.")
