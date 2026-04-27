"""DF11TransformerBlock: pre-decompress DF11Linear weights before block forward."""
from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn
from loguru import logger

from .df11_linear import DF11Linear, _env_flag


class DF11TransformerBlock(nn.Module):
    """Wrap a transformer block and materialize all DF11Linear weights up front.

    Before each forward pass, all compressed weight tensors in the block are
    decompressed and stored temporarily, then freed after the block forward.

    This is a correctness-first stepping stone toward the paper's batched
    decompression idea: the decompressions are still issued sequentially, but
    the linears now actually consume the pre-materialized weights.
    """

    def __init__(self, block: nn.Module, tt_device: Any) -> None:
        super().__init__()
        self.block = block
        self._tt_device = tt_device
        self._df11_linears: List["DF11Linear"] = []
        self._decompressed_weights: Dict[int, Any] = {}

        for _name, module in block.named_modules():
            if isinstance(module, DF11Linear):
                self._df11_linears.append(module)

    def _decompress_all(self) -> None:
        """Decompress all DF11Linear weights used by this block."""
        trace = _env_flag("DFLOAT11_TRACE_LINEAR", False)
        cache_weights = _env_flag("DFLOAT11_CACHE_WEIGHTS", True)

        for layer in self._df11_linears:
            if layer._enc_exp is None:
                continue
            if cache_weights and layer._cached_weight_linear_tt is not None:
                continue

            weight_linear_tt, decompress_ms, layout_ms = layer.materialize_weight_linear_tt(
                trace=trace
            )
            if trace:
                logger.info(
                    f"[df11-block] {layer._module_name}: materialized "
                    f"shape=[{layer.out_features},{layer.in_features}] "
                    f"decompress={decompress_ms:.2f}ms layout={layout_ms:.2f}ms "
                    f"cache={'on' if cache_weights else 'off'}"
                )
            if cache_weights:
                layer._cached_weight_linear_tt = weight_linear_tt
            else:
                self._decompressed_weights[id(layer)] = weight_linear_tt
                layer._block_weight_linear_tt = weight_linear_tt

    def _free_all(self) -> None:
        """Free all temporarily materialized weights."""
        for w in self._decompressed_weights.values():
            w.deallocate(force=True)
        self._decompressed_weights.clear()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Decompress all weights, run block forward, free weights."""
        self._decompress_all()

        try:
            result = self.block(*args, **kwargs)
        finally:
            for layer in self._df11_linears:
                layer._block_weight_linear_tt = None
            self._free_all()

        return result
