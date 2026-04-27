"""DF11TransformerBlock: batches decompression of all DF11Linear layers in one Program launch."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


class DF11TransformerBlock(nn.Module):
    """Wraps a transformer block, pre-decompressing all DF11Linear weights as a batch.

    Before each forward pass, all compressed weight tensors in the block are
    decompressed and stored temporarily, then freed after the block forward.

    This mirrors the paper's batched decompression idea, but the current
    implementation still issues the decompressions sequentially for correctness.
    """

    def __init__(self, block: nn.Module, tt_device: Any) -> None:
        super().__init__()
        self.block = block
        self._tt_device = tt_device
        self._df11_linears: List["DF11Linear"] = []
        self._decompressed_weights: Dict[int, Any] = {}

        from .df11_linear import DF11Linear
        for _name, module in block.named_modules():
            if isinstance(module, DF11Linear):
                self._df11_linears.append(module)

    def _decompress_all(self) -> None:
        """Decompress all DF11Linear weights used by this block."""
        try:
            from dfloat11_tt_cpp import dfloat11_decompress as _cpp_decompress
        except ImportError as e:
            raise RuntimeError(f"dfloat11_tt_cpp import failed: {e}")

        for layer in self._df11_linears:
            if layer._enc_exp is None:
                continue
            weight_tt = _cpp_decompress(
                layer._enc_exp,
                layer._sign_mant,
                layer._luts,
                layer._gaps,
                layer._outpos,
                layer._elem_starts,
                layer._elem_counts,
                layer._bit_starts,
                layer._elem_starts_host.tolist(),
                layer._elem_counts_host.tolist(),
                layer._bit_starts_host.tolist(),
                layer._k,
                layer._n,
                layer._T,
                layer._B,
                layer.out_features,
                layer.in_features,
                layer._R_pad,
                layer._C_pad,
                layer._n_elements,
                layer._n_bytes,
            )
            self._decompressed_weights[id(layer)] = weight_tt

    def _free_all(self) -> None:
        """Free all temporarily materialized weights."""
        for w in self._decompressed_weights.values():
            w.deallocate(force=True)
        self._decompressed_weights.clear()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Decompress all weights, run block forward, free weights."""
        self._decompress_all()

        saved = {}
        for layer in self._df11_linears:
            w_id = id(layer)
            if w_id in self._decompressed_weights:
                saved[w_id] = self._decompressed_weights[w_id]
                layer._decompressed_weight = saved[w_id]

        try:
            result = self.block(*args, **kwargs)
        finally:
            for layer in self._df11_linears:
                if hasattr(layer, "_decompressed_weight"):
                    del layer._decompressed_weight
            self._free_all()

        return result