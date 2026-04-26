"""DF11Embedding: nn.Embedding backed by a DFloat11-TT compressed table."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class DF11Embedding(nn.Module):
    """Embedding table stored compressed and decompressed through the TT device."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        device: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self._k: int = 0
        self._n: int = 0
        self._T: int = 0
        self._B: int = 0
        self._R_pad: int = 0
        self._C_pad: int = 0
        self._n_elements: int = 0
        self._n_bytes: int = 0

        self._enc_exp = None
        self._sign_mant = None
        self._luts = None
        self._gaps = None
        self._outpos = None

        self._tt_device = device
        self._module_name = "<unnamed>"
        self._cached_weight_torch: Optional[torch.Tensor] = None

    def load_bundle(self, bundle: Dict, tt_device: Any) -> None:
        import ttnn

        self._k = int(bundle["k"])
        self._n = int(bundle["n"])
        self._T = int(bundle["T"])
        self._B = int(bundle["B"])
        self._R_pad = int(bundle["R_pad"])
        self._C_pad = int(bundle["C_pad"])
        self._n_elements = int(bundle["n_elements"])
        self._n_bytes = int(bundle["n_bytes"])
        self._tt_device = tt_device

        def _to_ttnn_uint8(arr: np.ndarray) -> ttnn.Tensor:
            t = torch.from_numpy(arr.flatten().astype(np.uint8))
            return ttnn.from_torch(
                t,
                dtype=ttnn.uint8,
                device=tt_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        self._enc_exp = _to_ttnn_uint8(bundle["encoded_exponent"])
        self._sign_mant = _to_ttnn_uint8(bundle["sign_mantissa"])
        self._luts = _to_ttnn_uint8(bundle["luts"])
        self._gaps = _to_ttnn_uint8(bundle["gaps"])
        self._outpos = _to_ttnn_uint8(bundle["output_positions"].view(np.uint8))

    def clear_weight_cache(self) -> None:
        self._cached_weight_torch = None

    def _decompress_weight_to_torch(self) -> torch.Tensor:
        import ttnn

        try:
            from dfloat11_tt_cpp import dfloat11_decompress as _cpp_decompress
        except ImportError:
            raise RuntimeError(
                "dfloat11_tt_cpp C++ extension not built. Run 'make build' first."
            )

        weight_tt = _cpp_decompress(
            self._enc_exp,
            self._sign_mant,
            self._luts,
            self._gaps,
            self._outpos,
            self._k,
            self._n,
            self._T,
            self._B,
            self.num_embeddings,
            self.embedding_dim,
            self._R_pad,
            self._C_pad,
            self._n_elements,
            self._n_bytes,
        )
        weight = ttnn.to_torch(weight_tt)
        weight_tt.deallocate(force=True)
        return weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        trace = _env_flag("DFLOAT11_TRACE_LINEAR", False)
        cache_weights = _env_flag("DFLOAT11_CACHE_WEIGHTS", True)
        if trace:
            t0 = time.perf_counter()
            cache_state = "cached" if self._cached_weight_torch is not None else "cold"
            logger.info(
                f"[df11] {self._module_name}: embedding start "
                f"shape=[{self.num_embeddings},{self.embedding_dim}] k={self._k} "
                f"cache={cache_state if cache_weights else 'off'}"
            )

        if cache_weights and self._cached_weight_torch is not None:
            weight = self._cached_weight_torch
        else:
            weight = self._decompress_weight_to_torch()
            if cache_weights:
                self._cached_weight_torch = weight

        out = F.embedding(input_ids, weight, padding_idx=self.padding_idx)
        if trace:
            logger.info(
                f"[df11] {self._module_name}: embedding done in {time.perf_counter() - t0:.2f}s"
            )
        return out

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, df11_k={self._k}, df11_B={self._B}"
        )
