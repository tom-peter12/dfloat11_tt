"""DF11Linear: a drop-in replacement for nn.Linear with DFloat11-TT weights."""
from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from loguru import logger


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class DF11Linear(nn.Module):
    """Linear layer that decompresses DFloat11-TT weights on-the-fly.

    The weight is stored in compressed form as a set of device tensors.
    In forward(), it is decompressed to BF16, the matmul is executed, and
    the materialized weight is either freed or cached for token-generation
    reuse, depending on DFLOAT11_CACHE_WEIGHTS.

    Bias (if any) is stored uncompressed as a regular float buffer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Compressed bundle metadata (scalars)
        self._k:          int = 0
        self._n:          int = 0
        self._T:          int = 0
        self._B:          int = 0
        self._R_pad:      int = 0
        self._C_pad:      int = 0
        self._n_elements: int = 0
        self._n_bytes:    int = 0

        # These are TTNN device tensors, not torch.Tensor buffers.  Keep them as
        # plain attributes so PyTorch does not try to place them in state_dict().
        self._enc_exp = None
        self._sign_mant = None
        self._luts = None
        self._gaps = None
        self._outpos = None
        self.bias = None
        self._has_bias = bias

        self._tt_device = device  # Tenstorrent device handle (not a torch device)
        self._module_name = "<unnamed>"
        self._cached_weight_linear_tt = None

    def load_bundle(self, bundle: Dict, tt_device: Any) -> None:
        """Load a compressed bundle dict (from compress.load_model_bundle) onto TT device."""
        import ttnn

        self._k          = int(bundle["k"])
        self._n          = int(bundle["n"])
        self._T          = int(bundle["T"])
        self._B          = int(bundle["B"])
        self._R_pad      = int(bundle["R_pad"])
        self._C_pad      = int(bundle["C_pad"])
        self._n_elements = int(bundle["n_elements"])
        self._n_bytes    = int(bundle["n_bytes"])
        self._tt_device  = tt_device

        def _to_ttnn_uint8(arr: np.ndarray) -> ttnn.Tensor:
            t = torch.from_numpy(arr.flatten().astype(np.uint8))
            return ttnn.from_torch(t, dtype=ttnn.uint8, device=tt_device,
                                   layout=ttnn.ROW_MAJOR_LAYOUT)

        self._enc_exp   = _to_ttnn_uint8(bundle["encoded_exponent"])
        self._sign_mant = _to_ttnn_uint8(bundle["sign_mantissa"])
        self._luts      = _to_ttnn_uint8(bundle["luts"])
        self._gaps      = _to_ttnn_uint8(bundle["gaps"])
        self._outpos    = _to_ttnn_uint8(bundle["output_positions"].view(np.uint8))

    def clear_weight_cache(self) -> None:
        """Release the cached decompressed/transposed TT weight, if present."""
        if self._cached_weight_linear_tt is not None:
            try:
                self._cached_weight_linear_tt.deallocate(force=True)
            except Exception:
                pass
            self._cached_weight_linear_tt = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompress or reuse weights, then run matmul."""
        import ttnn
        try:
            from dfloat11_tt_cpp import dfloat11_decompress as _cpp_decompress
        except ImportError:
            raise RuntimeError(
                "dfloat11_tt_cpp C++ extension not built. Run 'make build' first."
            )

        trace = _env_flag("DFLOAT11_TRACE_LINEAR", False)
        cache_weights = _env_flag("DFLOAT11_CACHE_WEIGHTS", True)
        if trace:
            t0 = time.perf_counter()
            cache_state = "cached" if self._cached_weight_linear_tt is not None else "cold"
            logger.info(
                f"[df11] {self._module_name}: start "
                f"shape=[{self.out_features},{self.in_features}] k={self._k} "
                f"cache={cache_state if cache_weights else 'off'}"
            )

        if cache_weights and self._cached_weight_linear_tt is not None:
            weight_linear_tt = self._cached_weight_linear_tt
        else:
            # Decompress weight on-device, convert to tile layout, and transpose
            # once into the right orientation for ttnn.linear.
            weight_tt = _cpp_decompress(
                self._enc_exp, self._sign_mant, self._luts, self._gaps, self._outpos,
                self._k, self._n, self._T, self._B,
                self.out_features, self.in_features,
                self._R_pad, self._C_pad,
                self._n_elements, self._n_bytes,
            )
            weight_tiled_tt = ttnn.to_layout(weight_tt, ttnn.TILE_LAYOUT)
            weight_linear_tt = ttnn.transpose(weight_tiled_tt, 0, 1)
            weight_tiled_tt.deallocate(force=True)
            weight_tt.deallocate(force=True)

            if cache_weights:
                self._cached_weight_linear_tt = weight_linear_tt

        # Convert input to ttnn if needed
        input_is_torch = isinstance(x, torch.Tensor)
        if input_is_torch:
            x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=self._tt_device,
                                   layout=ttnn.TILE_LAYOUT)
        else:
            x_tt = x

        # ttnn.linear(a, b) computes a @ b.  PyTorch Linear stores weight as
        # [out_features, in_features], so transpose before passing it to TTNN.
        out_tt = ttnn.linear(x_tt, weight_linear_tt, bias=self.bias)

        if not cache_weights:
            weight_linear_tt.deallocate(force=True)

        # Return as torch tensor if input was torch
        if input_is_torch:
            out = ttnn.to_torch(out_tt)
            try:
                x_tt.deallocate(force=True)
                out_tt.deallocate(force=True)
            except Exception:
                pass
            if trace:
                logger.info(f"[df11] {self._module_name}: done in {time.perf_counter() - t0:.2f}s")
            return out
        if trace:
            logger.info(f"[df11] {self._module_name}: done in {time.perf_counter() - t0:.2f}s")
        return out_tt

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, df11_k={self._k}, df11_B={self._B}"
        )
