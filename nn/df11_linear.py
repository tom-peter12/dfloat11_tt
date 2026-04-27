"""DF11Linear: a drop-in replacement for nn.Linear with DFloat11-TT weights."""
from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from ._df11_split import compute_core_ranges, DEFAULT_MAX_CORES


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class DF11Linear(nn.Module):
    """Linear layer that decompresses DFloat11-TT weights on-the-fly."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

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

        self._outpos_host: Optional[np.ndarray] = None
        self._elem_starts = None
        self._elem_counts = None
        self._bit_starts = None
        self._elem_starts_host: Optional[np.ndarray] = None
        self._elem_counts_host: Optional[np.ndarray] = None
        self._bit_starts_host: Optional[np.ndarray] = None

        self.bias = None
        self._has_bias = bias

        self._tt_device = device
        self._module_name = "<unnamed>"
        self._cached_weight_linear_tt = None
        self._block_weight_linear_tt = None

    def load_bundle(self, bundle: Dict, tt_device: Any) -> None:
        """Load a compressed bundle dict onto the TT device."""
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

        def _to_ttnn_uint32(arr: np.ndarray) -> ttnn.Tensor:
            t = torch.from_numpy(arr.flatten().astype(np.uint32))
            return ttnn.from_torch(
                t,
                dtype=ttnn.uint32,
                device=tt_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        self._enc_exp = _to_ttnn_uint8(bundle["encoded_exponent"])
        self._sign_mant = _to_ttnn_uint8(bundle["sign_mantissa"])
        self._luts = _to_ttnn_uint8(bundle["luts"])
        self._gaps = _to_ttnn_uint8(bundle["gaps"])

        self._outpos_host = np.array(bundle["output_positions"], dtype=np.uint32, copy=True)
        self._outpos = _to_ttnn_uint8(self._outpos_host.view(np.uint8))

        # Compute page-aligned per-core ranges with correct bit_starts.
        # See nn/_df11_split.py for the rationale.
        self._elem_starts_host, self._elem_counts_host, self._bit_starts_host = (
            compute_core_ranges(
                bundle, self._R_pad, self._C_pad, max_cores=DEFAULT_MAX_CORES
            )
        )

        self._elem_starts = _to_ttnn_uint32(self._elem_starts_host)
        self._elem_counts = _to_ttnn_uint32(self._elem_counts_host)
        self._bit_starts = _to_ttnn_uint32(self._bit_starts_host)

    def clear_weight_cache(self) -> None:
        """Release the cached decompressed/transposed TT weight, if present."""
        if self._cached_weight_linear_tt is not None:
            try:
                self._cached_weight_linear_tt.deallocate(force=True)
            except Exception:
                pass
            self._cached_weight_linear_tt = None

    def materialize_weight_linear_tt(self, trace: bool = False) -> tuple[Any, float, float]:
        """Decompress this layer's DF11 weight and return matmul-ready TT weight."""
        import ttnn

        try:
            from dfloat11_tt_cpp import dfloat11_decompress as _cpp_decompress
        except ImportError as e:
            raise RuntimeError(f"dfloat11_tt_cpp import failed: {e}")

        td0 = time.perf_counter() if trace else 0.0
        weight_tt = _cpp_decompress(
            self._enc_exp,
            self._sign_mant,
            self._luts,
            self._gaps,
            self._outpos,
            self._elem_starts,
            self._elem_counts,
            self._bit_starts,
            self._elem_starts_host.tolist(),
            self._elem_counts_host.tolist(),
            self._bit_starts_host.tolist(),
            self._k,
            self._n,
            self._T,
            self._B,
            self.out_features,
            self.in_features,
            self._R_pad,
            self._C_pad,
            self._n_elements,
            self._n_bytes,
        )
        decompress_ms = (time.perf_counter() - td0) * 1000.0 if trace else 0.0

        tl0 = time.perf_counter() if trace else 0.0
        weight_tiled_tt = ttnn.to_layout(weight_tt, ttnn.TILE_LAYOUT)
        weight_tt.deallocate(force=True)
        weight_linear_tt = ttnn.transpose(weight_tiled_tt, 0, 1)
        weight_tiled_tt.deallocate(force=True)
        layout_ms = (time.perf_counter() - tl0) * 1000.0 if trace else 0.0
        return weight_linear_tt, decompress_ms, layout_ms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompress or reuse weights, then run matmul."""
        import ttnn

        trace = _env_flag("DFLOAT11_TRACE_LINEAR", False)
        cache_weights = _env_flag("DFLOAT11_CACHE_WEIGHTS", True)

        t0 = time.perf_counter() if trace else 0.0
        decompress_ms = 0.0
        layout_ms = 0.0
        owns_weight = False

        if self._block_weight_linear_tt is not None:
            weight_linear_tt = self._block_weight_linear_tt
            if trace:
                logger.info(
                    f"[df11] {self._module_name}: block predecoded weight "
                    f"shape=[{self.out_features},{self.in_features}]"
                )
        elif cache_weights and self._cached_weight_linear_tt is not None:
            weight_linear_tt = self._cached_weight_linear_tt
            if trace:
                logger.info(
                    f"[df11] {self._module_name}: cache hit "
                    f"shape=[{self.out_features},{self.in_features}]"
                )
        else:
            weight_linear_tt, decompress_ms, layout_ms = self.materialize_weight_linear_tt(trace=trace)

            if cache_weights:
                self._cached_weight_linear_tt = weight_linear_tt
            else:
                owns_weight = True

            if trace:
                logger.info(
                    f"[df11] {self._module_name}: cold decode "
                    f"shape=[{self.out_features},{self.in_features}] k={self._k} "
                    f"decompress={decompress_ms:.2f}ms layout={layout_ms:.2f}ms"
                )

        input_is_torch = isinstance(x, torch.Tensor)
        if input_is_torch:
            x_tt = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                device=self._tt_device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            x_tt = x

        tm0 = time.perf_counter() if trace else 0.0
        out_tt = ttnn.linear(x_tt, weight_linear_tt, bias=self.bias)
        matmul_ms = (time.perf_counter() - tm0) * 1000.0 if trace else 0.0

        if owns_weight:
            weight_linear_tt.deallocate(force=True)

        if input_is_torch:
            out = ttnn.to_torch(out_tt)
            try:
                x_tt.deallocate(force=True)
                out_tt.deallocate(force=True)
            except Exception:
                pass
            if trace:
                total_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(
                    f"[df11] {self._module_name}: total={total_ms:.2f}ms "
                    f"matmul={matmul_ms:.2f}ms"
                )
            return out

        if trace:
            total_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                f"[df11] {self._module_name}: total={total_ms:.2f}ms "
                f"matmul={matmul_ms:.2f}ms"
            )
        return out_tt

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, df11_k={self._k}, df11_B={self._B}"
        )
