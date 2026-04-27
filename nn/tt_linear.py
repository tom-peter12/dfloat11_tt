"""TTLinear: uncompressed BF16 reference linear using the same TTNN path as DF11."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from loguru import logger


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class TTLinear(nn.Module):
    """nn.Linear equivalent backed by uncompressed BF16 weights on a TT device."""

    def __init__(
        self,
        source: nn.Linear,
        device: Any,
        module_name: str = "<unnamed>",
    ) -> None:
        super().__init__()
        import ttnn

        self.in_features = source.in_features
        self.out_features = source.out_features
        self._tt_device = device
        self._module_name = module_name

        weight_tt = ttnn.from_torch(
            source.weight.detach().bfloat16(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        weight_tiled_tt = ttnn.to_layout(weight_tt, ttnn.TILE_LAYOUT)
        weight_tt.deallocate(force=True)
        self._weight_linear_tt = ttnn.transpose(weight_tiled_tt, 0, 1)
        weight_tiled_tt.deallocate(force=True)

        self.bias = None
        if source.bias is not None:
            self.bias = ttnn.from_torch(
                source.bias.detach().bfloat16(),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

    def clear_weight_cache(self) -> None:
        try:
            self._weight_linear_tt.deallocate(force=True)
        except Exception:
            pass
        if self.bias is not None:
            try:
                self.bias.deallocate(force=True)
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import ttnn

        trace = _env_flag("DFLOAT11_TRACE_LINEAR", False)
        t0 = time.perf_counter() if trace else 0.0

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

        out_tt = ttnn.linear(x_tt, self._weight_linear_tt, bias=self.bias)

        if input_is_torch:
            out = ttnn.to_torch(out_tt)
            try:
                x_tt.deallocate(force=True)
                out_tt.deallocate(force=True)
            except Exception:
                pass
            if trace:
                logger.info(
                    f"[ttref] {self._module_name}: total={(time.perf_counter() - t0) * 1000.0:.2f}ms"
                )
            return out

        if trace:
            logger.info(
                f"[ttref] {self._module_name}: total={(time.perf_counter() - t0) * 1000.0:.2f}ms"
            )
        return out_tt

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
