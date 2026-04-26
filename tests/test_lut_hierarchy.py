"""Unit tests for LUT hierarchy construction and correctness."""
from __future__ import annotations

import numpy as np
import pytest

from dfloat11_tt.compress.compressor import (
    _build_codec_with_max_len, build_luts
)


def test_luts_shape() -> None:
    """LUT array must have shape (k+1, 256) with k ≥ 1."""
    counter = {i: max(1, 100 - i * 2) for i in range(50)}
    table, _ = _build_codec_with_max_len(counter)
    luts = build_luts(table)
    assert luts.ndim == 2
    assert luts.shape[1] == 256
    assert luts.shape[0] >= 2  # at least 1 decode LUT + code-lengths row


def test_luts_code_length_row() -> None:
    """Last row of LUTs must store correct code lengths for all symbols."""
    counter = {0: 100, 1: 50, 2: 25, 3: 12, 4: 6}
    table, _ = _build_codec_with_max_len(counter)
    luts = build_luts(table)
    k = luts.shape[0] - 1  # last row = code lengths
    for sym, (length, _) in table.items():
        if isinstance(sym, int):
            assert int(luts[k, sym]) == length, (
                f"Symbol {sym}: expected length {length}, got {int(luts[k, sym])}"
            )


def test_lut_lookup_correctness() -> None:
    """Manual LUT lookup must decode every symbol correctly."""
    counter = {i: max(1, 200 - i * 3) for i in range(80)}
    table, _ = _build_codec_with_max_len(counter)
    luts = build_luts(table)
    k = luts.shape[0] - 1

    for sym, (length, code) in table.items():
        if not isinstance(sym, int):
            continue
        # Build a 64-bit MSB-aligned bit buffer with this code
        long_buffer = code << (64 - length)
        # Walk LUT chain
        top = (long_buffer >> 56) & 0xFF
        decoded = int(luts[0, top])
        lut_idx = 1
        while decoded >= 240 and lut_idx <= k:
            nx = 256 - decoded
            nb = (long_buffer >> (56 - lut_idx * 8)) & 0xFF
            decoded = int(luts[nx, nb])
            lut_idx += 1
        assert decoded == sym, f"LUT decoded {decoded}, expected {sym} (code={bin(code)}, len={length})"


def test_no_lut_collision() -> None:
    """Building LUTs must not raise ValueError (no hash collision)."""
    import numpy as np
    np.random.seed(42)
    counts = np.random.randint(1, 500, 256).tolist()
    counter = {i: c for i, c in enumerate(counts)}
    table, _ = _build_codec_with_max_len(counter)
    # Should not raise
    luts = build_luts(table)
    assert luts is not None
