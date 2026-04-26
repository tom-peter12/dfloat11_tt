"""Unit tests for Huffman tree construction and code assignment."""
from __future__ import annotations

import pytest
from dfloat11_tt.compress.compressor import (
    build_huffman_tree, _assign_codes, _build_codec_with_max_len
)


def test_basic_huffman_codes() -> None:
    """Simple 4-symbol case: codes must be prefix-free and lengths correct."""
    counter = {0: 10, 1: 5, 2: 3, 3: 1}
    tree = build_huffman_tree(counter)
    table = _assign_codes(tree)
    assert set(table.keys()) == {0, 1, 2, 3}
    codes = {s: (length, code) for s, (length, code) in table.items()}
    # Most frequent symbol gets shortest code
    assert codes[0][0] <= codes[1][0] <= codes[2][0]
    # Prefix-free check: no code is a prefix of another
    bit_strings = {s: bin(c)[2:].rjust(l, '0') for s, (l, c) in table.items()}
    for s1, bs1 in bit_strings.items():
        for s2, bs2 in bit_strings.items():
            if s1 != s2:
                assert not bs2.startswith(bs1), f"Code {s1}={bs1!r} is prefix of {s2}={bs2!r}"


def test_max_len_enforcement() -> None:
    """Codec with extreme skew must enforce max code length ≤ 32."""
    # Create highly skewed distribution that would produce very long codes
    counter = {i: 2**i for i in range(40)}
    table, _ = _build_codec_with_max_len(counter)
    max_len = max(l for l, _ in table.values())
    assert max_len <= 32, f"Max code length {max_len} exceeds 32"


def test_deterministic_tie_breaking() -> None:
    """Same input always produces exactly the same code table (determinism check)."""
    counter = {i: 100 for i in range(16)}  # uniform — maximum tie-breaking
    t1 = _assign_codes(build_huffman_tree(counter))
    t2 = _assign_codes(build_huffman_tree(counter))
    assert t1 == t2, "Huffman construction is not deterministic"


def test_single_symbol() -> None:
    """Single-symbol alphabet: code length must be 1."""
    counter = {42: 999}
    tree = build_huffman_tree(counter)
    table = _assign_codes(tree)
    assert 42 in table
    assert table[42][0] >= 1  # at least 1 bit


def test_all_256_symbols() -> None:
    """All 256 possible exponent values: should produce a valid code table."""
    import numpy as np
    np.random.seed(0)
    counts = np.random.randint(1, 1000, 256).tolist()
    counter = {i: c for i, c in enumerate(counts)}
    table, _ = _build_codec_with_max_len(counter)
    assert len(table) == 256
    max_len = max(l for l, _ in table.values())
    assert max_len <= 32
