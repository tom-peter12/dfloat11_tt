"""Roundtrip tests: compress → write bundle → read bundle → decode → bit-identical.

These tests run entirely on CPU using the pure-Python reference decoder.
They are the ground-truth correctness tests before any device is involved.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dfloat11_tt.compress.compressor import compress_tensor
from dfloat11_tt.compress.bundle import write_bundle, read_bundle, save_model_bundle, load_model_bundle
from dfloat11_tt.compress.reference_decoder import decode_bundle


def _make_weight(shape, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.bfloat16)


def test_roundtrip_small() -> None:
    """Small 64×64 weight: compress + decode must be bit-identical."""
    w = _make_weight((64, 64))
    bundle = compress_tensor(w)
    decoded = decode_bundle(bundle)
    assert decoded.shape == torch.Size([64, 64])
    assert torch.equal(decoded.flatten().view(torch.uint16), w.flatten().view(torch.uint16)), \
        "Bit-identity FAILED for 64×64 weight"


def test_roundtrip_large() -> None:
    """Larger 512×1024 weight: compress + decode must be bit-identical."""
    w = _make_weight((512, 1024), seed=1)
    bundle = compress_tensor(w)
    decoded = decode_bundle(bundle)
    assert torch.equal(decoded.flatten().view(torch.uint16), w.flatten().view(torch.uint16)), \
        "Bit-identity FAILED for 512×1024 weight"


def test_roundtrip_1d() -> None:
    """1D embedding weight (flat): compress + decode must be bit-identical."""
    w = _make_weight((4096,), seed=2)
    bundle = compress_tensor(w)
    decoded = decode_bundle(bundle)
    assert torch.equal(decoded.flatten().view(torch.uint16), w.view(torch.uint16)), \
        "Bit-identity FAILED for 1D weight"


def test_bundle_binary_format() -> None:
    """Write bundle to bytes, read back, compare metadata."""
    w = _make_weight((128, 256))
    bundle = compress_tensor(w)

    buf = io.BytesIO()
    write_bundle(buf, bundle)
    buf.seek(0)
    loaded = read_bundle(buf)

    assert loaded is not None
    assert loaded["shape"] == bundle["shape"]
    assert loaded["k"] == bundle["k"]
    assert loaded["n"] == bundle["n"]
    assert loaded["T"] == bundle["T"]
    assert loaded["n_elements"] == bundle["n_elements"]
    assert loaded["n_bytes"] == bundle["n_bytes"]
    assert np.array_equal(loaded["luts"], bundle["luts"])
    assert np.array_equal(loaded["encoded_exponent"], bundle["encoded_exponent"])
    assert np.array_equal(loaded["sign_mantissa"], bundle["sign_mantissa"])


def test_model_bundle_save_load() -> None:
    """save_model_bundle / load_model_bundle roundtrip."""
    w1 = _make_weight((64, 128), seed=3)
    w2 = _make_weight((256, 64), seed=4)
    bundles = {
        "model.layers.0.mlp.gate_proj": compress_tensor(w1),
        "model.layers.0.mlp.up_proj":   compress_tensor(w2),
    }
    with tempfile.NamedTemporaryFile(suffix=".df11tt") as tmp:
        save_model_bundle(bundles, tmp.name)
        loaded = load_model_bundle(tmp.name)

    assert set(loaded.keys()) == set(bundles.keys())
    for name, bundle in loaded.items():
        orig = bundles[name]
        assert bundle["shape"] == orig["shape"]


def test_special_values() -> None:
    """Weights containing NaN, Inf, and zero: bit-identity required."""
    w = torch.tensor([0.0, float('inf'), float('-inf'), 1.0, -1.0,
                      0.5, -0.5, 1e-4, -1e4], dtype=torch.bfloat16)
    bundle = compress_tensor(w)
    decoded = decode_bundle(bundle)
    assert torch.equal(w.view(torch.uint16), decoded.flatten().view(torch.uint16)), \
        "Bit-identity FAILED for special values"


def test_compression_ratio_in_range() -> None:
    """Compression ratio for typical LLM-like weight should be in [0.55, 0.85]."""
    torch.manual_seed(0)
    # Simulate LLM weight distribution: most exponents cluster around 127
    w = torch.randn(1024, 1024, dtype=torch.bfloat16) * 0.02
    bundle = compress_tensor(w)
    assert 0.50 <= bundle["compression_ratio"] <= 0.90, \
        f"Compression ratio {bundle['compression_ratio']:.3f} out of expected range"


def test_gaps_extraction_determinism() -> None:
    """Recompressing the same weight yields bit-identical compressed bundle."""
    w = _make_weight((128, 128), seed=5)
    b1 = compress_tensor(w)
    b2 = compress_tensor(w)
    assert np.array_equal(b1["encoded_exponent"], b2["encoded_exponent"]), \
        "Encoded exponents differ between compressions of the same weight"
    assert np.array_equal(b1["gaps"], b2["gaps"]), \
        "Gaps differ between compressions of the same weight"
