"""Smoke tests for the Tenstorrent kernel (requires real Blackhole device).

These tests are skipped automatically if no TT device is available.
They test a tiny synthetic input to verify the kernel executes without crashing
and produces bit-identical output.

Run with:
    TT_METAL_VISIBLE_DEVICES=0 \
    TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    pytest tests/test_kernel_smoke.py -v
"""
from __future__ import annotations

import os
import numpy as np
import pytest
import torch

from dfloat11_tt.nn._df11_split import DEFAULT_MAX_CORES, compute_core_ranges

# Physical device ID that works on P150_X4 with VISIBLE_DEVICES=0 + p150 descriptor
_TT_DEVICE_ID = int(os.environ.get("DFLOAT11_TT_DEVICE_ID", "3"))


def _has_tt_device() -> bool:
    try:
        import ttnn
        d = ttnn.CreateDevice(device_id=int(os.environ["DFLOAT11_TT_DEVICE_ID"]))
        ttnn.CloseDevice(d)
        return True
    except Exception:
        return False


# Mark all tests in this module as requiring a TT device
pytestmark = pytest.mark.skipif(
    not _has_tt_device(),
    reason="No Tenstorrent device available"
)


@pytest.fixture(scope="module")
def tt_device():
    import ttnn
    dev = ttnn.CreateDevice(device_id=_TT_DEVICE_ID)
    yield dev
    ttnn.CloseDevice(dev)


def test_smoke_tiny(tt_device) -> None:
    """64×64 weight: device decompression produces bit-identical output."""
    try:
        from dfloat11_tt_cpp import dfloat11_decompress
    except ImportError:
        pytest.skip("dfloat11_tt_cpp extension not built")

    import ttnn
    from dfloat11_tt.compress.compressor import compress_tensor

    torch.manual_seed(0)
    w = torch.randn(64, 64, dtype=torch.bfloat16)
    bundle = compress_tensor(w)

    def _to_ttnn_uint8(arr):
        return ttnn.from_torch(
            torch.from_numpy(arr.flatten().astype(np.uint8)),
            dtype=ttnn.uint8, device=tt_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def _to_ttnn_uint32(arr):
        return ttnn.from_torch(
            torch.from_numpy(arr.flatten().astype(np.uint32)),
            dtype=ttnn.uint32, device=tt_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    enc_tt   = _to_ttnn_uint8(bundle["encoded_exponent"])
    sm_tt    = _to_ttnn_uint8(bundle["sign_mantissa"])
    luts_tt  = _to_ttnn_uint8(bundle["luts"])
    gaps_tt  = _to_ttnn_uint8(bundle["gaps"])
    outpos_tt = _to_ttnn_uint8(bundle["output_positions"].view(np.uint8))
    elem_starts, elem_counts, bit_starts = compute_core_ranges(
        bundle, bundle["R_pad"], bundle["C_pad"], max_cores=DEFAULT_MAX_CORES
    )

    out_tt = dfloat11_decompress(
        enc_tt, sm_tt, luts_tt, gaps_tt, outpos_tt,
        _to_ttnn_uint32(elem_starts),
        _to_ttnn_uint32(elem_counts),
        _to_ttnn_uint32(bit_starts),
        elem_starts.tolist(),
        elem_counts.tolist(),
        bit_starts.tolist(),
        bundle["k"], bundle["n"], bundle["T"], bundle["B"],
        64, 64, bundle["R_pad"], bundle["C_pad"],
        bundle["n_elements"], bundle["n_bytes"],
    )

    out_host = ttnn.to_torch(out_tt).to(torch.bfloat16)
    # Trim to original shape (strip padding)
    out_host = out_host[:64, :64]

    assert torch.equal(w.view(torch.uint16), out_host.view(torch.uint16)), \
        "Bit-identity FAILED on device (tiny 64×64)"

    out_tt.deallocate(force=True)


def test_smoke_medium(tt_device) -> None:
    """1024×1024 weight: device decompression bit-identical."""
    try:
        from dfloat11_tt_cpp import dfloat11_decompress
    except ImportError:
        pytest.skip("dfloat11_tt_cpp extension not built")

    import ttnn
    from dfloat11_tt.compress.compressor import compress_tensor

    torch.manual_seed(1)
    w = torch.randn(1024, 1024, dtype=torch.bfloat16) * 0.02

    bundle = compress_tensor(w)

    def _to_ttnn_uint8(arr):
        return ttnn.from_torch(
            torch.from_numpy(arr.flatten().astype(np.uint8)),
            dtype=ttnn.uint8, device=tt_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    def _to_ttnn_uint32(arr):
        return ttnn.from_torch(
            torch.from_numpy(arr.flatten().astype(np.uint32)),
            dtype=ttnn.uint32, device=tt_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    enc_tt    = _to_ttnn_uint8(bundle["encoded_exponent"])
    sm_tt     = _to_ttnn_uint8(bundle["sign_mantissa"])
    luts_tt   = _to_ttnn_uint8(bundle["luts"])
    gaps_tt   = _to_ttnn_uint8(bundle["gaps"])
    outpos_tt = _to_ttnn_uint8(bundle["output_positions"].view(np.uint8))
    elem_starts, elem_counts, bit_starts = compute_core_ranges(
        bundle, bundle["R_pad"], bundle["C_pad"], max_cores=DEFAULT_MAX_CORES
    )

    out_tt = dfloat11_decompress(
        enc_tt, sm_tt, luts_tt, gaps_tt, outpos_tt,
        _to_ttnn_uint32(elem_starts),
        _to_ttnn_uint32(elem_counts),
        _to_ttnn_uint32(bit_starts),
        elem_starts.tolist(),
        elem_counts.tolist(),
        bit_starts.tolist(),
        bundle["k"], bundle["n"], bundle["T"], bundle["B"],
        1024, 1024, bundle["R_pad"], bundle["C_pad"],
        bundle["n_elements"], bundle["n_bytes"],
    )

    out_host = ttnn.to_torch(out_tt)[:1024, :1024].to(torch.bfloat16)
    assert torch.equal(w.view(torch.uint16), out_host.view(torch.uint16)), \
        "Bit-identity FAILED on device (1024×1024)"
    out_tt.deallocate(force=True)
