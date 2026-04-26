"""Binary format reader/writer for DFloat11-TT bundles."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional, Union

import numpy as np
import torch

MAGIC = b"DF11TT01"
DTYPE_BF16 = 0


def _pack_header(bundle: Dict) -> bytes:
    shape = bundle["shape"]
    ndim = len(shape)
    header = struct.pack(
        f"<8sBB{ndim}QBBII IIQ Q",
        MAGIC,
        bundle["dtype"],
        ndim,
        *shape,
        bundle["k"],
        bundle["n"],
        bundle["T"],
        bundle["B"],
        bundle["R_pad"],
        bundle["C_pad"],
        bundle["n_elements"],
        bundle["n_bytes"],
    )
    return header


def _header_size(ndim: int) -> int:
    # 8(magic) + 1(dtype) + 1(ndim) + 8*ndim(shape) + 1(k) + 1(n) + 4(T) + 4(B)
    # + 4(R_pad) + 4(C_pad) + 8(n_elements) + 8(n_bytes)
    return 8 + 1 + 1 + 8 * ndim + 1 + 1 + 4 + 4 + 4 + 4 + 8 + 8


def write_bundle(f: BinaryIO, bundle: Dict) -> None:
    """Write one compressed tensor bundle to a binary file object."""
    f.write(_pack_header(bundle))
    luts: np.ndarray = bundle["luts"]           # shape (k+1, 256)
    f.write(luts.astype(np.uint8).tobytes())
    f.write(bundle["encoded_exponent"].astype(np.uint8).tobytes())
    f.write(bundle["sign_mantissa"].astype(np.uint8).tobytes())
    f.write(bundle["gaps"].astype(np.uint8).tobytes())
    f.write(bundle["output_positions"].astype(np.uint32).tobytes())


def read_bundle(f: BinaryIO) -> Optional[Dict]:
    """Read one compressed tensor bundle from a binary file object.

    Returns None at EOF.
    """
    magic = f.read(8)
    if not magic:
        return None
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")

    dtype, ndim = struct.unpack("<BB", f.read(2))
    shape = list(struct.unpack(f"<{ndim}Q", f.read(8 * ndim)))
    k, n = struct.unpack("<BB", f.read(2))
    T, B, R_pad, C_pad = struct.unpack("<IIII", f.read(16))
    n_elements, n_bytes = struct.unpack("<QQ", f.read(16))

    luts = np.frombuffer(f.read((k + 1) * 256), dtype=np.uint8).reshape(k + 1, 256)
    encoded_exponent = np.frombuffer(f.read(n_bytes), dtype=np.uint8)
    sign_mantissa = np.frombuffer(f.read(n_elements), dtype=np.uint8)

    gap_bytes = (T * B * 5 + 7) // 8
    gaps = np.frombuffer(f.read(gap_bytes), dtype=np.uint8)
    output_positions = np.frombuffer(f.read((B + 1) * 4), dtype=np.uint32)

    return {
        "dtype": dtype,
        "shape": shape,
        "k": k,
        "n": n,
        "T": T,
        "B": B,
        "R_pad": R_pad,
        "C_pad": C_pad,
        "n_elements": n_elements,
        "n_bytes": n_bytes,
        "luts": luts,
        "encoded_exponent": encoded_exponent,
        "sign_mantissa": sign_mantissa,
        "gaps": gaps,
        "output_positions": output_positions,
    }


def iter_bundles(f: BinaryIO) -> Iterator[Dict]:
    """Iterate over all bundles in a multi-bundle file."""
    while True:
        b = read_bundle(f)
        if b is None:
            return
        yield b


def save_model_bundle(
    tensor_dict: Dict[str, Dict],
    path: Union[str, Path],
) -> None:
    """Save a dict of {tensor_name: bundle} to a single file.

    Format: for each tensor, write 4-byte name-length, name bytes, then the bundle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        # File header: number of tensors
        f.write(struct.pack("<I", len(tensor_dict)))
        for name, bundle in tensor_dict.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            write_bundle(f, bundle)


def load_model_bundle(path: Union[str, Path]) -> Dict[str, Dict]:
    """Load a model bundle file saved by save_model_bundle."""
    path = Path(path)
    result: Dict[str, Dict] = {}
    with open(path, "rb") as f:
        (n_tensors,) = struct.unpack("<I", f.read(4))
        for _ in range(n_tensors):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("utf-8")
            bundle = read_bundle(f)
            assert bundle is not None
            result[name] = bundle
    return result
