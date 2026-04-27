"""Compute correct per-core ranges for the DF11 device decoder.

The reader kernel in ``kernels/reader_df11.cpp`` is a *purely sequential*
Huffman decoder: each core starts at ``core_bit_start`` (an absolute bit
offset into the encoded bitstream) and at element index ``core_elem_start``,
then decodes ``elem_count`` elements, emitting them into output pages
``[core_elem_start * 2 // page_size, ...)``.

Two correctness constraints must hold for the host-side split:

1. ``core_bit_start`` must be the *exact* cumulative-code-length bit position
   at which the first element of the core's range was emitted by the
   compressor. The encoded bitstream is fully contiguous (no per-thread or
   per-block padding — see ``compress/compressor.py::encode_exponents``);
   only the prefix sum of code lengths gives the right offset. The previous
   formula ``blk_start * T * n * 8`` assumed block storage boundaries are
   aligned with code starts, which they are not.

2. Cores may start/end in the middle of an output page. The kernel must write
   only the byte range owned by that core for the first/last page, otherwise
   adjacent cores can clobber each other's page fragments.

The default split is page-aligned. That keeps all device reads/writes on stable
row/page boundaries while still using many cores. Bit starts are computed only
for the page boundaries assigned to active cores, so model load does not need a
full Python walk of every symbol. ``DFLOAT11_SPLIT_MODE=block`` is kept as an
experimental/debug split for the partial-page path.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np


def _default_max_cores() -> int:
    raw = os.environ.get("DFLOAT11_MAX_CORES", "30")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


# Default to multicore; page splitting is the correctness-preserving default.
DEFAULT_MAX_CORES = _default_max_cores()
_PTR_MIN = 240


def _extract_gap(gaps: np.ndarray, thread_id: int) -> int:
    bit_pos = thread_id * 5
    byte_idx = bit_pos // 8
    if byte_idx >= len(gaps):
        return 0
    byte_low = int(gaps[byte_idx])
    byte_high = int(gaps[byte_idx + 1]) if byte_idx + 1 < len(gaps) else 0
    short_be = (byte_low << 8) | byte_high
    return (short_be >> (11 - (bit_pos % 8))) & 0x1F


def _walk_bit_positions(
    encoded: np.ndarray,
    luts: np.ndarray,
    k: int,
    n_elements: int,
    n_bytes: int,
    page_elements: int,
    n_pages: int,
) -> np.ndarray:
    """Decode the bitstream sequentially and record the bit position at every
    page boundary (element index == p * page_elements for p in 0..n_pages).

    Returns an array of length ``n_pages + 1`` where entry ``p`` is the bit
    position of the first encoded code for element ``p * page_elements``.
    Entry ``n_pages`` is the bit position immediately past the last element
    (== total encoded bit length).
    """
    page_bit_starts = np.zeros(n_pages + 1, dtype=np.uint64)

    # Local bindings for speed in the inner loop.
    lut0  = luts[0]
    lut_k = luts[k]
    luts_arr = luts  # for chained lookups

    bit_pos = 0
    elem_idx = 0
    next_page_elem = 0  # we'll record at elem_idx == 0 first
    next_page_idx = 0

    enc = encoded  # uint8 numpy array

    while elem_idx < n_elements:
        # Record at every page boundary we cross (handles the first page too).
        while elem_idx == next_page_elem and next_page_idx <= n_pages:
            page_bit_starts[next_page_idx] = bit_pos
            next_page_idx += 1
            if next_page_idx <= n_pages:
                next_page_elem = next_page_idx * page_elements
            else:
                next_page_elem = n_elements + 1  # disable

        byte_idx = bit_pos >> 3
        bit_gap  = bit_pos & 7

        # Build 64-bit big-endian buffer from 8 bytes starting at byte_idx,
        # padding with 0 past EOF — matches the kernel.
        end = byte_idx + 8
        if end <= n_bytes:
            chunk = enc[byte_idx:end]
            buf = (
                (int(chunk[0]) << 56) | (int(chunk[1]) << 48) |
                (int(chunk[2]) << 40) | (int(chunk[3]) << 32) |
                (int(chunk[4]) << 24) | (int(chunk[5]) << 16) |
                (int(chunk[6]) << 8)  |  int(chunk[7])
            )
        else:
            buf = 0
            for i in range(8):
                v = int(enc[byte_idx + i]) if (byte_idx + i) < n_bytes else 0
                buf = (buf << 8) | v
        if bit_gap:
            buf = (buf << bit_gap) & 0xFFFFFFFFFFFFFFFF

        # Hierarchical LUT lookup (mirrors lut_lookup in reader_df11.cpp).
        d = int(lut0[(buf >> 56) & 0xFF])
        if d >= _PTR_MIN and (256 - d) < k:
            nx = 256 - d
            d = int(luts_arr[nx, (buf >> 48) & 0xFF])
            if d >= _PTR_MIN and (256 - d) < k:
                nx2 = 256 - d
                d = int(luts_arr[nx2, (buf >> 40) & 0xFF])
                if d >= _PTR_MIN and (256 - d) < k:
                    nx3 = 256 - d
                    d = int(luts_arr[nx3, (buf >> 32) & 0xFF])
        code_len = int(lut_k[d])
        if code_len == 0:
            raise RuntimeError(
                f"DF11 host bit-position walk: code_len=0 at elem={elem_idx} "
                f"bit_pos={bit_pos}; bundle is malformed."
            )

        bit_pos += code_len
        elem_idx += 1

    # Record the final boundary (one past last element).
    while next_page_idx <= n_pages:
        page_bit_starts[next_page_idx] = bit_pos
        next_page_idx += 1

    return page_bit_starts


def _lookup_code_len(encoded: np.ndarray, luts: np.ndarray, k: int, n_bytes: int, bit_pos: int) -> int:
    byte_idx = bit_pos >> 3
    bit_gap = bit_pos & 7

    end = byte_idx + 8
    if end <= n_bytes:
        chunk = encoded[byte_idx:end]
        buf = (
            (int(chunk[0]) << 56) | (int(chunk[1]) << 48) |
            (int(chunk[2]) << 40) | (int(chunk[3]) << 32) |
            (int(chunk[4]) << 24) | (int(chunk[5]) << 16) |
            (int(chunk[6]) << 8)  |  int(chunk[7])
        )
    else:
        buf = 0
        for i in range(8):
            v = int(encoded[byte_idx + i]) if (byte_idx + i) < n_bytes else 0
            buf = (buf << 8) | v
    if bit_gap:
        buf = (buf << bit_gap) & 0xFFFFFFFFFFFFFFFF

    d = int(luts[0, (buf >> 56) & 0xFF])
    if d >= _PTR_MIN and (256 - d) < k:
        d = int(luts[256 - d, (buf >> 48) & 0xFF])
        if d >= _PTR_MIN and (256 - d) < k:
            d = int(luts[256 - d, (buf >> 40) & 0xFF])
            if d >= _PTR_MIN and (256 - d) < k:
                d = int(luts[256 - d, (buf >> 32) & 0xFF])

    code_len = int(luts[k, d])
    if code_len == 0:
        raise RuntimeError(
            f"DF11 host bit-position seek: code_len=0 at bit_pos={bit_pos}; "
            "bundle is malformed."
        )
    return code_len


def _advance_bit_position(
    encoded: np.ndarray,
    luts: np.ndarray,
    k: int,
    n_bytes: int,
    bit_pos: int,
    n_symbols: int,
) -> int:
    for _ in range(n_symbols):
        bit_pos += _lookup_code_len(encoded, luts, k, n_bytes, bit_pos)
    return bit_pos


def _block_bit_start(bundle: Dict, block_id: int) -> int:
    threads_per_block = int(bundle["T"])
    bytes_per_thread = int(bundle["n"])
    thread_start = block_id * threads_per_block
    return thread_start * bytes_per_thread * 8 + _extract_gap(
        np.asarray(bundle["gaps"], dtype=np.uint8),
        thread_start,
    )


def _get_or_compute_elem_bit_start(bundle: Dict, elem_idx: int) -> int:
    """Return exact encoded bit position for an element boundary.

    The bundle stores exact element and bit positions at compressor block
    boundaries. To seek a page boundary, start from the nearest containing
    block and decode only the symbols between that block and ``elem_idx``.
    """
    elem_idx = int(elem_idx)
    if elem_idx <= 0:
        return 0
    n_elements = int(bundle["n_elements"])
    if elem_idx >= n_elements:
        elem_idx = n_elements

    cache = bundle.setdefault("_df11_elem_bit_start_cache", {0: 0})
    cached = cache.get(elem_idx)
    if cached is not None:
        return int(cached)

    output_positions = np.asarray(bundle["output_positions"], dtype=np.uint32)
    block_id = int(np.searchsorted(output_positions, elem_idx, side="right") - 1)
    block_id = max(0, min(block_id, len(output_positions) - 1))
    block_elem_start = int(output_positions[block_id])

    bit_pos = _block_bit_start(bundle, block_id)
    if elem_idx > block_elem_start:
        encoded = np.asarray(bundle["encoded_exponent"], dtype=np.uint8)
        luts = np.asarray(bundle["luts"], dtype=np.uint8)
        bit_pos = _advance_bit_position(
            encoded,
            luts,
            int(bundle["k"]),
            int(bundle["n_bytes"]),
            bit_pos,
            elem_idx - block_elem_start,
        )

    cache[elem_idx] = int(bit_pos)
    return int(bit_pos)


def _get_or_build_page_bit_starts(bundle: Dict, page_elements: int, n_pages: int) -> np.ndarray:
    """Return cached per-page bit starts for this bundle, computing on first call."""
    cache = bundle.get("_df11_page_bit_starts_cache")
    if cache is not None and cache["page_elements"] == page_elements and cache["n_pages"] == n_pages:
        return cache["page_bit_starts"]

    encoded = np.asarray(bundle["encoded_exponent"], dtype=np.uint8)
    luts    = np.asarray(bundle["luts"], dtype=np.uint8)
    k          = int(bundle["k"])
    n_elements = int(bundle["n_elements"])
    n_bytes    = int(bundle["n_bytes"])

    page_bit_starts = _walk_bit_positions(
        encoded, luts, k, n_elements, n_bytes, page_elements, n_pages
    )

    bundle["_df11_page_bit_starts_cache"] = {
        "page_elements": page_elements,
        "n_pages": n_pages,
        "page_bit_starts": page_bit_starts,
    }
    return page_bit_starts


def _compute_block_core_ranges(
    bundle: Dict,
    max_cores: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ranges at compressor block boundaries without walking symbols."""
    n_elements = int(bundle["n_elements"])
    n_blocks = int(bundle["B"])
    if n_blocks <= 0:
        return (
            np.array([0], dtype=np.uint32),
            np.array([n_elements], dtype=np.uint32),
            np.array([0], dtype=np.uint32),
        )

    n_cores = min(max(1, int(max_cores)), n_blocks)
    output_positions = np.asarray(bundle["output_positions"], dtype=np.uint32)
    gaps = np.asarray(bundle["gaps"], dtype=np.uint8)
    threads_per_block = int(bundle["T"])
    bytes_per_thread = int(bundle["n"])
    bits_per_thread = bytes_per_thread * 8

    blocks_per_core_base = n_blocks // n_cores
    remainder = n_blocks % n_cores

    elem_starts = []
    elem_counts = []
    bit_starts = []

    block_cursor = 0
    for ci in range(n_cores):
        n_blks = blocks_per_core_base + (1 if ci < remainder else 0)
        if n_blks <= 0:
            continue
        blk_start = block_cursor
        blk_end = min(block_cursor + n_blks, n_blocks)

        elem_start = int(output_positions[blk_start])
        elem_end = int(output_positions[blk_end]) if blk_end < len(output_positions) else n_elements
        elem_end = min(elem_end, n_elements)

        thread_start = blk_start * threads_per_block
        bit_start = thread_start * bits_per_thread + _extract_gap(gaps, thread_start)

        elem_starts.append(elem_start)
        elem_counts.append(max(0, elem_end - elem_start))
        bit_starts.append(bit_start)

        block_cursor = blk_end

    return (
        np.array(elem_starts, dtype=np.uint32),
        np.array(elem_counts, dtype=np.uint32),
        np.array(bit_starts, dtype=np.uint32),
    )


def compute_core_ranges(
    bundle: Dict,
    R_pad: int,
    C_pad: int,
    max_cores: int = DEFAULT_MAX_CORES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (elem_starts, elem_counts, bit_starts) for the device split.

    Splits the compressed stream into up to ``max_cores`` contiguous chunks.
    By default chunks start on output page boundaries and seek only the active
    core starts. ``DFLOAT11_SPLIT_MODE=page_full`` keeps the old full page table
    walk for debugging; ``DFLOAT11_SPLIT_MODE=block`` exercises the experimental
    partial-page path.

    Each chunk's ``bit_start`` is the actual cumulative-Huffman-code bit
    position at which that chunk's first element was emitted by the
    compressor.

    Args:
        bundle: bundle dict (as returned by ``compress/bundle.py::read_bundle``
            or ``load_model_bundle``). Mutated to cache boundary bit starts for
            re-use on subsequent calls with the same shape.
        R_pad: padded output rows (== number of output pages).
        C_pad: padded output columns (== elements per output page).
        max_cores: cap on the number of cores. The actual number used is
            ``min(max_cores, R_pad)``.

    Returns:
        Three uint32 numpy arrays, one entry per active core.
    """
    n_elements = int(bundle["n_elements"])
    split_mode = os.environ.get("DFLOAT11_SPLIT_MODE", "page").lower()

    if split_mode == "block":
        return _compute_block_core_ranges(bundle, max_cores=max_cores)

    if R_pad <= 0 or C_pad <= 0:
        # Degenerate; single-core fallback (page boundaries are meaningless here).
        return (
            np.array([0], dtype=np.uint32),
            np.array([n_elements], dtype=np.uint32),
            np.array([0], dtype=np.uint32),
        )

    page_elements = C_pad
    n_pages = R_pad

    n_cores = min(max(1, int(max_cores)), n_pages, max(1, int(bundle["B"])))
    if n_cores <= 1:
        return (
            np.array([0], dtype=np.uint32),
            np.array([n_elements], dtype=np.uint32),
            np.array([0], dtype=np.uint32),
        )

    if split_mode == "page_full":
        page_bit_starts = _get_or_build_page_bit_starts(bundle, page_elements, n_pages)
    else:
        page_bit_starts = None

    # Split the n_pages output pages as evenly as possible across n_cores cores.
    pages_per_core_base = n_pages // n_cores
    remainder = n_pages % n_cores

    elem_starts = []
    elem_counts = []
    bit_starts  = []

    page_cursor = 0
    for ci in range(n_cores):
        n_p = pages_per_core_base + (1 if ci < remainder else 0)
        if n_p == 0:
            continue
        page_start = page_cursor
        page_end   = page_cursor + n_p

        elem_start = page_start * page_elements
        elem_end   = page_end   * page_elements
        if elem_end > n_elements:
            elem_end = n_elements

        if page_bit_starts is None:
            bit_start = _get_or_compute_elem_bit_start(bundle, elem_start)
        else:
            bit_start = int(page_bit_starts[page_start])

        elem_starts.append(elem_start)
        elem_counts.append(elem_end - elem_start)
        bit_starts.append(bit_start)

        page_cursor = page_end

    es = np.array(elem_starts, dtype=np.uint32)
    ec = np.array(elem_counts, dtype=np.uint32)
    bs = np.array(bit_starts,  dtype=np.uint32)

    # Sanity check: bit_starts strictly increasing, elem_starts page-aligned,
    # ranges contiguous and covering [0, n_elements).
    if len(es) > 0:
        assert es[0] == 0, f"first elem_start must be 0, got {es[0]}"
        assert int(es[-1]) + int(ec[-1]) == n_elements, (
            f"core ranges must cover all {n_elements} elements; "
            f"got last_start={int(es[-1])} + count={int(ec[-1])}"
        )
        for i in range(1, len(es)):
            assert int(es[i]) == int(es[i - 1]) + int(ec[i - 1]), (
                "core ranges must be contiguous"
            )
            assert int(es[i]) % page_elements == 0, (
                f"core {i} elem_start={int(es[i])} not page-aligned to {page_elements}"
            )
            assert int(bs[i]) > int(bs[i - 1]), (
                f"core {i} bit_start={int(bs[i])} not after core {i-1} bit_start={int(bs[i-1])}"
            )

    return es, ec, bs
