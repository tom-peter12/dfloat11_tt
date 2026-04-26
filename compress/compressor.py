"""DFloat11-TT compressor: Huffman encode BF16 exponents, build LUT hierarchy."""
from __future__ import annotations

import heapq
import struct
from copy import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


BYTES_PER_THREAD: int = 8
THREADS_PER_BLOCK: int = 512


# ---------------------------------------------------------------------------
# Huffman tree construction
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _HNode:
    freq: int
    symbol: int = field(compare=False, default=-1)  # -1 means internal node
    left: Optional["_HNode"] = field(compare=False, default=None)
    right: Optional["_HNode"] = field(compare=False, default=None)


def build_huffman_tree(counter: Dict[int, int]) -> _HNode:
    """Build a Huffman tree from a frequency counter.

    Tie-breaking rule: when two nodes have equal frequency, the node with the
    lower symbol value wins (or internal nodes beat symbols of equal freq).
    This matches dahuffman's heap ordering for integer keys, ensuring
    bit-identical codes across independent compressions of the same weights.
    """
    # Use (freq, symbol, node) to ensure deterministic tie-breaking.
    # Internal nodes use symbol=-1 which sorts before any real symbol value,
    # mimicking dahuffman's insertion-order tie-break for the same inputs.
    heap: List[Tuple[int, int, _HNode]] = []
    for sym, freq in sorted(counter.items()):
        node = _HNode(freq=freq, symbol=sym)
        heapq.heappush(heap, (freq, sym, node))

    if len(heap) == 1:
        # Edge case: single symbol — wrap in a dummy root.
        freq, sym, node = heapq.heappop(heap)
        root = _HNode(freq=freq, symbol=-1, left=node, right=_HNode(freq=0, symbol=0))
        return root

    internal_counter = 0  # tiebreak for internal nodes: use large negative values so they win
    while len(heap) > 1:
        f1, s1, n1 = heapq.heappop(heap)
        f2, s2, n2 = heapq.heappop(heap)
        merged = _HNode(freq=f1 + f2, symbol=-1, left=n1, right=n2)
        internal_counter -= 1  # more negative → higher priority (lower value)
        heapq.heappush(heap, (f1 + f2, internal_counter, merged))

    _, _, root = heap[0]
    return root


def _assign_codes(node: _HNode, code: int = 0, length: int = 0,
                  table: Optional[Dict[int, Tuple[int, int]]] = None) -> Dict[int, Tuple[int, int]]:
    """DFS assign Huffman codes. Returns {symbol: (length, code)}."""
    if table is None:
        table = {}
    if node.symbol >= 0:
        table[node.symbol] = (length, code)
    else:
        if node.left:
            _assign_codes(node.left, code << 1, length + 1, table)
        if node.right:
            _assign_codes(node.right, (code << 1) | 1, length + 1, table)
    return table


def get_codec(weight: torch.Tensor) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, int]]:
    """Compute exponent histogram and build Huffman code table.

    Returns:
        table: {exponent_value: (code_length, code_bits)}
        counter: {exponent_value: frequency}
    """
    W = weight.view(torch.int16)
    exponent_8bits = ((W >> 7) & 0xFF).to(torch.int32)
    vals, freqs = torch.unique(exponent_8bits, return_counts=True)
    counter: Dict[int, int] = {int(v): int(f) for v, f in zip(vals, freqs)}
    return _build_codec_with_max_len(counter)


def _build_codec_with_max_len(counter: Dict[int, int]) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, int]]:
    """Build Huffman codec enforcing max code length ≤ 32.

    If the initial tree exceeds 32 bits, iteratively promote the rarest
    symbols (set their frequency to 1) until the constraint is satisfied.
    This matches the original dfloat11_utils.get_32bit_codec() behavior.
    """
    freq_array = np.array(list(counter.values()), dtype=np.int64)
    compressed_counter = counter
    min_k = 2

    while True:
        root = build_huffman_tree(compressed_counter)
        table = _assign_codes(root)
        max_len = max(length for length, _ in table.values()) if table else 0
        if max_len <= 32:
            return table, compressed_counter
        # Promote min_k rarest symbols
        min_indices = np.argpartition(freq_array, min_k)[:min_k]
        min_keys = np.array(list(counter.keys()))[min_indices]
        compressed_counter = copy(counter)
        for k in min_keys:
            compressed_counter[int(k)] = 1
        min_k += 1


def build_luts(table: Dict[int, Tuple[int, int]]) -> np.ndarray:
    """Build the hierarchical LUT array from a Huffman code table.

    Returns ndarray of shape (k+1, 256) dtype uint8, where k is the number of
    decode LUTs. The last row maps exponent_value → code_length_in_bits.
    This is a direct port of dfloat11_utils.get_luts(), preserving the exact
    same LUT structure so device-side decode works unchanged.
    """
    prefixes = ['']

    for key, (bits, val) in table.items():
        if isinstance(key, int):
            prefix = bin(val)[2:].rjust(bits, '0')[:((bits - 1) // 8 * 8)]
            if prefix not in prefixes:
                prefixes.append(prefix)

    prefixes.sort(key=len)

    luts = np.zeros((len(prefixes), 256), dtype=np.uint8)

    for pi, p in enumerate(prefixes):
        bytes_dict: Dict[int, int] = {}
        pl = len(p) // 8
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                bin_val = bin(val)[2:].rjust(bits, '0')
                if bin_val.startswith(p):
                    if (bits - 1) // 8 == pl:
                        dict_key = int(bin_val[(pl * 8):].ljust(8, '0'), 2)
                        dict_value = key
                    else:
                        dict_key = int(bin_val[(pl * 8):(pl * 8 + 8)], 2)
                        dict_value = 256 - prefixes.index(bin_val[:(pl * 8 + 8)])
                    if dict_key in bytes_dict and bytes_dict[dict_key] != dict_value:
                        raise ValueError(f"LUT collision at key {dict_key}: {bytes_dict[dict_key]} vs {dict_value}")
                    bytes_dict[dict_key] = dict_value

        curr_val = 0
        for i in range(256):
            if i in bytes_dict:
                curr_val = bytes_dict[i]
            luts[pi, i] = curr_val

    lens = np.zeros((1, 256), dtype=np.uint8)
    for key, (bits, _val) in table.items():
        if isinstance(key, int):
            lens[0, key] = bits

    return np.concatenate((luts, lens), axis=0)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_exponents(
    exponent_values: List[int],
    table: Dict[int, Tuple[int, int]],
    bytes_per_thread: int = BYTES_PER_THREAD,
    threads_per_block: int = THREADS_PER_BLOCK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a list of exponent values using the Huffman table.

    Returns:
        encoded:          uint8 array of packed bitstream
        gaps:             uint8 array (5-bit packed, big-endian), one per thread
        output_positions: uint32 array, length = n_blocks + 1
    """
    encoded_bytes: List[int] = []
    gaps: List[int] = []
    output_positions: List[int] = []

    buffer = 0
    size = 0
    total_size = 0  # total bits emitted so far
    element_count = 0

    bits_per_thread = 8 * bytes_per_thread
    bits_per_block = bits_per_thread * threads_per_block

    for s in exponent_values:
        # Record gap for this thread if we're at the start of a new thread's range
        if total_size // bits_per_thread + 1 > len(gaps):
            gaps.append(total_size - (total_size // bits_per_thread) * bits_per_thread)

        # Record block output position if we're at the start of a new block
        if total_size // bits_per_block + 1 > len(output_positions):
            output_positions.append(element_count)

        b, v = table[s]
        buffer = (buffer << b) + v
        size += b
        total_size += b
        element_count += 1

        while size >= 8:
            byte = (buffer >> (size - 8)) & 0xFF
            encoded_bytes.append(byte)
            buffer = buffer - ((buffer >> (size - 8)) << (size - 8))
            size -= 8

    # Flush remaining bits
    if size > 0:
        if total_size // bits_per_thread + 1 > len(gaps):
            gaps.append(total_size - (total_size // bits_per_thread) * bits_per_thread)
        if total_size // bits_per_block + 1 > len(output_positions):
            output_positions.append(element_count)
        encoded_bytes.append((buffer << (8 - size)) & 0xFF)

    output_positions.append(len(exponent_values))

    # Pad gaps to fill all threads in all blocks
    n_blocks = int(np.ceil(len(encoded_bytes) / (threads_per_block * bytes_per_thread)))
    total_threads = threads_per_block * n_blocks
    while len(gaps) < total_threads:
        gaps.append(0)

    # Pack gaps as 5-bit big-endian
    binary_str_gaps = [format(g, '05b') for g in gaps]
    binary_bits = [int(bit) for binary in binary_str_gaps for bit in binary]
    packed_gaps = np.packbits(binary_bits, bitorder='big')

    return (
        np.frombuffer(bytes(encoded_bytes), dtype=np.uint8),
        packed_gaps,
        np.array(output_positions, dtype=np.uint32),
    )


def compress_tensor(
    weight: torch.Tensor,
    bytes_per_thread: int = BYTES_PER_THREAD,
    threads_per_block: int = THREADS_PER_BLOCK,
) -> Dict:
    """Compress a BF16 weight tensor to DFloat11-TT format.

    Returns a dict with all fields needed to write the bundle format.
    """
    assert weight.dtype == torch.bfloat16, f"Expected bfloat16, got {weight.dtype}"
    original_shape = list(weight.shape)
    flat = weight.detach().cpu().flatten()

    W = flat.view(torch.int16)
    exponent_8bits = ((W >> 7) & 0xFF).to(torch.uint8)
    sign_mantissa = ((W >> 8).to(torch.int16) & 0x80 | W.to(torch.int16) & 0x7F).to(torch.uint8)

    table, compressed_counter = get_codec(flat)
    luts_array = build_luts(table)  # shape (k+1, 256)
    k = luts_array.shape[0] - 1    # number of decode LUTs

    encoded, gaps, output_positions = encode_exponents(
        exponent_8bits.tolist(), table, bytes_per_thread, threads_per_block
    )

    # Padded dimensions for tile alignment (multiple of 32)
    ndim = len(original_shape)
    if ndim >= 2:
        R = original_shape[-2]
        C = original_shape[-1]
    else:
        R = 1
        C = original_shape[-1]
    R_pad = int(np.ceil(R / 32)) * 32
    C_pad = int(np.ceil(C / 32)) * 32

    n_elements = int(flat.numel())
    n_bytes = int(encoded.shape[0])
    n_blocks = int(np.ceil(n_bytes / (threads_per_block * bytes_per_thread)))

    return {
        "dtype": 0,  # bfloat16
        "shape": original_shape,
        "k": k,
        "n": bytes_per_thread,
        "T": threads_per_block,
        "B": n_blocks,
        "R_pad": R_pad,
        "C_pad": C_pad,
        "n_elements": n_elements,
        "n_bytes": n_bytes,
        "luts": luts_array,
        "encoded_exponent": encoded,
        "sign_mantissa": sign_mantissa.numpy(),
        "gaps": gaps,
        "output_positions": output_positions,
        "table": table,
        "compression_ratio": (encoded.nbytes + sign_mantissa.numel() + output_positions.nbytes + gaps.nbytes) / (n_elements * 2),
    }
