"""Pure-Python reference decoder for bit-identity testing.

This decoder implements the same algorithm as the CUDA kernel and the
Tenstorrent RISC-V kernel, but in pure Python. It is used exclusively
for correctness verification, not for inference performance.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def _extract_gap(gaps: np.ndarray, thread_id: int) -> int:
    """Extract the 5-bit gap value for a given thread ID from the packed gaps array."""
    bit_pos = thread_id * 5
    byte0 = int(gaps[bit_pos // 8])
    byte1 = int(gaps[bit_pos // 8 + 1]) if (bit_pos // 8 + 1) < len(gaps) else 0
    short = (byte1 << 8) | byte0
    # The 5 bits are packed big-endian across bytes, matching the CUDA extraction:
    # gaps[thread_id * 5 / 8 + 1] is the HIGH byte (buffer[8] in CUDA),
    # gaps[thread_id * 5 / 8]     is the LOW byte  (buffer[9] in CUDA).
    # short_buffer = (buffer[9] << 8) | buffer[8] in CUDA = big-endian short.
    # gap = (short_buffer >> (11 - (thread_id * 5 % 8))) & 0x1f
    shift = 11 - (bit_pos % 8)
    # Re-read as big-endian:
    byte_high = int(gaps[bit_pos // 8 + 1]) if (bit_pos // 8 + 1) < len(gaps) else 0
    byte_low  = int(gaps[bit_pos // 8])
    short_be = (byte_low << 8) | byte_high
    return (short_be >> shift) & 0x1F


def _lut_lookup(luts: np.ndarray, long_buffer: int) -> int:
    """Perform the hierarchical LUT lookup on a 64-bit MSB-aligned bit buffer.

    Returns the decoded exponent value (0-255).

    Pointer values 240-255 in a decode LUT row indicate a multi-byte code that
    needs a second lookup in the next row.  We only follow a pointer if the
    target row index (= 256 - decoded) is still within the decode LUT rows
    (< k).  If decoded >= 240 but the target would be the code-length row or
    beyond, the value IS the symbol (exponent), not a pointer.
    """
    k = luts.shape[0] - 1  # number of decode LUTs (last row is code lengths)
    decoded = int(luts[0, (long_buffer >> 56) & 0xFF])

    lut_idx = 1
    while decoded >= 240 and (256 - decoded) < k:
        next_table = 256 - decoded
        next_byte = (long_buffer >> (56 - lut_idx * 8)) & 0xFF
        decoded = int(luts[next_table, next_byte])
        lut_idx += 1

    return decoded


def decode_bundle(bundle: Dict) -> torch.Tensor:
    """Decode a DFloat11-TT bundle back to BF16 tensor.

    This is the reference implementation used for bit-identity testing.
    Implements the exact same two-phase algorithm as the CUDA/RISC-V kernels.
    """
    luts = bundle["luts"]                          # (k+1, 256) uint8
    encoded = bundle["encoded_exponent"]            # uint8 array
    sign_mantissa = bundle["sign_mantissa"]         # uint8 array
    gaps = bundle["gaps"]                           # packed 5-bit gaps
    output_positions = bundle["output_positions"]   # uint32 block starts
    T = bundle["T"]                                 # threads per block
    n = bundle["n"]                                 # bytes per thread (8)
    k = bundle["k"]                                 # number of decode LUTs
    n_elements = bundle["n_elements"]
    n_bytes = bundle["n_bytes"]
    shape = bundle["shape"]

    outputs = np.zeros(n_elements, dtype=np.uint16)

    total_threads = len(gaps) * 8 // 5  # approximate; actual = ceil(n_bytes / n)
    total_threads = int(np.ceil(n_bytes / n))

    for global_thread_id in range(total_threads):
        block_id = global_thread_id // T
        thread_id_in_block = global_thread_id % T

        byte_start = global_thread_id * n
        if byte_start >= n_bytes:
            break

        # Load 12 bytes (8 for this thread, 4 from overlap)
        register_buffer = np.zeros(12, dtype=np.uint8)
        for i in range(12):
            if byte_start + i < n_bytes:
                register_buffer[i] = encoded[byte_start + i]

        # Extract gap
        if global_thread_id * 5 // 8 + 1 < len(gaps):
            gap = _extract_gap(gaps, global_thread_id)
        else:
            gap = 0

        # Build 64-bit big-endian buffer from first 8 bytes (rb[0] at MSB)
        long_buffer = 0
        for i in range(8):
            long_buffer = (long_buffer << 8) | int(register_buffer[i])
        long_buffer &= 0xFFFFFFFFFFFFFFFF

        long_buffer = (long_buffer << gap) & 0xFFFFFFFFFFFFFFFF
        free_bits = gap

        thread_counter = 0

        # Phase 1 counting — first 32+gap bits
        while free_bits < 32:
            if byte_start * 8 + gap + thread_counter * 1 >= n_bytes * 8:
                break
            decoded = _lut_lookup(luts, long_buffer)
            code_len = int(luts[k, decoded])
            if code_len == 0:
                break
            thread_counter += 1
            long_buffer = (long_buffer << code_len) & 0xFFFFFFFFFFFFFFFF
            free_bits += code_len

        # Load bytes 8-11 into the upper bits (rb[8] at MSB of extra)
        extra = 0
        for i in range(4):
            extra = (extra << 8) | int(register_buffer[8 + i])
        extra &= 0xFFFFFFFF
        if free_bits >= 32:
            long_buffer = (long_buffer | (extra << (free_bits - 32))) & 0xFFFFFFFFFFFFFFFF
            free_bits -= 32

        # Continue counting until we've consumed this thread's byte range
        while (4 + free_bits // 8) < n:
            decoded = _lut_lookup(luts, long_buffer)
            code_len = int(luts[k, decoded])
            if code_len == 0:
                break
            thread_counter += 1
            long_buffer = (long_buffer << code_len) & 0xFFFFFFFFFFFFFFFF
            free_bits += code_len

        # Compute output position for this thread using block output positions
        block_start = int(output_positions[block_id])
        # We need per-thread positions: computed by prefix sum across threads in block.
        # For reference decoder, compute this directly using a sequential scan.
        # (This is cached per block for efficiency.)
        pass  # handled below via full sequential decode

    # Simpler O(n) reference: fully sequential decode (no threading)
    _decode_sequential(luts, encoded, sign_mantissa, gaps, output_positions,
                       T, n, k, n_elements, n_bytes, outputs)

    # Reconstruct BF16 tensor
    bf16_flat = torch.from_numpy(outputs.view(np.uint16)).view(torch.bfloat16)
    return bf16_flat.reshape(shape)


def _decode_sequential(
    luts: np.ndarray,
    encoded: np.ndarray,
    sign_mantissa: np.ndarray,
    gaps: np.ndarray,
    output_positions: np.ndarray,
    T: int,
    n: int,
    k: int,
    n_elements: int,
    n_bytes: int,
    outputs: np.ndarray,
) -> None:
    """Sequential reference decoder matching the CUDA two-phase algorithm.

    Processes all threads sequentially, computing each thread's output position
    via an intra-block prefix sum. Writes assembled BF16 uint16 into outputs[].
    """
    total_threads = int(np.ceil(n_bytes / n))
    n_blocks = int(np.ceil(total_threads / T))

    output_idx_global = 0

    for block_id in range(n_blocks):
        block_thread_counters: List[int] = []
        block_start_element = int(output_positions[block_id])

        # Phase 1: count symbols per thread in this block
        for t in range(T):
            global_thread_id = block_id * T + t
            if global_thread_id >= total_threads:
                break
            byte_start = global_thread_id * n
            if byte_start >= n_bytes:
                break

            register_buffer = np.zeros(12, dtype=np.uint8)
            for i in range(12):
                if byte_start + i < n_bytes:
                    register_buffer[i] = encoded[byte_start + i]

            gap = _extract_gap(gaps, global_thread_id) if global_thread_id * 5 // 8 < len(gaps) else 0

            long_buffer = 0
            for i in range(8):
                long_buffer = (long_buffer << 8) | int(register_buffer[i])
            long_buffer &= 0xFFFFFFFFFFFFFFFF
            long_buffer = (long_buffer << gap) & 0xFFFFFFFFFFFFFFFF
            free_bits = gap
            thread_counter = 0

            while free_bits < 32:
                decoded = _lut_lookup(luts, long_buffer)
                code_len = int(luts[k, decoded])
                if code_len == 0:
                    break
                thread_counter += 1
                long_buffer = (long_buffer << code_len) & 0xFFFFFFFFFFFFFFFF
                free_bits += code_len

            extra = 0
            for i in range(4):
                extra = (extra << 8) | int(register_buffer[8 + i])
            extra &= 0xFFFFFFFF
            if free_bits >= 32:
                long_buffer = (long_buffer | (extra << (free_bits - 32))) & 0xFFFFFFFFFFFFFFFF
                free_bits -= 32

            while (4 + free_bits // 8) < n:
                decoded = _lut_lookup(luts, long_buffer)
                code_len = int(luts[k, decoded])
                if code_len == 0:
                    break
                thread_counter += 1
                long_buffer = (long_buffer << code_len) & 0xFFFFFFFFFFFFFFFF
                free_bits += code_len

            block_thread_counters.append(thread_counter)

        # Compute prefix sum for per-thread output positions within block
        thread_output_pos: List[int] = []
        pos = block_start_element
        for count in block_thread_counters:
            thread_output_pos.append(pos)
            pos += count

        block_end_element = int(output_positions[block_id + 1]) if block_id + 1 < len(output_positions) else n_elements

        # Phase 2: decode and write
        for t, (t_pos, t_count) in enumerate(zip(thread_output_pos, block_thread_counters)):
            global_thread_id = block_id * T + t
            byte_start = global_thread_id * n
            if byte_start >= n_bytes:
                break

            register_buffer = np.zeros(12, dtype=np.uint8)
            for i in range(12):
                if byte_start + i < n_bytes:
                    register_buffer[i] = encoded[byte_start + i]

            gap = _extract_gap(gaps, global_thread_id) if global_thread_id * 5 // 8 < len(gaps) else 0

            long_buffer = 0
            for i in range(8):
                long_buffer = (long_buffer << 8) | int(register_buffer[i])
            long_buffer &= 0xFFFFFFFFFFFFFFFF
            long_buffer = (long_buffer << gap) & 0xFFFFFFFFFFFFFFFF
            free_bits = gap

            output_idx = t_pos
            end_output_idx = min(output_idx + t_count, n_elements)

            # First half (bytes 0-7, first 32 bits)
            while free_bits < 32 and output_idx < end_output_idx:
                decoded = _lut_lookup(luts, long_buffer)
                code_len = int(luts[k, decoded])
                if code_len == 0:
                    break

                sm = int(sign_mantissa[output_idx])
                # Reassemble BF16: high byte = sign|upper7_exp, low byte = low1_exp|mantissa
                # decoded is the 8-bit exponent
                high = (sm & 0x80) | (decoded >> 1)
                low  = ((decoded & 1) << 7) | (sm & 0x7F)
                # Little-endian uint16: low byte at lower address
                outputs[output_idx] = (high << 8) | low

                output_idx += 1
                long_buffer = (long_buffer << code_len) & 0xFFFFFFFFFFFFFFFF
                free_bits += code_len

            extra = 0
            for i in range(4):
                extra = (extra << 8) | int(register_buffer[8 + i])
            extra &= 0xFFFFFFFF
            if free_bits >= 32:
                long_buffer = (long_buffer | (extra << (free_bits - 32))) & 0xFFFFFFFFFFFFFFFF
                free_bits -= 32

            while output_idx < end_output_idx:
                decoded = _lut_lookup(luts, long_buffer)
                code_len = int(luts[k, decoded])
                if code_len == 0:
                    break

                sm = int(sign_mantissa[output_idx])
                high = (sm & 0x80) | (decoded >> 1)
                low  = ((decoded & 1) << 7) | (sm & 0x7F)
                outputs[output_idx] = (high << 8) | low

                output_idx += 1
                long_buffer = (long_buffer << code_len) & 0xFFFFFFFFFFFFFFFF
                free_bits += code_len
