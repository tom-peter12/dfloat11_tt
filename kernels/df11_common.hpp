// SPDX-License-Identifier: Apache-2.0
// DFloat11-TT: common constants, bit-pack helpers, and LUT structure for all kernels.

#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Protocol constants (must match compressor.py)
// ---------------------------------------------------------------------------
constexpr uint32_t DF11_BYTES_PER_THREAD = 8;
constexpr uint32_t DF11_MAX_CODE_LEN     = 32;   // max Huffman code length enforced by Python
constexpr uint32_t DF11_MAX_LUT_DEPTH    = 4;    // max LUT chain depth (ceil(32/8))
constexpr uint32_t DF11_LUT_ENTRIES      = 256;  // one entry per byte value
constexpr uint32_t DF11_LUT_PTR_MIN      = 240;  // entries >= 240 are next-LUT pointers
constexpr uint32_t DF11_TILE_W           = 32;   // tile width
constexpr uint32_t DF11_TILE_H           = 32;   // tile height
constexpr uint32_t DF11_FACE_W           = 16;
constexpr uint32_t DF11_FACE_H           = 16;
constexpr uint32_t DF11_TILE_BYTES       = DF11_TILE_W * DF11_TILE_H * sizeof(uint16_t); // 2048

// ---------------------------------------------------------------------------
// BF16 assembly: decoded exponent (8 bits) + sign_mantissa byte → uint16_t
//
// BF16 bit layout (little-endian uint16):
//   bits 15-8 (high byte): sign(1) | exponent_hi7(7)
//   bits  7-0 (low byte):  exponent_lo1(1) | mantissa(7)
//
// sign_mantissa byte = (sign << 7) | mantissa
// decoded = 8-bit exponent
// ---------------------------------------------------------------------------
FORCE_INLINE uint16_t df11_assemble_bf16(uint8_t sm, uint8_t decoded_exp) {
    uint8_t high = (sm & 0x80u) | (decoded_exp >> 1u);
    uint8_t low  = ((decoded_exp & 1u) << 7u) | (sm & 0x7Fu);
    return static_cast<uint16_t>(high) << 8u | static_cast<uint16_t>(low);
}

// ---------------------------------------------------------------------------
// Hierarchical LUT lookup on a 64-bit MSB-aligned bit buffer.
//
// lut_base: pointer to the start of the flat (k+1)×256 LUT array.
// k: number of decode LUT rows (last row = code-lengths).
// long_buffer: 64-bit value, current Huffman bits aligned to MSB.
// out_code_len: output, number of bits consumed by this symbol.
// Returns: decoded exponent value (0-255).
// ---------------------------------------------------------------------------
FORCE_INLINE uint8_t df11_lut_lookup(
    const uint8_t* __restrict__ lut_base,
    uint32_t k,
    uint64_t long_buffer,
    uint8_t& out_code_len)
{
    uint8_t decoded = lut_base[(long_buffer >> 56) & 0xFF];
    if (decoded >= DF11_LUT_PTR_MIN && (256u - decoded) < k) {
        uint32_t next = static_cast<uint32_t>(256u - decoded);
        decoded = lut_base[DF11_LUT_ENTRIES * next + ((long_buffer >> 48) & 0xFF)];
        if (decoded >= DF11_LUT_PTR_MIN && (256u - decoded) < k) {
            uint32_t next2 = static_cast<uint32_t>(256u - decoded);
            decoded = lut_base[DF11_LUT_ENTRIES * next2 + ((long_buffer >> 40) & 0xFF)];
            if (decoded >= DF11_LUT_PTR_MIN && (256u - decoded) < k) {
                uint32_t next3 = static_cast<uint32_t>(256u - decoded);
                decoded = lut_base[DF11_LUT_ENTRIES * next3 + ((long_buffer >> 32) & 0xFF)];
            }
        }
    }
    out_code_len = lut_base[DF11_LUT_ENTRIES * k + decoded];
    return decoded;
}

// ---------------------------------------------------------------------------
// Gaps array extraction: get the 5-bit gap for thread_id.
// gaps_base: pointer to the packed 5-bit array.
// ---------------------------------------------------------------------------
FORCE_INLINE uint8_t df11_extract_gap(const uint8_t* __restrict__ gaps_base, uint32_t thread_id) {
    uint32_t bit_pos = thread_id * 5u;
    uint8_t byte_lo  = gaps_base[bit_pos / 8u];
    uint8_t byte_hi  = gaps_base[bit_pos / 8u + 1u];
    uint16_t short_val = (static_cast<uint16_t>(byte_lo) << 8u) | static_cast<uint16_t>(byte_hi);
    uint32_t shift = 11u - (bit_pos % 8u);
    return static_cast<uint8_t>((short_val >> shift) & 0x1Fu);
}

// ---------------------------------------------------------------------------
// Tiled output offset calculation (Path A: decompress directly into tile layout).
//
// For a matrix of padded dimensions (R_pad, C_pad), given global element index idx:
// row = idx / C_orig, col = idx % C_orig (using original un-padded C for element mapping)
// then placed at tile-major address in the (R_pad, C_pad) padded tiled tensor.
// ---------------------------------------------------------------------------
FORCE_INLINE uint32_t df11_tiled_byte_offset(
    uint32_t idx,
    uint32_t C_orig,
    uint32_t C_pad)
{
    uint32_t row = idx / C_orig;
    uint32_t col = idx % C_orig;
    uint32_t n_tile_cols = C_pad / DF11_TILE_W;
    uint32_t tile_row  = row / DF11_TILE_H;
    uint32_t tile_col  = col / DF11_TILE_W;
    uint32_t face      = (row % DF11_TILE_H) / DF11_FACE_H * 2u
                        + (col % DF11_TILE_W) / DF11_FACE_W;
    uint32_t intra     = (row % DF11_FACE_H) * DF11_FACE_W + (col % DF11_FACE_W);
    return (tile_row * n_tile_cols + tile_col) * DF11_TILE_BYTES
         + face * DF11_FACE_W * DF11_FACE_H * sizeof(uint16_t)
         + intra * sizeof(uint16_t);
}
