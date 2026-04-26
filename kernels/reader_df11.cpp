// SPDX-License-Identifier: Apache-2.0
// DFloat11-TT reader+decode kernel (BRISC / DataMovement0).
//
// Responsibilities:
//   1. Load LUT array from DRAM into L1 scratch.
//   2. For each assigned block:
//      a. Read encoded exponent bytes from DRAM into a temp L1 buffer.
//      b. Read sign_mantissa bytes from DRAM into a temp L1 buffer.
//      c. Read that block's gaps and output_positions into L1 scratch.
//      d. Run the two-phase DFloat11 decode (prefix-sum + decode) directly.
//      e. Push decoded BF16 row-major pages into cb_decoded for the writer.
//
// Compile-time args:
//   0: CHUNK_BYTES      (page size = 4096)
//   1: T                (threads per block, 512)
//   2: N_BPT            (bytes per logical thread, 8)
//   3: K                (number of decode LUTs)
//   4: unused
//   5: unused
//   6: n_blocks_core    (blocks assigned to this core)
//   7: output_page_bytes (row-major output page size)
//   8: encoded_exponent page size
//   9: sign_mantissa page size
//   10: luts page size
//   11: gaps page size
//   12: output_positions page size
//
// Runtime args:
//   0: encoded_exponent DRAM base address
//   1: sign_mantissa DRAM base address
//   2: luts DRAM base address
//   3: gaps DRAM base address
//   4: output_positions DRAM base address
//   5: core_byte_start
//   6: core_byte_end
//   7: (unused)
//   8: (unused)
//   9: block_id_start
//   10: block_id_end
//   11: lut_total_bytes  ((k+1)*256)
//   12: (unused)
//   13: unused
//   14: unused
//   15: lut_l1_addr
//   16: gaps_l1_addr
//   17: outpos_l1_addr
//   18: scratch_l1_addr  (T * sizeof(uint32_t) scratch for prefix sum)
//   19: enc_l1_addr      (temp buffer for one block's encoded bytes + overlap)
//   20: sm_l1_addr       (temp buffer for one block's sign_mantissa bytes)
//   21: wbuf_l1_addr     (unused; CB page is used as the write buffer)
//   22: n_elements_global
//   23: n_output_pages

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

// FORCE_INLINE is already defined by risc_attribs.h (included via firmware wrapper)

// ---- bit manipulation helper ----
FORCE_INLINE uint64_t u64_shl(uint64_t v, uint32_t s) {
    return s >= 64u ? 0ull : (v << s);
}

FORCE_INLINE uint8_t lut_lookup(
    const uint8_t* __restrict__ lut_base,
    uint32_t k,
    uint64_t buf,
    uint8_t& code_len)
{
    constexpr uint32_t ENTRIES = 256u;
    constexpr uint8_t  PTR_MIN = 240u;
    uint8_t d = lut_base[(buf >> 56) & 0xFFu];
    if (d >= PTR_MIN && (256u - d) < k) {
        uint32_t nx = 256u - d;
        d = lut_base[ENTRIES * nx + ((buf >> 48) & 0xFFu)];
        if (d >= PTR_MIN && (256u - d) < k) {
            uint32_t nx2 = 256u - d;
            d = lut_base[ENTRIES * nx2 + ((buf >> 40) & 0xFFu)];
            if (d >= PTR_MIN && (256u - d) < k) {
                uint32_t nx3 = 256u - d;
                d = lut_base[ENTRIES * nx3 + ((buf >> 32) & 0xFFu)];
            }
        }
    }
    code_len = lut_base[ENTRIES * k + d];
    return d;
}

FORCE_INLINE uint8_t extract_gap(
    const uint8_t* __restrict__ gaps_base,
    uint32_t thread_id,
    uint32_t gap_byte_start)
{
    uint32_t bit_pos = thread_id * 5u - gap_byte_start * 8u;
    // short_be = (gaps[byte] << 8) | gaps[byte+1]  (matching CUDA big-endian convention)
    uint8_t byte_lo  = gaps_base[bit_pos / 8u];
    uint8_t byte_hi  = gaps_base[bit_pos / 8u + 1u];
    uint16_t sv = (static_cast<uint16_t>(byte_lo) << 8u) | static_cast<uint16_t>(byte_hi);
    uint32_t shift = 11u - (bit_pos % 8u);
    return static_cast<uint8_t>((sv >> shift) & 0x1Fu);
}

FORCE_INLINE void zero_page(uint8_t* page, uint32_t page_bytes) {
    for (uint32_t i = 0; i < page_bytes; i++) page[i] = 0;
}

template <typename Accessor>
FORCE_INLINE void read_tensor_bytes(
    const Accessor& accessor,
    uint32_t global_byte_offset,
    uint32_t dst_l1_addr,
    uint32_t byte_count,
    uint32_t page_size)
{
    uint32_t copied = 0;
    while (copied < byte_count) {
        if (page_size == 0) break;
        uint32_t abs = global_byte_offset + copied;
        uint32_t page_id = abs / page_size;
        uint32_t page_offset = abs - page_id * page_size;
        uint32_t n = page_size - page_offset;
        uint32_t remaining = byte_count - copied;
        if (n > remaining) n = remaining;
        uint64_t noc = accessor.get_noc_addr(page_id, page_offset);
        noc_async_read(noc, dst_l1_addr + copied, n);
        noc_async_read_barrier();
        copied += n;
    }
}

void kernel_main() {
    constexpr uint32_t CHUNK       = get_compile_time_arg_val(0);
    constexpr uint32_t T           = get_compile_time_arg_val(1);
    constexpr uint32_t N_BPT       = get_compile_time_arg_val(2);
    constexpr uint32_t K           = get_compile_time_arg_val(3);
    constexpr uint32_t n_blks_core = get_compile_time_arg_val(6);
    constexpr uint32_t OUT_PAGE_BYTES = get_compile_time_arg_val(7);
    constexpr uint32_t ENC_PAGE_BYTES = get_compile_time_arg_val(8);
    constexpr uint32_t SM_PAGE_BYTES = get_compile_time_arg_val(9);
    constexpr uint32_t LUT_PAGE_BYTES = get_compile_time_arg_val(10);
    constexpr uint32_t GAPS_PAGE_BYTES = get_compile_time_arg_val(11);
    constexpr uint32_t OUTPOS_PAGE_BYTES = get_compile_time_arg_val(12);

    constexpr auto enc_args = TensorAccessorArgs<13>();
    constexpr auto sm_args = TensorAccessorArgs<enc_args.next_compile_time_args_offset()>();
    constexpr auto lut_args = TensorAccessorArgs<sm_args.next_compile_time_args_offset()>();
    constexpr auto gaps_args = TensorAccessorArgs<lut_args.next_compile_time_args_offset()>();
    constexpr auto outpos_args = TensorAccessorArgs<gaps_args.next_compile_time_args_offset()>();

    const uint32_t enc_dram_addr    = get_arg_val<uint32_t>(0);
    const uint32_t sm_dram_addr     = get_arg_val<uint32_t>(1);
    const uint32_t lut_dram_addr    = get_arg_val<uint32_t>(2);
    const uint32_t gaps_dram_addr   = get_arg_val<uint32_t>(3);
    const uint32_t outpos_dram_addr = get_arg_val<uint32_t>(4);
    const uint32_t core_byte_start  = get_arg_val<uint32_t>(5);
    const uint32_t core_byte_end    = get_arg_val<uint32_t>(6);
    // args 7,8 unused
    const uint32_t block_id_start   = get_arg_val<uint32_t>(9);
    // arg 10 unused (block_id_end derivable)
    const uint32_t lut_total_bytes  = get_arg_val<uint32_t>(11);
    // arg 12 unused
    // args 13,14 unused
    const uint32_t lut_l1_addr      = get_arg_val<uint32_t>(15);
    const uint32_t gaps_l1_addr     = get_arg_val<uint32_t>(16);
    const uint32_t outpos_l1_addr   = get_arg_val<uint32_t>(17);
    const uint32_t scratch_l1_addr  = get_arg_val<uint32_t>(18);
    const uint32_t enc_l1_addr      = get_arg_val<uint32_t>(19);
    const uint32_t sm_l1_addr       = get_arg_val<uint32_t>(20);
    // arg 21 unused
    const uint32_t n_elements_global= get_arg_val<uint32_t>(22);
    const uint32_t n_output_pages   = get_arg_val<uint32_t>(23);

    constexpr uint32_t cb_decoded = 16;  // CBIndex::c_16

    const auto enc_accessor = TensorAccessor(enc_args, enc_dram_addr, ENC_PAGE_BYTES);
    const auto sm_accessor = TensorAccessor(sm_args, sm_dram_addr, SM_PAGE_BYTES);
    const auto lut_accessor = TensorAccessor(lut_args, lut_dram_addr, LUT_PAGE_BYTES);
    const auto gaps_accessor = TensorAccessor(gaps_args, gaps_dram_addr, GAPS_PAGE_BYTES);
    const auto outpos_accessor = TensorAccessor(outpos_args, outpos_dram_addr, OUTPOS_PAGE_BYTES);
    // ------------------------------------------------------------------
    // Step 1: Load LUTs into L1 scratch.
    // ------------------------------------------------------------------
    read_tensor_bytes(lut_accessor, 0, lut_l1_addr, lut_total_bytes, LUT_PAGE_BYTES);

    const uint8_t*  lut_base    = reinterpret_cast<const uint8_t*>(lut_l1_addr);
    uint32_t*       scratch     = reinterpret_cast<uint32_t*>(scratch_l1_addr);

    // ------------------------------------------------------------------
    // Correctness-first path: decode the Huffman bitstream sequentially.
    //
    // This avoids the CUDA-style gap/prefix-sum split while we validate the
    // end-to-end device path.  It still materializes the BF16 tensor on-device
    // and streams row-major pages through cb_decoded to the writer.
    // ------------------------------------------------------------------
    constexpr uint32_t ENC_WINDOW_BYTES = CHUNK;
    const uint8_t* enc_window = reinterpret_cast<const uint8_t*>(enc_l1_addr);
    const uint8_t* sm_window = reinterpret_cast<const uint8_t*>(sm_l1_addr);

    uint32_t enc_window_start = 0xFFFFFFFFu;
    uint32_t enc_window_count = 0;
    uint32_t bit_pos = 0;
    uint32_t out_idx = 0;

    for (uint32_t page_id = 0; page_id < n_output_pages; page_id++) {
        cb_reserve_back(cb_decoded, 1);
        uint8_t* page_buf = reinterpret_cast<uint8_t*>(get_write_ptr(cb_decoded));
        zero_page(page_buf, OUT_PAGE_BYTES);

        uint32_t page_start_byte = page_id * OUT_PAGE_BYTES;
        uint32_t total_output_bytes = n_elements_global * sizeof(uint16_t);
        uint32_t valid_page_bytes = OUT_PAGE_BYTES;
        if (page_start_byte + valid_page_bytes > total_output_bytes) {
            valid_page_bytes = total_output_bytes - page_start_byte;
        }
        uint32_t valid_elems = valid_page_bytes / sizeof(uint16_t);

        if (valid_elems > 0) {
            read_tensor_bytes(sm_accessor, out_idx, sm_l1_addr, valid_elems, SM_PAGE_BYTES);
        }

        for (uint32_t elem_in_page = 0; elem_in_page < valid_elems && out_idx < n_elements_global; elem_in_page++) {
            uint32_t byte_idx = bit_pos / 8u;
            uint32_t bit_gap = bit_pos - byte_idx * 8u;

            uint32_t enc_window_end = enc_window_start + enc_window_count;
            if (enc_window_start == 0xFFFFFFFFu ||
                byte_idx < enc_window_start ||
                ((byte_idx + 8u) > enc_window_end && enc_window_end < core_byte_end)) {
                enc_window_start = byte_idx & ~31u;
                enc_window_count = ENC_WINDOW_BYTES + 8u;
                if (enc_window_start + enc_window_count > core_byte_end) {
                    enc_window_count = core_byte_end - enc_window_start;
                }
                read_tensor_bytes(enc_accessor, enc_window_start, enc_l1_addr, enc_window_count, ENC_PAGE_BYTES);
            }

            uint32_t local_byte = byte_idx - enc_window_start;
            uint64_t long_buffer = 0;
            for (uint32_t i = 0; i < 8u; i++) {
                uint8_t v = 0;
                if (local_byte + i < enc_window_count) {
                    v = enc_window[local_byte + i];
                }
                long_buffer = (long_buffer << 8) | v;
            }
            long_buffer = u64_shl(long_buffer, bit_gap);

            uint8_t code_len = 0;
            uint8_t decoded = lut_lookup(lut_base, K, long_buffer, code_len);
            if (code_len == 0) {
                break;
            }

            uint8_t sm = sm_window[elem_in_page];
            uint8_t high = (sm & 0x80u) | (decoded >> 1u);
            uint8_t low  = ((decoded & 1u) << 7u) | (sm & 0x7Fu);
            uint32_t out_byte = elem_in_page * sizeof(uint16_t);
            page_buf[out_byte] = low;
            page_buf[out_byte + 1u] = high;

            bit_pos += code_len;
            out_idx++;
        }

        cb_push_back(cb_decoded, 1);
    }
}
