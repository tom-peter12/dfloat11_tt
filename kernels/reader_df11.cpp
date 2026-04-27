// SPDX-License-Identifier: Apache-2.0
// DFloat11-TT reader+decode kernel (BRISC / DataMovement0).
//
// Correctness-first sequential reader adapted for multicore:
//   - Each core starts from its own global element start and global bit start
//   - Each core decodes only its assigned output page range
//   - Writer places those pages at the correct global page offset

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

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
    constexpr uint8_t PTR_MIN = 240u;
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

FORCE_INLINE void zero_page(uint8_t* page, uint32_t page_bytes) {
    for (uint32_t i = 0; i < page_bytes; i++) {
        page[i] = 0;
    }
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
        if (page_size == 0) {
            break;
        }
        uint32_t abs = global_byte_offset + copied;
        uint32_t page_id = abs / page_size;
        uint32_t page_offset = abs - page_id * page_size;
        uint32_t n = page_size - page_offset;
        uint32_t remaining = byte_count - copied;
        if (n > remaining) {
            n = remaining;
        }
        uint64_t noc = accessor.get_noc_addr(page_id, page_offset);
        noc_async_read(noc, dst_l1_addr + copied, n);
        noc_async_read_barrier();
        copied += n;
    }
}

void kernel_main() {
    constexpr uint32_t CHUNK = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t N_BPT = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
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

    const uint32_t enc_dram_addr = get_arg_val<uint32_t>(0);
    const uint32_t sm_dram_addr = get_arg_val<uint32_t>(1);
    const uint32_t lut_dram_addr = get_arg_val<uint32_t>(2);
    const uint32_t gaps_dram_addr = get_arg_val<uint32_t>(3);
    const uint32_t outpos_dram_addr = get_arg_val<uint32_t>(4);
    const uint32_t core_byte_start = get_arg_val<uint32_t>(5);
    const uint32_t core_byte_end = get_arg_val<uint32_t>(6);
    const uint32_t block_id_start = get_arg_val<uint32_t>(9);
    const uint32_t lut_total_bytes = get_arg_val<uint32_t>(11);
    const uint32_t lut_l1_addr = get_arg_val<uint32_t>(15);
    const uint32_t gaps_l1_addr = get_arg_val<uint32_t>(16);
    const uint32_t outpos_l1_addr = get_arg_val<uint32_t>(17);
    const uint32_t scratch_l1_addr = get_arg_val<uint32_t>(18);
    const uint32_t enc_l1_addr = get_arg_val<uint32_t>(19);
    const uint32_t sm_l1_addr = get_arg_val<uint32_t>(20);
    const uint32_t n_elements_global = get_arg_val<uint32_t>(22);
    const uint32_t n_output_pages = get_arg_val<uint32_t>(23);
    const uint32_t core_elem_start = get_arg_val<uint32_t>(24);
    const uint32_t core_bit_start = get_arg_val<uint32_t>(25);
    const uint32_t core_elem_count = get_arg_val<uint32_t>(26);

    constexpr uint32_t cb_decoded = 16;

    const auto enc_accessor = TensorAccessor(enc_args, enc_dram_addr, ENC_PAGE_BYTES);
    const auto sm_accessor = TensorAccessor(sm_args, sm_dram_addr, SM_PAGE_BYTES);
    const auto lut_accessor = TensorAccessor(lut_args, lut_dram_addr, LUT_PAGE_BYTES);
    const auto gaps_accessor = TensorAccessor(gaps_args, gaps_dram_addr, GAPS_PAGE_BYTES);
    const auto outpos_accessor = TensorAccessor(outpos_args, outpos_dram_addr, OUTPOS_PAGE_BYTES);

    read_tensor_bytes(lut_accessor, 0, lut_l1_addr, lut_total_bytes, LUT_PAGE_BYTES);

    const uint8_t* lut_base = reinterpret_cast<const uint8_t*>(lut_l1_addr);

    constexpr uint32_t ENC_WINDOW_BYTES = CHUNK;
    const uint8_t* enc_window = reinterpret_cast<const uint8_t*>(enc_l1_addr);
    const uint8_t* sm_window = reinterpret_cast<const uint8_t*>(sm_l1_addr);

    uint32_t enc_window_start = 0xFFFFFFFFu;
    uint32_t enc_window_count = 0;
    uint32_t bit_pos = core_bit_start;
    uint32_t out_idx = core_elem_start;
    const uint32_t page_elements = OUT_PAGE_BYTES / sizeof(uint16_t);
    const uint32_t core_elem_end =
        (core_elem_start + core_elem_count > n_elements_global)
            ? n_elements_global
            : (core_elem_start + core_elem_count);
    const uint32_t page_start_global = (core_elem_start * sizeof(uint16_t)) / OUT_PAGE_BYTES;

    for (uint32_t page_id = 0; page_id < n_output_pages; page_id++) {
        cb_reserve_back(cb_decoded, 1);
        uint8_t* page_buf = reinterpret_cast<uint8_t*>(get_write_ptr(cb_decoded));
        zero_page(page_buf, OUT_PAGE_BYTES);

        uint32_t global_page_start_elem = (page_start_global + page_id) * page_elements;
        uint32_t global_page_end_elem = global_page_start_elem + page_elements;
        if (global_page_end_elem > n_elements_global) {
            global_page_end_elem = n_elements_global;
        }

        uint32_t decode_start_elem = out_idx;
        if (decode_start_elem < global_page_start_elem) {
            decode_start_elem = global_page_start_elem;
        }
        uint32_t decode_end_elem = core_elem_end;
        if (decode_end_elem > global_page_end_elem) {
            decode_end_elem = global_page_end_elem;
        }

        uint32_t valid_elems = 0;
        uint32_t page_elem_offset = 0;
        if (decode_end_elem > decode_start_elem) {
            valid_elems = decode_end_elem - decode_start_elem;
            page_elem_offset = decode_start_elem - global_page_start_elem;
        }

        if (valid_elems > 0) {
            read_tensor_bytes(sm_accessor, decode_start_elem, sm_l1_addr, valid_elems, SM_PAGE_BYTES);
        }

        for (uint32_t elem_in_page = 0; elem_in_page < valid_elems && out_idx < core_elem_end; elem_in_page++) {
            uint32_t byte_idx = bit_pos / 8u;
            uint32_t bit_gap = bit_pos - byte_idx * 8u;

            uint32_t enc_window_end = enc_window_start + enc_window_count;
            if (enc_window_start == 0xFFFFFFFFu ||
                byte_idx < enc_window_start ||
                ((byte_idx + 8u) > enc_window_end && enc_window_end < core_byte_end)) {
                // Blackhole NOC reads require a 64-byte aligned source for
                // this path. A 32-byte start can read the previous half-line,
                // shifting multicore decode by 256 bits on affected cores.
                enc_window_start = byte_idx & ~63u;
                enc_window_count = ENC_WINDOW_BYTES + 64u + 8u;
                if (enc_window_start >= core_byte_end) {
                    enc_window_count = 0;
                } else if (enc_window_start + enc_window_count > core_byte_end) {
                    enc_window_count = core_byte_end - enc_window_start;
                }
                if (enc_window_count > 0) {
                    read_tensor_bytes(
                        enc_accessor,
                        enc_window_start,
                        enc_l1_addr,
                        enc_window_count,
                        ENC_PAGE_BYTES);
                }
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
            uint8_t low = ((decoded & 1u) << 7u) | (sm & 0x7Fu);
            uint32_t out_byte = (page_elem_offset + elem_in_page) * sizeof(uint16_t);
            page_buf[out_byte] = low;
            page_buf[out_byte + 1u] = high;

            bit_pos += code_len;
            out_idx++;
        }

        cb_push_back(cb_decoded, 1);
    }
}
