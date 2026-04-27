// SPDX-License-Identifier: Apache-2.0
// DFloat11-TT writer kernel (runs on NCRISC / DataMovement1).
//
// Pops decoded row-major pages from cb_decoded and writes them to a TTNN tensor.
//
// Runtime args:
//   0: output_dram_addr      (base address of output BF16 tiled tensor in DRAM)
//   1: page_offset_start     (index of first row-major page this core writes)
//   2: n_pages_total         (total number of pages this core writes)
//   3: output_nbytes         (valid bytes in the row-major output tensor)
//   4: core_elem_start       (first global BF16 element this core owns)
//   5: core_elem_count       (number of BF16 elements this core owns)
//
// Circular buffers consumed:
//   cb_decoded (c_16): assembled row-major BF16 pages from reader kernel

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

void kernel_main() {
    const uint32_t output_dram_addr   = get_arg_val<uint32_t>(0);
    const uint32_t page_offset_start  = get_arg_val<uint32_t>(1);
    const uint32_t n_pages_total      = get_arg_val<uint32_t>(2);
    const uint32_t output_nbytes      = get_arg_val<uint32_t>(3);
    const uint32_t core_elem_start    = get_arg_val<uint32_t>(4);
    const uint32_t core_elem_count    = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_decoded   = 16;                      // CBIndex::c_16
    constexpr uint32_t PAGE_BYTES   = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const auto dst = TensorAccessor(dst_args, output_dram_addr, PAGE_BYTES);
    const uint32_t range_start_byte = core_elem_start * sizeof(uint16_t);
    uint32_t range_end_byte = range_start_byte + core_elem_count * sizeof(uint16_t);
    if (range_end_byte > output_nbytes) {
        range_end_byte = output_nbytes;
    }

    for (uint32_t page_idx = 0; page_idx < n_pages_total; page_idx++) {
        cb_wait_front(cb_decoded, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_decoded);

        uint32_t global_page_id = page_offset_start + page_idx;
        uint32_t page_start_byte = global_page_id * PAGE_BYTES;
        uint32_t page_end_byte = page_start_byte + PAGE_BYTES;
        if (page_end_byte > output_nbytes) {
            page_end_byte = output_nbytes;
        }

        uint32_t write_start_byte = range_start_byte > page_start_byte ? range_start_byte : page_start_byte;
        uint32_t write_end_byte = range_end_byte < page_end_byte ? range_end_byte : page_end_byte;

        uint32_t write_bytes = 0;
        uint32_t page_byte_offset = 0;
        if (write_end_byte > write_start_byte) {
            write_bytes = write_end_byte - write_start_byte;
            page_byte_offset = write_start_byte - page_start_byte;
        }

        if (write_bytes > 0) {
            uint64_t noc_dst = dst.get_noc_addr(global_page_id, page_byte_offset);
            noc_async_write(l1_read_addr + page_byte_offset, noc_dst, write_bytes);
        }

        cb_pop_front(cb_decoded, 1);
    }

    // Final barrier ensures all writes to DRAM are committed before the
    // host-side Finish() call returns. Required on Blackhole (see bring-up guide).
    noc_async_write_barrier();
}
