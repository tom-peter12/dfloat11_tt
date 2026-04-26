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

    constexpr uint32_t cb_decoded   = 16;                      // CBIndex::c_16
    constexpr uint32_t PAGE_BYTES   = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const auto dst = TensorAccessor(dst_args, output_dram_addr, PAGE_BYTES);

    for (uint32_t page_idx = 0; page_idx < n_pages_total; page_idx++) {
        cb_wait_front(cb_decoded, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_decoded);

        uint32_t dest_offset = (page_offset_start + page_idx) * PAGE_BYTES;
        uint64_t noc_dst = dst.get_noc_addr(page_offset_start + page_idx);
        uint32_t write_bytes = PAGE_BYTES;
        if (dest_offset + write_bytes > output_nbytes) {
            write_bytes = output_nbytes - dest_offset;
        }

        if (write_bytes > 0) {
            noc_async_write(l1_read_addr, noc_dst, write_bytes);
        }

        cb_pop_front(cb_decoded, 1);
    }

    // Final barrier ensures all writes to DRAM are committed before the
    // host-side Finish() call returns. Required on Blackhole (see bring-up guide).
    noc_async_write_barrier();
}
