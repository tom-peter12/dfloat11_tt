// SPDX-License-Identifier: Apache-2.0
// DFloat11-TT host program: builds and launches the Metalium decompression Program.
//
// Primary entry point: decompress_df11(device, bundle_map, tensor_name) -> ttnn::Tensor
//
// The Metalium Program shards the encoded exponent stream across all available
// compute cores (up to 130 on Blackhole), assigning contiguous byte ranges.

#include "df11_op.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/storage.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/types.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

#ifndef DF11_KERNEL_PREFIX
#define DF11_KERNEL_PREFIX "dfloat11_tt/kernels/"
#endif

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace dfloat11_tt {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static constexpr uint32_t CHUNK_BYTES       = 4096u;
static constexpr uint32_t TILE_BYTES        = 2048u;  // 32*32*2
static constexpr uint32_t L1_LUT_OFFSET     = 0x0000u;
static constexpr uint32_t L1_GAPS_OFFSET    = 0x0500u; // after 1280 bytes LUT
static constexpr uint32_t L1_OUTPOS_OFFSET  = 0x1500u;
static constexpr uint32_t L1_SCRATCH_OFFSET = 0x2500u;
static constexpr uint32_t L1_WBUF_OFFSET    = 0x3500u;
// Each of the above regions has generous padding; total << 1.5MB per Tensix.

static uint32_t round_up_32(uint32_t v) {
    return ((v + 31u) / 32u) * 32u;
}

static uint32_t gap_byte_for_thread(uint32_t thread_id) {
    return (thread_id * 5u) / 8u;
}

// ---------------------------------------------------------------------------
// DF11Bundle: holds DRAM buffers for one compressed tensor (already on device)
// ---------------------------------------------------------------------------

struct DF11Bundle {
    std::shared_ptr<distributed::MeshBuffer> encoded_exponent;
    std::shared_ptr<distributed::MeshBuffer> sign_mantissa;
    std::shared_ptr<distributed::MeshBuffer> luts;
    std::shared_ptr<distributed::MeshBuffer> gaps;
    std::shared_ptr<distributed::MeshBuffer> output_positions;

    uint32_t k;           // number of decode LUTs
    uint32_t n;           // bytes per thread (8)
    uint32_t T;           // threads per block (512)
    uint32_t B;           // total blocks
    uint32_t R;           // original row count
    uint32_t C;           // original col count
    uint32_t R_pad;       // padded rows (multiple of 32)
    uint32_t C_pad;       // padded cols (multiple of 32)
    uint64_t n_elements;
    uint64_t n_bytes;

    uint32_t lut_total_bytes() const { return (k + 1u) * 256u; }
};

// ---------------------------------------------------------------------------
// decompress_df11: main entry point
// ---------------------------------------------------------------------------

ttnn::Tensor decompress_df11(
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    const DF11Bundle& bundle,
    uint32_t n_cores_override   // 0 = use all available cores
) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    uint32_t n_cores = grid.x * grid.y;
    if (n_cores_override > 0 && n_cores_override < n_cores)
        n_cores = n_cores_override;

    uint32_t total_threads = static_cast<uint32_t>(
        std::ceil(static_cast<double>(bundle.n_bytes) / bundle.n)
    );
    uint32_t blocks_per_core = (bundle.B + n_cores - 1) / n_cores;
    // Some cores may get one fewer block if B is not divisible by n_cores.

    // -----------------------------------------------------------------------
    // Allocate output DRAM buffer (tiled BF16 layout)
    // -----------------------------------------------------------------------
    uint32_t out_rows = bundle.R_pad;
    uint32_t out_cols = bundle.C_pad;
    uint32_t n_tiles  = (out_rows / 32u) * (out_cols / 32u);
    uint32_t out_bytes = n_tiles * TILE_BYTES;

    distributed::DeviceLocalBufferConfig out_dram_cfg{
        .page_size   = TILE_BYTES,
        .buffer_type = BufferType::DRAM,
    };
    distributed::ReplicatedBufferConfig out_buf_cfg{.size = out_bytes};
    auto out_buf = distributed::MeshBuffer::create(out_buf_cfg, out_dram_cfg, mesh_device.get());

    // -----------------------------------------------------------------------
    // Build Metalium Program
    // -----------------------------------------------------------------------
    Program program = CreateProgram();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());

    // Assign blocks to cores linearly (row-major core traversal)
    uint32_t block_cursor = 0;
    uint32_t tile_cursor  = 0;

    for (uint32_t core_i = 0; core_i < n_cores && block_cursor < bundle.B; core_i++) {
        CoreCoord core{core_i % grid.x, core_i / grid.x};

        uint32_t my_block_start = block_cursor;
        uint32_t my_block_end   = std::min(block_cursor + blocks_per_core, bundle.B);
        uint32_t my_n_blocks    = my_block_end - my_block_start;
        if (my_n_blocks == 0) continue;

        uint32_t my_thread_start = my_block_start * bundle.T;
        uint32_t my_thread_end   = my_block_end   * bundle.T;
        uint32_t my_byte_start   = my_thread_start * bundle.n;
        uint32_t my_byte_end     = std::min<uint32_t>(my_thread_end * bundle.n,
                                                      static_cast<uint32_t>(bundle.n_bytes));
        uint32_t my_elem_start   = (my_block_start < bundle.B)
                                   ? 0u  // filled from output_positions at runtime
                                   : 0u;

        // gaps: the 5-bit entries for threads [my_thread_start, my_thread_end)
        uint32_t gap_byte_start  = gap_byte_for_thread(my_thread_start);
        uint32_t gap_byte_end    = gap_byte_for_thread(my_thread_end) + 2u; // +2 for last partial
        uint32_t gap_byte_count  = gap_byte_end - gap_byte_start;

        // Compute L1 base address for this core (we use a fixed layout per core)
        // Each core uses its own L1 addresses (L1 is per-Tensix).
        uint32_t lut_l1      = L1_LUT_OFFSET;
        uint32_t gaps_l1     = L1_GAPS_OFFSET;
        uint32_t outpos_l1   = L1_OUTPOS_OFFSET;
        uint32_t scratch_l1  = L1_SCRATCH_OFFSET;
        uint32_t wbuf_l1     = L1_WBUF_OFFSET;

        // Number of tiles produced by this core
        // Approximate: we know block_start_elem from output_positions[my_block_start],
        // and block_end_elem from output_positions[my_block_end].
        // We compute this conservatively: my_n_blocks * T * n / 2 elements at most.
        // For CB sizing, use a safe upper bound.
        uint32_t max_elems_per_block = bundle.T * 8u; // 8 bytes/thread × 1 bit/elem at min (conservative)
        uint32_t max_elems_core = my_n_blocks * max_elems_per_block;
        uint32_t max_tiles_core = (max_elems_core * 2u + TILE_BYTES - 1) / TILE_BYTES + 1u;

        // ---- Circular buffers ----
        // cb_encoded: pages of CHUNK_BYTES each, double-buffered
        CircularBufferConfig cb_enc_cfg(2u * CHUNK_BYTES, {{CBIndex::c_0, DataFormat::UInt8}});
        cb_enc_cfg.set_page_size(CBIndex::c_0, CHUNK_BYTES);
        CreateCircularBuffer(program, core, cb_enc_cfg);

        // cb_signmant: pages of CHUNK_BYTES each, double-buffered
        CircularBufferConfig cb_sm_cfg(2u * CHUNK_BYTES, {{CBIndex::c_1, DataFormat::UInt8}});
        cb_sm_cfg.set_page_size(CBIndex::c_1, CHUNK_BYTES);
        CreateCircularBuffer(program, core, cb_sm_cfg);

        // cb_decoded: tiled BF16 output, hold up to 4 tiles
        CircularBufferConfig cb_dec_cfg(4u * TILE_BYTES, {{CBIndex::c_16, DataFormat::Float16_b}});
        cb_dec_cfg.set_page_size(CBIndex::c_16, TILE_BYTES);
        CreateCircularBuffer(program, core, cb_dec_cfg);

        // ---- Reader kernel ----
        std::vector<uint32_t> reader_ct{CHUNK_BYTES};
        KernelHandle reader_id = CreateKernel(
            program,
            DF11_KERNEL_PREFIX "reader_df11.cpp",
            core,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_0,
                .noc          = NOC::RISCV_0_default,
                .compile_args = reader_ct,
            });

        SetRuntimeArgs(program, reader_id, core, {
            static_cast<uint32_t>(bundle.encoded_exponent->address()),
            static_cast<uint32_t>(bundle.sign_mantissa->address()),
            static_cast<uint32_t>(bundle.luts->address()),
            static_cast<uint32_t>(bundle.gaps->address()),
            static_cast<uint32_t>(bundle.output_positions->address()),
            my_byte_start,
            my_byte_end,
            0u,  // elem start: derived from output_positions at runtime
            0u,  // elem end
            my_block_start,
            my_block_end,
            bundle.lut_total_bytes(),
            0u,  // gap_bit_start (unused — kernel uses gap_byte_start)
            gap_byte_start,
            gap_byte_count,
            lut_l1,
            gaps_l1,
            outpos_l1,
        });

        // ---- Compute kernel ----
        std::vector<uint32_t> compute_ct{
            CHUNK_BYTES,
            bundle.T,
            bundle.n,
            bundle.k,
            bundle.C,
            bundle.C_pad,
            my_n_blocks,
        };
        KernelHandle compute_id = CreateKernel(
            program,
            DF11_KERNEL_PREFIX "decompress_df11.cpp",
            core,
            ComputeConfig{
                .math_approx_mode = false,
                .compile_args     = compute_ct,
            });

        SetRuntimeArgs(program, compute_id, core, {
            lut_l1,
            gaps_l1,
            outpos_l1,
            scratch_l1,
            wbuf_l1,
            static_cast<uint32_t>(bundle.n_elements),
            my_block_start,
            my_thread_start,
        });

        // ---- Writer kernel ----
        KernelHandle writer_id = CreateKernel(
            program,
            DF11_KERNEL_PREFIX "writer_df11.cpp",
            core,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_1,
                .noc          = NOC::RISCV_1_default,
                .compile_args = {},
            });

        SetRuntimeArgs(program, writer_id, core, {
            static_cast<uint32_t>(out_buf->address()),
            tile_cursor,
            max_tiles_core,
        });

        tile_cursor   += max_tiles_core;
        block_cursor   = my_block_end;
    }

    // -----------------------------------------------------------------------
    // Launch and wait
    // -----------------------------------------------------------------------
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    // -----------------------------------------------------------------------
    // Wrap output MeshBuffer as a ttnn::Tensor (tiled BF16 layout)
    // -----------------------------------------------------------------------
    TensorSpec spec(
        Shape({bundle.R_pad, bundle.C_pad}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16,
            tt::tt_metal::PageConfig(Layout::TILE),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}
        )
    );
    std::vector<distributed::MeshCoordinate> coords{distributed::MeshCoordinate{0, 0}};
    return ttnn::Tensor(
        DeviceStorage(out_buf, std::move(coords)),
        spec,
        TensorTopology{}
    );
}

}  // namespace dfloat11_tt
