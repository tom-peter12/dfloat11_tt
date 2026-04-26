// SPDX-License-Identifier: Apache-2.0
// DFloat11-TT: TT-NN device operation implementation.
//
// Wraps the multi-core Metalium Program (same logic as df11_op.cpp) behind the
// ttnn::device_operation interface so it participates in the TT-NN op graph.

#include "dfloat11_decompress.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#ifndef DF11_KERNEL_PREFIX
#define DF11_KERNEL_PREFIX "dfloat11_tt/kernels/"
#endif

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::dfloat11 {

// ---------------------------------------------------------------------------
// select_program_factory: always use MultiCore
// ---------------------------------------------------------------------------
DFloat11DecompressDeviceOp::program_factory_t
DFloat11DecompressDeviceOp::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    return MultiCore{};
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------
void DFloat11DecompressDeviceOp::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    TT_FATAL(attrs.n_elements > 0, "n_elements must be > 0");
    TT_FATAL(attrs.n_bytes   > 0, "n_bytes must be > 0");
    TT_FATAL(attrs.k > 0 && attrs.k <= 5, "k must be in [1,5]");
    TT_FATAL(args.encoded_exponent.dtype() == DataType::UINT8, "encoded_exponent must be UINT8");
    TT_FATAL(args.sign_mantissa.dtype()    == DataType::UINT8, "sign_mantissa must be UINT8");
    TT_FATAL(args.luts.dtype()             == DataType::UINT8, "luts must be UINT8");
}

// ---------------------------------------------------------------------------
// compute_output_specs: row-major BF16 tensor of shape (R, C).
//
// The validation path prioritizes bit-identical materialization.  Tiled,
// multi-core output can be reintroduced once the basic device decode is proven
// correct on hardware.
// ---------------------------------------------------------------------------
DFloat11DecompressDeviceOp::spec_return_value_t
DFloat11DecompressDeviceOp::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&)
{
    return TensorSpec(
        Shape({attrs.R, attrs.C}),
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16,
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}
        )
    );
}

// ---------------------------------------------------------------------------
// create_output_tensors
// ---------------------------------------------------------------------------
DFloat11DecompressDeviceOp::tensor_return_value_t
DFloat11DecompressDeviceOp::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    auto spec = compute_output_specs(attrs, args);
    return create_device_tensor(spec, args.encoded_exponent.device());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static constexpr uint32_t CHUNK_BYTES      = 4096u;
static constexpr uint32_t TILE_BYTES       = 2048u;

// L1 scratch layout relative to scratch_base.  scratch_base is placed after the
// statically allocated CB region for the core.
static constexpr uint32_t L1_LUT_OFFSET    = 0x0000u;  // up to 1536 bytes ((k<=5)+lens) before gaps
static constexpr uint32_t L1_GAPS_OFFSET   = 0x0600u;  // one block of packed 5-bit gaps
static constexpr uint32_t L1_OUTPOS_OFFSET = 0x1600u;  // two uint32 output positions per block
static constexpr uint32_t L1_SCRATCH_OFF   = 0x2600u;  // 512 uint32 counters / prefix values
static constexpr uint32_t L1_ENC_OFFSET    = 0x2E00u;  // one block of encoded exponent bytes
static constexpr uint32_t L1_SM_OFFSET     = 0x4000u;  // one block of sign/mantissa bytes
static constexpr uint32_t L1_WBUF_OFFSET   = 0xC000u;  // unused, kept for runtime-arg ABI stability

static uint32_t gap_byte_for_thread(uint32_t tid) { return (tid * 5u) / 8u; }

static uint32_t align_up(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
}

// ---------------------------------------------------------------------------
// MultiCore::create
// ---------------------------------------------------------------------------
DFloat11DecompressDeviceOp::MultiCore::cached_program_t
DFloat11DecompressDeviceOp::MultiCore::create(
    const operation_attributes_t& attrs,
    const tensor_args_t& args,
    tensor_return_value_t& output)
{
    auto* device = args.encoded_exponent.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t n_cores = grid.x * grid.y;
    if (attrs.n_cores_override > 0 && attrs.n_cores_override < n_cores)
        n_cores = attrs.n_cores_override;
    // Correctness-first path: one core owns the full row-major output stream.
    // This avoids cross-core page/tile ordering hazards while bring-up tests
    // prove the decompressor itself.
    n_cores = 1;

    Program program = CreateProgram();

    auto* enc_buf    = args.encoded_exponent.buffer();
    auto* sm_buf     = args.sign_mantissa.buffer();
    auto* lut_buf    = args.luts.buffer();
    auto* gaps_buf   = args.gaps.buffer();
    auto* outpos_buf = args.output_positions.buffer();
    auto* out_buf    = output.buffer();

    uint32_t blocks_per_core = (attrs.B + n_cores - 1) / n_cores;
    uint32_t block_cursor = 0;
    uint32_t page_cursor  = 0;
    uint32_t n_active = 0;

    KernelHandle reader_id  = 0;
    KernelHandle compute_id = 0;
    KernelHandle writer_id  = 0;

    for (uint32_t ci = 0; ci < n_cores && block_cursor < attrs.B; ci++) {
        CoreCoord core{ci % grid.x, ci / grid.x};

        uint32_t blk_start = block_cursor;
        uint32_t blk_end   = std::min(block_cursor + blocks_per_core, attrs.B);
        uint32_t n_blks    = blk_end - blk_start;
        if (n_blks == 0) continue;

        uint32_t thr_start   = blk_start * attrs.T;
        uint32_t thr_end     = blk_end   * attrs.T;
        uint32_t byte_start  = thr_start * attrs.n;
        uint32_t byte_end    = std::min<uint32_t>(thr_end * attrs.n,
                                                   static_cast<uint32_t>(attrs.n_bytes));
        uint32_t gsb         = gap_byte_for_thread(thr_start);
        uint32_t geb         = gap_byte_for_thread(thr_end) + 2u;
        uint32_t gbc         = geb - gsb;

        uint32_t total_output_bytes = static_cast<uint32_t>(attrs.n_elements * sizeof(uint16_t));
        uint32_t output_page_size = out_buf->page_size();
        uint32_t total_output_pages = out_buf->num_pages();
        uint32_t cb_region_bytes = 4u * output_page_size;
        uint32_t scratch_base = align_up(
            device->allocator()->get_base_allocator_addr(HalMemType::L1) + cb_region_bytes,
            32u);

        // Only cb_decoded is needed (reader produces, writer consumes)
        {
            CircularBufferConfig cfg(4u * output_page_size, {{CBIndex::c_16, DataFormat::Float16_b}});
            cfg.set_page_size(CBIndex::c_16, output_page_size);
            CreateCircularBuffer(program, core, cfg);
        }

        // Reader (BRISC) — does all decode work
        std::vector<uint32_t> reader_ct{
            CHUNK_BYTES, attrs.T, attrs.n, attrs.k,
            attrs.C, attrs.C_pad, n_blks, output_page_size,
            static_cast<uint32_t>(enc_buf->page_size()),
            static_cast<uint32_t>(sm_buf->page_size()),
            static_cast<uint32_t>(lut_buf->page_size()),
            static_cast<uint32_t>(gaps_buf->page_size()),
            static_cast<uint32_t>(outpos_buf->page_size()),
        };
        TensorAccessorArgs(*enc_buf).append_to(reader_ct);
        TensorAccessorArgs(*sm_buf).append_to(reader_ct);
        TensorAccessorArgs(*lut_buf).append_to(reader_ct);
        TensorAccessorArgs(*gaps_buf).append_to(reader_ct);
        TensorAccessorArgs(*outpos_buf).append_to(reader_ct);
        auto rid = CreateKernel(program, DF11_KERNEL_PREFIX "reader_df11.cpp", core,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_0,
                .noc          = NOC::RISCV_0_default,
                .compile_args = reader_ct,
            });
        if (ci == 0) reader_id = rid;

        SetRuntimeArgs(program, rid, core, {
            enc_buf->address(), sm_buf->address(), lut_buf->address(),
            gaps_buf->address(), outpos_buf->address(),
            byte_start, byte_end, 0u, 0u,
            blk_start, blk_end,
            (attrs.k + 1u) * 256u, 0u,
            gsb, gbc,
            scratch_base + L1_LUT_OFFSET,
            scratch_base + L1_GAPS_OFFSET,
            scratch_base + L1_OUTPOS_OFFSET,
            scratch_base + L1_SCRATCH_OFF,
            scratch_base + L1_ENC_OFFSET,
            scratch_base + L1_SM_OFFSET,
            scratch_base + L1_WBUF_OFFSET,
            static_cast<uint32_t>(attrs.n_elements),
            total_output_pages,
        });

        // Compute (TRISC) — blank, no-op
        auto cid = CreateKernel(program, DF11_KERNEL_PREFIX "decompress_df11.cpp", core,
            ComputeConfig{
                .math_approx_mode = false,
                .compile_args = {},
            });
        if (ci == 0) compute_id = cid;

        // Writer (NCRISC)
        std::vector<uint32_t> writer_ct{output_page_size};
        TensorAccessorArgs(*out_buf).append_to(writer_ct);
        auto wid = CreateKernel(program, DF11_KERNEL_PREFIX "writer_df11.cpp", core,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_1,
                .noc          = NOC::RISCV_1_default,
                .compile_args = writer_ct,
            });
        if (ci == 0) writer_id = wid;

        SetRuntimeArgs(program, wid, core, {
            out_buf->address(),
            page_cursor,
            total_output_pages,
            total_output_bytes,
        });

        page_cursor   += total_output_pages;
        block_cursor   = blk_end;
        n_active++;
    }

    return {
        std::move(program),
        {.reader_id  = reader_id,
         .compute_id = compute_id,
         .writer_id  = writer_id,
         .n_active_cores = n_active},
    };
}

// ---------------------------------------------------------------------------
// MultiCore::override_runtime_arguments (for cache hits)
// ---------------------------------------------------------------------------
void DFloat11DecompressDeviceOp::MultiCore::override_runtime_arguments(
    cached_program_t& cached,
    const operation_attributes_t& attrs,
    const tensor_args_t& args,
    tensor_return_value_t& output)
{
    // Update source and destination buffer addresses for the first core's reader/writer.
    // A full re-sweep over all cores would be needed for truly different bundles,
    // but for same-shape bundles (the common case with model caching) only the
    // addresses change.
    auto& prog = cached.program;
    auto& sv   = cached.shared_variables;

    auto* device = args.encoded_exponent.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t n_cores = std::min<uint32_t>(grid.x * grid.y, static_cast<uint32_t>(sv.n_active_cores));
    n_cores = std::min<uint32_t>(n_cores, 1u);

    uint32_t blocks_per_core = (attrs.B + n_cores - 1) / n_cores;
    uint32_t block_cursor = 0;
    uint32_t page_cursor  = 0;

    for (uint32_t ci = 0; ci < n_cores && block_cursor < attrs.B; ci++) {
        CoreCoord core{ci % grid.x, ci / grid.x};
        uint32_t blk_start = block_cursor;
        uint32_t blk_end   = std::min(block_cursor + blocks_per_core, attrs.B);
        uint32_t n_blks    = blk_end - blk_start;
        if (n_blks == 0) continue;

        uint32_t thr_start  = blk_start * attrs.T;
        uint32_t thr_end    = blk_end   * attrs.T;
        uint32_t byte_start = thr_start * attrs.n;
        uint32_t byte_end   = std::min<uint32_t>(thr_end * attrs.n, static_cast<uint32_t>(attrs.n_bytes));
        uint32_t gsb        = gap_byte_for_thread(thr_start);
        uint32_t gbc        = gap_byte_for_thread(thr_end) + 2u - gsb;
        uint32_t total_output_bytes = static_cast<uint32_t>(attrs.n_elements * sizeof(uint16_t));
        uint32_t total_output_pages = output.buffer()->num_pages();

        {
            auto& ra = GetRuntimeArgs(prog, sv.reader_id, core);
            ra[0] = args.encoded_exponent.buffer()->address();
            ra[1] = args.sign_mantissa.buffer()->address();
            ra[2] = args.luts.buffer()->address();
            ra[3] = args.gaps.buffer()->address();
            ra[4] = args.output_positions.buffer()->address();
            ra[5] = byte_start; ra[6] = byte_end;
            ra[9] = blk_start;  ra[10] = blk_end;
            ra[11] = (attrs.k + 1u) * 256u;
            ra[13] = gsb;  ra[14] = gbc;
            ra[22] = static_cast<uint32_t>(attrs.n_elements);
            ra[23] = total_output_pages;
        }
        // compute kernel is blank — no runtime args to update
        {
            auto& ra = GetRuntimeArgs(prog, sv.writer_id, core);
            ra[0] = output.buffer()->address();
            ra[1] = page_cursor;
            ra[2] = total_output_pages;
            ra[3] = total_output_bytes;
        }

        page_cursor  += total_output_pages;
        block_cursor  = blk_end;
    }
}

}  // namespace ttnn::operations::dfloat11

// ---------------------------------------------------------------------------
// Top-level ttnn::dfloat11_decompress
// ---------------------------------------------------------------------------
namespace ttnn {

Tensor dfloat11_decompress(
    const Tensor& encoded_exponent,
    const Tensor& sign_mantissa,
    const Tensor& luts,
    const Tensor& gaps,
    const Tensor& output_positions,
    uint32_t k, uint32_t n, uint32_t T, uint32_t B,
    uint32_t R, uint32_t C, uint32_t R_pad, uint32_t C_pad,
    uint64_t n_elements, uint64_t n_bytes,
    uint32_t n_cores_override)
{
    using namespace ttnn::operations::dfloat11;
    return ttnn::device_operation::launch<DFloat11DecompressDeviceOp>(
        DFloat11DecompressDeviceOp::operation_attributes_t{
            k, n, T, B, R, C, R_pad, C_pad, n_elements, n_bytes, n_cores_override},
        DFloat11DecompressDeviceOp::tensor_args_t{
            encoded_exponent, sign_mantissa, luts, gaps, output_positions}
    );
}

}  // namespace ttnn
