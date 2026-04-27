// SPDX-License-Identifier: Apache-2.0
// DFloat11-TT: TT-NN device operation implementation.
//
// Real multicore implementation:
//   - Distributes DFloat11 blocks across available Tensix cores
//   - Each core owns a contiguous range of blocks and writes its output
//     to the correct pages in the interleaved DRAM output tensor
//   - Output is row-major BF16

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

#ifndef DF11_ENABLE_METRICS
#define DF11_ENABLE_METRICS 1
#endif

#if DF11_ENABLE_METRICS
#include <chrono>
#include <cstdio>

struct DecompressMetrics {
    uint32_t n_elements;
    uint32_t n_bytes;
    uint32_t n_blocks;
    uint32_t n_cores_used;
    uint32_t blocks_per_core_avg;
    uint32_t blocks_per_core_max;
    uint32_t output_pages_total;
    double   program_create_ms;
    uint32_t R, C, R_pad, C_pad;
    uint32_t k;
};

static void log_metrics(const DecompressMetrics& m) {
    fprintf(stderr,
        "[df11-metrics] elements=%u bytes=%u blocks=%u cores=%u "
        "blk/core_avg=%u blk/core_max=%u out_pages=%u "
        "shape=(%u,%u) padded=(%u,%u) k=%u "
        "program_create=%.2fms\n",
        m.n_elements, m.n_bytes, m.n_blocks, m.n_cores_used,
        m.blocks_per_core_avg, m.blocks_per_core_max, m.output_pages_total,
        m.R, m.C, m.R_pad, m.C_pad, m.k,
        m.program_create_ms);
}
#endif

DFloat11DecompressDeviceOp::program_factory_t
DFloat11DecompressDeviceOp::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&)
{
    return MultiCore{};
}

void DFloat11DecompressDeviceOp::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    TT_FATAL(attrs.n_elements > 0, "n_elements must be > 0");
    TT_FATAL(attrs.n_bytes   > 0, "n_bytes must be > 0");
    TT_FATAL(attrs.k > 0 && attrs.k <= 5, "k must be in [1,5]");
    TT_FATAL(args.encoded_exponent.dtype() == DataType::UINT8, "encoded_exponent must be UINT8");
    TT_FATAL(args.sign_mantissa.dtype()    == DataType::UINT8, "sign_mantissa must be UINT8");
    TT_FATAL(args.luts.dtype()             == DataType::UINT8, "luts must be UINT8");
    TT_FATAL(args.elem_starts.dtype()      == DataType::UINT32, "elem_starts must be UINT32");
    TT_FATAL(args.elem_counts.dtype()      == DataType::UINT32, "elem_counts must be UINT32");
    TT_FATAL(args.bit_starts.dtype()       == DataType::UINT32, "bit_starts must be UINT32");
    TT_FATAL(attrs.n_active_ranges > 0, "n_active_ranges must be > 0");
    TT_FATAL(
        attrs.n_active_ranges <= DFloat11DecompressDeviceOp::MAX_DF11_CORES,
        "n_active_ranges exceeds MAX_DF11_CORES");
}

void DFloat11DecompressDeviceOp::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    validate_on_program_cache_miss(attrs, args);
}

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

DFloat11DecompressDeviceOp::tensor_return_value_t
DFloat11DecompressDeviceOp::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args)
{
    auto spec = compute_output_specs(attrs, args);
    return create_device_tensor(spec, args.encoded_exponent.device());
}

static constexpr uint32_t CHUNK_BYTES      = 4096u;

static uint32_t gap_byte_for_thread(uint32_t tid) { return (tid * 5u) / 8u; }

static uint32_t align_up(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
}

static constexpr uint32_t L1_LUT_OFFSET    = 0x0000u;
static constexpr uint32_t L1_GAPS_OFFSET   = 0x0600u;
static constexpr uint32_t L1_OUTPOS_OFFSET = 0x1600u;
static constexpr uint32_t L1_SCRATCH_OFF   = 0x2600u;
static constexpr uint32_t L1_ENC_OFFSET    = 0x2E00u;
static constexpr uint32_t L1_SM_OFFSET     = 0x4000u;
static constexpr uint32_t L1_WBUF_OFFSET   = 0xC000u;

DFloat11DecompressDeviceOp::MultiCore::cached_program_t
DFloat11DecompressDeviceOp::MultiCore::create(
    const operation_attributes_t& attrs,
    const tensor_args_t& args,
    tensor_return_value_t& output)
{
#if DF11_ENABLE_METRICS
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    auto* device = args.encoded_exponent.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t max_cores = grid.x * grid.y;

    uint32_t n_cores = max_cores;
    if (attrs.n_cores_override > 0 && attrs.n_cores_override < n_cores)
        n_cores = attrs.n_cores_override;
    if (n_cores > attrs.B) n_cores = attrs.B;
    if (n_cores == 0) n_cores = 1;
    if (attrs.n_active_ranges > 0 && attrs.n_active_ranges < n_cores)
        n_cores = attrs.n_active_ranges;

    Program program = CreateProgram();

    auto* enc_buf    = args.encoded_exponent.buffer();
    auto* sm_buf     = args.sign_mantissa.buffer();
    auto* lut_buf    = args.luts.buffer();
    auto* gaps_buf   = args.gaps.buffer();
    auto* outpos_buf = args.output_positions.buffer();
    auto* out_buf    = output.buffer();

    uint32_t total_output_bytes = static_cast<uint32_t>(attrs.n_elements * sizeof(uint16_t));
    uint32_t output_page_size = out_buf->page_size();
    uint32_t total_output_pages = out_buf->num_pages();

    uint32_t blocks_per_core_base = attrs.B / n_cores;
    uint32_t remainder_blocks = attrs.B % n_cores;

    uint32_t block_cursor = 0;
    uint32_t n_active = 0;

    std::vector<KernelHandle> reader_ids;
    std::vector<KernelHandle> compute_ids;
    std::vector<KernelHandle> writer_ids;
    std::vector<CoreCoord> active_cores;
    reader_ids.reserve(n_cores);
    compute_ids.reserve(n_cores);
    writer_ids.reserve(n_cores);
    active_cores.reserve(n_cores);

#if DF11_ENABLE_METRICS
    uint32_t max_blks_any_core = 0;
#endif

    for (uint32_t ci = 0; ci < n_cores && block_cursor < attrs.B; ci++) {
        CoreCoord core{ci % grid.x, ci / grid.x};

        uint32_t n_blks = blocks_per_core_base + (ci < remainder_blocks ? 1u : 0u);
        if (n_blks == 0) continue;

        uint32_t blk_start = block_cursor;
        uint32_t blk_end   = block_cursor + n_blks;
        if (blk_end > attrs.B) blk_end = attrs.B;
        n_blks = blk_end - blk_start;

#if DF11_ENABLE_METRICS
        if (n_blks > max_blks_any_core) max_blks_any_core = n_blks;
#endif

        uint32_t thr_start   = blk_start * attrs.T;
        uint32_t thr_end     = blk_end   * attrs.T;
        uint32_t byte_start  = thr_start * attrs.n;
        uint32_t byte_end    = std::min<uint32_t>(thr_end * attrs.n,
                                                   static_cast<uint32_t>(attrs.n_bytes));

        uint32_t gsb = gap_byte_for_thread(thr_start);
        uint32_t geb = gap_byte_for_thread(thr_end) + 2u;
        uint32_t gbc = geb - gsb;

        uint32_t core_elem_start = attrs.elem_starts_host[ci];
        uint32_t core_bit_start  = attrs.bit_starts_host[ci];

        uint64_t elem_start = static_cast<uint64_t>(core_elem_start);
        uint64_t elem_count = static_cast<uint64_t>(attrs.elem_counts_host[ci]);
        uint64_t elem_end   = elem_start + elem_count;

        if (elem_end > attrs.n_elements) elem_end = attrs.n_elements;
        if (elem_start > elem_end) elem_start = elem_end;

        uint32_t page_start = static_cast<uint32_t>((elem_start * 2u) / output_page_size);
        uint32_t page_end   = static_cast<uint32_t>(
            ((elem_end * 2u) + output_page_size - 1u) / output_page_size);
        if (page_end > total_output_pages) page_end = total_output_pages;

        uint32_t n_pages_core = (page_end > page_start) ? (page_end - page_start) : 0u;

        uint32_t cb_region_bytes = 4u * output_page_size;
        uint32_t scratch_base = align_up(
            device->allocator()->get_base_allocator_addr(HalMemType::L1) + cb_region_bytes,
            32u);

        {
            CircularBufferConfig cfg(4u * output_page_size,
                                     {{CBIndex::c_16, DataFormat::Float16_b}});
            cfg.set_page_size(CBIndex::c_16, output_page_size);
            CreateCircularBuffer(program, core, cfg);
        }

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
        reader_ids.push_back(rid);

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
            n_pages_core,
            core_elem_start,
            core_bit_start,
        });

        auto cid = CreateKernel(program, DF11_KERNEL_PREFIX "decompress_df11.cpp", core,
            ComputeConfig{
                .math_approx_mode = false,
                .compile_args = {},
            });
        compute_ids.push_back(cid);

        std::vector<uint32_t> writer_ct{output_page_size};
        TensorAccessorArgs(*out_buf).append_to(writer_ct);
        auto wid = CreateKernel(program, DF11_KERNEL_PREFIX "writer_df11.cpp", core,
            DataMovementConfig{
                .processor    = DataMovementProcessor::RISCV_1,
                .noc          = NOC::RISCV_1_default,
                .compile_args = writer_ct,
            });
        writer_ids.push_back(wid);
        active_cores.push_back(core);

        SetRuntimeArgs(program, wid, core, {
            out_buf->address(),
            page_start,
            n_pages_core,
            total_output_bytes,
        });

        block_cursor = blk_end;
        n_active++;
    }

#if DF11_ENABLE_METRICS
    auto t_end = std::chrono::high_resolution_clock::now();
    double create_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    log_metrics({
        .n_elements = static_cast<uint32_t>(attrs.n_elements),
        .n_bytes = static_cast<uint32_t>(attrs.n_bytes),
        .n_blocks = attrs.B,
        .n_cores_used = n_active,
        .blocks_per_core_avg = n_active > 0 ? attrs.B / n_active : 0,
        .blocks_per_core_max = max_blks_any_core,
        .output_pages_total = total_output_pages,
        .program_create_ms = create_ms,
        .R = attrs.R, .C = attrs.C,
        .R_pad = attrs.R_pad, .C_pad = attrs.C_pad,
        .k = attrs.k,
    });
#endif

    return {
        std::move(program),
        {.reader_ids = std::move(reader_ids),
         .compute_ids = std::move(compute_ids),
         .writer_ids = std::move(writer_ids),
         .cores = std::move(active_cores),
         .n_active_cores = n_active},
    };
}

void DFloat11DecompressDeviceOp::MultiCore::override_runtime_arguments(
    cached_program_t& cached,
    const operation_attributes_t& attrs,
    const tensor_args_t& args,
    tensor_return_value_t& output)
{
    auto& prog = cached.program;
    auto& sv   = cached.shared_variables;

    auto* device = args.encoded_exponent.device();
    CoreCoord grid = device->compute_with_storage_grid_size();
    uint32_t max_cores = grid.x * grid.y;
    uint32_t n_cores = std::min<uint32_t>(max_cores, static_cast<uint32_t>(sv.n_active_cores));
    if (attrs.n_cores_override > 0 && attrs.n_cores_override < n_cores)
        n_cores = attrs.n_cores_override;
    if (n_cores > attrs.B) n_cores = attrs.B;
    if (n_cores == 0) n_cores = 1;
    if (attrs.n_active_ranges > 0 && attrs.n_active_ranges < n_cores)
        n_cores = attrs.n_active_ranges;

    uint32_t total_output_bytes = static_cast<uint32_t>(attrs.n_elements * sizeof(uint16_t));
    uint32_t output_page_size = output.buffer()->page_size();
    uint32_t total_output_pages = output.buffer()->num_pages();
    uint32_t blocks_per_core_base = attrs.B / n_cores;
    uint32_t remainder_blocks = attrs.B % n_cores;

    uint32_t block_cursor = 0;

    TT_FATAL(sv.reader_ids.size() >= n_cores, "reader_ids size mismatch");
    TT_FATAL(sv.writer_ids.size() >= n_cores, "writer_ids size mismatch");
    TT_FATAL(sv.cores.size() >= n_cores, "cores size mismatch");

    for (uint32_t ci = 0; ci < n_cores && block_cursor < attrs.B; ci++) {
        CoreCoord core = sv.cores[ci];

        uint32_t n_blks = blocks_per_core_base + (ci < remainder_blocks ? 1u : 0u);
        uint32_t blk_start = block_cursor;
        uint32_t blk_end   = std::min(block_cursor + n_blks, attrs.B);
        n_blks = blk_end - blk_start;
        if (n_blks == 0) continue;

        uint32_t thr_start  = blk_start * attrs.T;
        uint32_t thr_end    = blk_end   * attrs.T;
        uint32_t byte_start = thr_start * attrs.n;
        uint32_t byte_end   = std::min<uint32_t>(
            thr_end * attrs.n,
            static_cast<uint32_t>(attrs.n_bytes));

        uint32_t gsb = gap_byte_for_thread(thr_start);
        uint32_t geb = gap_byte_for_thread(thr_end) + 2u;
        uint32_t gbc = geb - gsb;

        uint32_t core_elem_start = attrs.elem_starts_host[ci];
        uint32_t core_bit_start  = attrs.bit_starts_host[ci];

        uint64_t elem_start = static_cast<uint64_t>(core_elem_start);
        uint64_t elem_count = static_cast<uint64_t>(attrs.elem_counts_host[ci]);
        uint64_t elem_end   = elem_start + elem_count;

        if (elem_end > attrs.n_elements) elem_end = attrs.n_elements;
        if (elem_start > elem_end) elem_start = elem_end;

        uint32_t page_start = static_cast<uint32_t>((elem_start * 2u) / output_page_size);
        uint32_t page_end   = static_cast<uint32_t>(
            ((elem_end * 2u) + output_page_size - 1u) / output_page_size);
        if (page_end > total_output_pages) page_end = total_output_pages;

        uint32_t n_pages_core = (page_end > page_start) ? (page_end - page_start) : 0u;

        {
            auto& ra = GetRuntimeArgs(prog, sv.reader_ids[ci], core);
            ra[0] = args.encoded_exponent.buffer()->address();
            ra[1] = args.sign_mantissa.buffer()->address();
            ra[2] = args.luts.buffer()->address();
            ra[3] = args.gaps.buffer()->address();
            ra[4] = args.output_positions.buffer()->address();
            ra[5] = byte_start;
            ra[6] = byte_end;
            ra[9] = blk_start;
            ra[10] = blk_end;
            ra[11] = (attrs.k + 1u) * 256u;
            ra[13] = gsb;
            ra[14] = gbc;
            ra[22] = static_cast<uint32_t>(attrs.n_elements);
            ra[23] = n_pages_core;
            ra[24] = core_elem_start;
            ra[25] = core_bit_start;
        }
        {
            auto& ra = GetRuntimeArgs(prog, sv.writer_ids[ci], core);
            ra[0] = output.buffer()->address();
            ra[1] = page_start;
            ra[2] = n_pages_core;
            ra[3] = total_output_bytes;
        }

        block_cursor = blk_end;
    }
}

}  // namespace ttnn::operations::dfloat11

namespace ttnn {

Tensor dfloat11_decompress(
    const Tensor& encoded_exponent,
    const Tensor& sign_mantissa,
    const Tensor& luts,
    const Tensor& gaps,
    const Tensor& output_positions,
    const Tensor& elem_starts,
    const Tensor& elem_counts,
    const Tensor& bit_starts,
    const std::vector<uint32_t>& elem_starts_host,
    const std::vector<uint32_t>& elem_counts_host,
    const std::vector<uint32_t>& bit_starts_host,
    uint32_t k, uint32_t n, uint32_t T, uint32_t B,
    uint32_t R, uint32_t C, uint32_t R_pad, uint32_t C_pad,
    uint64_t n_elements, uint64_t n_bytes,
    uint32_t n_cores_override)
{
    using namespace ttnn::operations::dfloat11;

    DFloat11DecompressDeviceOp::operation_attributes_t attrs{
        k, n, T, B, R, C, R_pad, C_pad, n_elements, n_bytes, n_cores_override
    };

    uint32_t n_ranges = static_cast<uint32_t>(elem_starts_host.size());
    if (static_cast<uint32_t>(elem_counts_host.size()) < n_ranges) {
        n_ranges = static_cast<uint32_t>(elem_counts_host.size());
    }
    if (static_cast<uint32_t>(bit_starts_host.size()) < n_ranges) {
        n_ranges = static_cast<uint32_t>(bit_starts_host.size());
    }
    if (n_ranges > DFloat11DecompressDeviceOp::MAX_DF11_CORES) {
        n_ranges = DFloat11DecompressDeviceOp::MAX_DF11_CORES;
    }

    attrs.n_active_ranges = n_ranges;
    for (uint32_t i = 0; i < n_ranges; ++i) {
        attrs.elem_starts_host[i] = elem_starts_host[i];
        attrs.elem_counts_host[i] = elem_counts_host[i];
        attrs.bit_starts_host[i] = bit_starts_host[i];
    }

    return ttnn::device_operation::launch<DFloat11DecompressDeviceOp>(
        attrs,
        DFloat11DecompressDeviceOp::tensor_args_t{
            encoded_exponent,
            sign_mantissa,
            luts,
            gaps,
            output_positions,
            elem_starts,
            elem_counts,
            bit_starts
        }
    );
}

}  // namespace ttnn