// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <variant>
#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::dfloat11 {

/**
 * DFloat11Decompress TT-NN device operation.
 *
 * Takes a compressed DFloat11-TT tensor bundle (stored as a collection of
 * raw byte tensors) and returns a tiled BF16 tensor on the same device.
 */
struct DFloat11DecompressDeviceOp {
    static constexpr uint32_t MAX_DF11_CORES = 256;

    struct operation_attributes_t {
        uint32_t k;           // number of decode LUTs
        uint32_t n;           // bytes per thread
        uint32_t T;           // threads per block
        uint32_t B;           // total blocks
        uint32_t R;           // original rows
        uint32_t C;           // original cols
        uint32_t R_pad;       // padded rows
        uint32_t C_pad;       // padded cols
        uint64_t n_elements;
        uint64_t n_bytes;
        uint32_t n_cores_override; // 0 = use all

        uint32_t n_active_ranges;
        std::vector<uint32_t> elem_starts_host;
        std::vector<uint32_t> elem_counts_host;
        std::vector<uint32_t> bit_starts_host;
    };

    struct tensor_args_t {
        const Tensor& encoded_exponent;
        const Tensor& sign_mantissa;
        const Tensor& luts;
        const Tensor& gaps;
        const Tensor& output_positions;
        const Tensor& elem_starts;
        const Tensor& elem_counts;
        const Tensor& bit_starts;
    };

    using spec_return_value_t   = ttnn::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct MultiCore {
        struct shared_variables_t {
            std::vector<tt::tt_metal::KernelHandle> reader_ids;
            std::vector<tt::tt_metal::KernelHandle> compute_ids;
            std::vector<tt::tt_metal::KernelHandle> writer_ids;
            std::vector<tt::tt_metal::CoreCoord> cores;
            std::size_t n_active_cores;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& attrs,
            const tensor_args_t& args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_program_t& cached,
            const operation_attributes_t& attrs,
            const tensor_args_t& args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<MultiCore>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(
        const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
};

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
    uint32_t n_cores_override = 0
);
}  // namespace ttnn
