// SPDX-License-Identifier: Apache-2.0
// Python bindings for ttnn.experimental.dfloat11_decompress

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "dfloat11_decompress.hpp"

namespace nb = nanobind;

NB_MODULE(dfloat11_tt_cpp, m) {
    m.doc() = "DFloat11-TT: on-device lossless LLM decompression for Tenstorrent Blackhole";

    m.def(
        "dfloat11_decompress",
        [](const ttnn::Tensor& encoded_exponent,
           const ttnn::Tensor& sign_mantissa,
           const ttnn::Tensor& luts,
           const ttnn::Tensor& gaps,
           const ttnn::Tensor& output_positions,
           uint32_t k, uint32_t n, uint32_t T, uint32_t B,
           uint32_t R, uint32_t C, uint32_t R_pad, uint32_t C_pad,
           uint64_t n_elements, uint64_t n_bytes,
           uint32_t n_cores_override) -> ttnn::Tensor
        {
            return ttnn::dfloat11_decompress(
                encoded_exponent, sign_mantissa, luts, gaps, output_positions,
                k, n, T, B, R, C, R_pad, C_pad, n_elements, n_bytes, n_cores_override);
        },
        nb::arg("encoded_exponent"),
        nb::arg("sign_mantissa"),
        nb::arg("luts"),
        nb::arg("gaps"),
        nb::arg("output_positions"),
        nb::arg("k"),
        nb::arg("n"),
        nb::arg("T"),
        nb::arg("B"),
        nb::arg("R"),
        nb::arg("C"),
        nb::arg("R_pad"),
        nb::arg("C_pad"),
        nb::arg("n_elements"),
        nb::arg("n_bytes"),
        nb::arg("n_cores_override") = 0u,
        R"doc(
dfloat11_decompress(encoded_exponent, sign_mantissa, luts, gaps, output_positions,
                    k, n, T, B, R, C, R_pad, C_pad, n_elements, n_bytes,
                    n_cores_override=0) -> ttnn.Tensor

Decompress a DFloat11-TT bundle on-device using Tenstorrent Blackhole cores.

Returns a row-major BF16 tensor of shape (R, C) in device DRAM.
        )doc"
    );
}
