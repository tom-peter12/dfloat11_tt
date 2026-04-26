// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include <tt-metalium/distributed.hpp>
#include <ttnn/tensor/tensor.hpp>

namespace dfloat11_tt {

struct DF11Bundle;  // forward decl (defined in df11_op.cpp)

/**
 * Decompress a single DFloat11-TT bundle on-device.
 *
 * @param mesh_device  Tenstorrent device handle.
 * @param bundle       Compressed tensor metadata + DRAM buffers.
 * @param n_cores_override  If >0, use at most this many cores (for profiling).
 * @return  A tiled BF16 tensor in device DRAM, ready for ttnn::linear.
 */
ttnn::Tensor decompress_df11(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device,
    const DF11Bundle& bundle,
    uint32_t n_cores_override = 0
);

}  // namespace dfloat11_tt
