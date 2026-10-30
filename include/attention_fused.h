#pragma once

#include "matmul_config.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

inline constexpr int kFusedMaxHeadDim = 128;
inline constexpr int kFusedRowThreads = kThreads / kBlockM;
static_assert(kThreads % kBlockM == 0 && kThreads % kWarpSize == 0);
static_assert(kFusedRowThreads > 0 && kFusedRowThreads <= kWarpSize &&
              (kFusedRowThreads & (kFusedRowThreads - 1)) == 0);
static_assert(kBlockN % kFusedRowThreads == 0 && kFusedMaxHeadDim % kFusedRowThreads == 0);

inline constexpr std::size_t fused_fp16_shared_bytes(int d) {
    if (d <= 0 || d > kFusedMaxHeadDim) return 0;
    const int padded_d = ((d + kTileK - 1) / kTileK) * kTileK;
    return sizeof(__half) * static_cast<std::size_t>(padded_d) * (kBlockM + 2 * kBlockN)
         + sizeof(float) * kBlockM * kBlockN;
}

inline constexpr std::size_t tilemix_routing_words_per_row(int N, int key_tiles_per_group) {
    if (N <= 0 || key_tiles_per_group <= 0) return 0;
    const std::size_t key_tiles = (static_cast<std::size_t>(N) - 1) / kBlockN + 1;
    const std::size_t groups = (key_tiles - 1) / key_tiles_per_group + 1;
    return (groups - 1) / 64 + 1;
}

inline constexpr std::size_t tilemix_routing_word_count(int M, int N, int key_tiles_per_group) {
    if (M <= 0) return 0;
    return ((static_cast<std::size_t>(M) - 1) / kBlockM + 1)
         * tilemix_routing_words_per_row(N, key_tiles_per_group);
}

struct TileMixRouting {.
    const std::uint64_t* words = nullptr;
    std::size_t word_count = 0;
    int key_tiles_per_group = 1;
    std::uint8_t* decoded_bits = nullptr;
};

cudaError_t launch_fused_fp16_attention(const __half* Q, const __half* K, const __half* V,
                                       float* O, int M, int N, int d, float score_scale,
                                       cudaStream_t stream = nullptr,
                                       const TileMixRouting& routing = {});
