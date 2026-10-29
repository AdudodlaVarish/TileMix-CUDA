#pragma once

#include "matmul_config.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

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

cudaError_t launch_fused_fp16_attention(const __half* Q, const __half* K, const __half* V,
                                       float* O, int M, int N, int d, float score_scale,
                                       cudaStream_t stream = nullptr);
