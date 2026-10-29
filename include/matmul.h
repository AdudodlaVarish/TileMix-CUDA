#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

__global__ void tiled_matmul(const __half* Q, const __half* K, float* S,
                             int M, int N, int d, float score_scale,
                             const uint64_t* routing_words);
