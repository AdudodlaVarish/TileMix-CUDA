#include "attention_fused.h"

#include <math_constants.h>
#include <mma.h>

#include <algorithm>
#include <cmath>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "FP16 WMMA attention requires sm_70 or newer."
#endif

namespace {
namespace wmma = nvcuda::wmma;

__global__ void fused_fp16_attention(const __half* Q, const __half* K, const __half* V,
                                    float* O, int M, int N, int d, int padded_d,
                                    float score_scale) {
    extern __shared__ __align__(32) unsigned char storage[];
    __half* q_tile = reinterpret_cast<__half*>(storage);
    __half* k_tile = q_tile + kBlockM * padded_d;
    __half* v_tile = k_tile + kBlockN * padded_d;
    float* weights = reinterpret_cast<float*>(v_tile + kBlockN * padded_d);

    const int thread = threadIdx.x;
    const int warp_key_begin = (thread / kWarpSize) * 16;
    const std::size_t query_begin = static_cast<std::size_t>(blockIdx.x) * kBlockM;
    const int local_query = thread / kFusedRowThreads;
    const int row_lane = thread % kFusedRowThreads;
    const std::size_t query = query_begin + local_query;
    const bool query_valid = query < static_cast<std::size_t>(M);

    constexpr int kOutputSlots = kFusedMaxHeadDim / kFusedRowThreads;
    constexpr int kScoreSlots = kBlockN / kFusedRowThreads;
    float output[kOutputSlots];
    #pragma unroll
    for (int slot = 0; slot < kOutputSlots; ++slot) output[slot] = 0.0f;
    float running_maximum = -CUDART_INF_F;
    float denominator = 0.0f;

    for (int index = thread; index < kBlockM * padded_d; index += kThreads) {
        const int feature = index % padded_d;
        const std::size_t q = query_begin + index / padded_d;
        q_tile[index] = q < static_cast<std::size_t>(M) && feature < d
            ? Q[q * d + feature] : __float2half(0.0f);
    }
    __syncthreads();

    for (std::size_t key_begin = 0; key_begin < static_cast<std::size_t>(N);
         key_begin += kBlockN) {
        for (int index = thread; index < kBlockN * padded_d; index += kThreads) {
            const int feature = index % padded_d;
            const std::size_t key = key_begin + index / padded_d;
            const bool valid = key < static_cast<std::size_t>(N) && feature < d;
            k_tile[index] = valid ? K[key * d + feature] : __float2half(0.0f);
            v_tile[index] = valid ? V[key * d + feature] : __float2half(0.0f);
        }
        __syncthreads();
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> scores;
            wmma::fill_fragment(scores, 0.0f);
            for (int feature = 0; feature < padded_d; feature += kTileK) {
                wmma::load_matrix_sync(a, q_tile + feature, padded_d);
                wmma::load_matrix_sync(b, k_tile + warp_key_begin * padded_d + feature, padded_d);
                wmma::mma_sync(scores, a, b, scores);
            }
            
            wmma::store_matrix_sync(weights + warp_key_begin, scores, kBlockN,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        float row_scores[kScoreSlots];
        float tile_maximum = -CUDART_INF_F;
        #pragma unroll
        for (int slot = 0; slot < kScoreSlots; ++slot) {
            const int key = row_lane + slot * kFusedRowThreads;
            row_scores[slot] = query_valid && key_begin + key < static_cast<std::size_t>(N)
                ? weights[local_query * kBlockN + key] * score_scale : -CUDART_INF_F;
            tile_maximum = fmaxf(tile_maximum, row_scores[slot]);
        }
        
        #pragma unroll
        for (int offset = kFusedRowThreads / 2; offset > 0; offset /= 2) {
            tile_maximum = fmaxf(tile_maximum,
                __shfl_xor_sync(0xffffffffu, tile_maximum, offset, kFusedRowThreads));
        }
        const bool has_values = tile_maximum != -CUDART_INF_F;
        const float next_maximum = fmaxf(running_maximum, tile_maximum);
        const float alpha = has_values ? expf(running_maximum - next_maximum) : 1.0f;
        float tile_sum = 0.0f;
        #pragma unroll
        for (int slot = 0; slot < kScoreSlots; ++slot) {
            const int key = row_lane + slot * kFusedRowThreads;
            const float p = has_values ? expf(row_scores[slot] - next_maximum) : 0.0f;
            weights[local_query * kBlockN + key] = p;
            tile_sum += p;
        }
        #pragma unroll
        for (int offset = kFusedRowThreads / 2; offset > 0; offset /= 2) {
            tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, offset, kFusedRowThreads);
        }
        denominator = fmaf(alpha, denominator, tile_sum);
        running_maximum = next_maximum;
        __syncthreads();

        if (query_valid) {
            #pragma unroll
            for (int slot = 0; slot < kOutputSlots; ++slot) {
                const int feature = row_lane + slot * kFusedRowThreads;
                if (feature < d) {
                    float sum = alpha * output[slot];
                    #pragma unroll 1
                    for (int key = 0; key < kBlockN; ++key) {
                        sum = fmaf(weights[local_query * kBlockN + key],
                                   __half2float(v_tile[key * padded_d + feature]), sum);
                    }
                    output[slot] = sum;
                }
            }
        }
        __syncthreads();
    }

    if (query_valid) {
        #pragma unroll
        for (int slot = 0; slot < kOutputSlots; ++slot) {
            const int feature = row_lane + slot * kFusedRowThreads;
            if (feature < d) O[query * d + feature] = output[slot] / denominator;
        }
    }
}
}  // namespace

cudaError_t launch_fused_fp16_attention(const __half* Q, const __half* K, const __half* V,
                                       float* O, int M, int N, int d, float score_scale,
                                       cudaStream_t stream) {
    if (!Q || !K || !V || !O || M <= 0 || N <= 0 || d <= 0 || !std::isfinite(score_scale)) {
        return cudaErrorInvalidValue;
    }
    if (d > kFusedMaxHeadDim) return cudaErrorNotSupported;
    int device;
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) return status;
    if (properties.major < 7) return cudaErrorNotSupported;
    const unsigned query_tiles = static_cast<unsigned>((M - 1) / kBlockM + 1);
    const std::size_t shared_bytes = fused_fp16_shared_bytes(d);
    if (query_tiles > static_cast<unsigned>(properties.maxGridSize[0]) ||
        kThreads > properties.maxThreadsPerBlock ||
        shared_bytes > std::max(properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin)) {
        return cudaErrorInvalidConfiguration;
    }
    if (shared_bytes > properties.sharedMemPerBlock) {
        status = cudaFuncSetAttribute(fused_fp16_attention,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
        if (status != cudaSuccess) return status;
    }
    const int padded_d = ((d + kTileK - 1) / kTileK) * kTileK;
    fused_fp16_attention<<<query_tiles, kThreads, shared_bytes, stream>>>(
        Q, K, V, O, M, N, d, padded_d, score_scale);
    return cudaGetLastError();
}
