#include "attention_fused.h"

#include <math_constants.h>
#include <mma.h>

#include <algorithm>
#include <cmath>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#error "FP16/INT8 WMMA TileMix requires sm_75 or newer."
#endif

namespace {
namespace wmma = nvcuda::wmma;

template <int BlockRows>
__global__ void quantize_blocks(const __half* input, std::int8_t* output, __half* scales,
                                int rows, int d) {
    constexpr int threads = 256;
    __shared__ float maximum[threads];
    const int thread = threadIdx.x;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * BlockRows;
    float value_max = 0.0f;
    for (int index = thread; index < BlockRows * d; index += threads) {
        const std::size_t row = first + index / d;
        if (row < static_cast<std::size_t>(rows)) {
            value_max = fmaxf(value_max, fabsf(__half2float(input[row * d + index % d])));
        }
    }
    maximum[thread] = value_max;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride /= 2) {
        if (thread < stride) maximum[thread] = fmaxf(maximum[thread], maximum[thread + stride]);
        __syncthreads();
    }
    // Quantize with the FP32 scale; only the reusable dequantization scale is FP16.
    const float scale = __fdiv_rn(maximum[0], 127.0f);
    if (thread == 0) scales[blockIdx.x] = __float2half_rn(scale);
    for (int index = thread; index < BlockRows * d; index += threads) {
        const std::size_t row = first + index / d;
        if (row < static_cast<std::size_t>(rows)) {
            const std::size_t offset = row * d + index % d;
            const float x = __half2float(input[offset]);
            const float normalized = scale > 0.0f ? __fdiv_rn(x, scale) : 0.0f;
            const int rounded = __float2int_rz(normalized + copysignf(0.5f, normalized));
            output[offset] = static_cast<std::int8_t>(max(-127, min(127, rounded)));
        }
    }
}

__global__ void fused_fp16_attention(const __half* Q, const __half* K, const __half* V,
                                    float* O, int M, int N, int d, int padded_d,
                                    float score_scale, TileMixRouting routing,
                                    std::size_t routing_words_per_row, std::size_t key_tiles,
                                    TileMixQuantized quantized) {
    extern __shared__ __align__(32) unsigned char storage[];
    __half* q_tile = reinterpret_cast<__half*>(storage);
    __half* k_tile = q_tile + kBlockM * padded_d;
    __half* v_tile = k_tile + kBlockN * padded_d;
    auto* q8_tile = reinterpret_cast<std::int8_t*>(v_tile + kBlockN * padded_d);
    auto* k8_tile = q8_tile + kBlockM * padded_d;
    float* weights = reinterpret_cast<float*>(quantized.q ? k8_tile + kBlockN * padded_d : q8_tile);

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
        if (quantized.q) {
            const int local_row = index / padded_d;
            // Store feature chunks [d/16, row, 16] so every INT8 WMMA load
            // starts at a 32-byte-aligned address, including feature step 16.
            const int packed = (feature / 16) * kBlockM * 16 + local_row * 16 + feature % 16;
            q8_tile[packed] = q < static_cast<std::size_t>(M) && feature < d
                ? quantized.q[q * d + feature] : 0;
        }
    }
    __syncthreads();

    for (std::size_t key_begin = 0; key_begin < static_cast<std::size_t>(N);
         key_begin += kBlockN) {
        
        const std::size_t key_tile = key_begin / kBlockN;
        const std::size_t group = key_tile / routing.key_tiles_per_group;
        const std::size_t mask_index = static_cast<std::size_t>(blockIdx.x) * routing_words_per_row + group / 64;
        const std::uint64_t word = routing.words ? routing.words[mask_index] : std::uint64_t{0};
        const bool int8_requested = ((word >> (group % 64)) & std::uint64_t{1}) != 0;
        if (thread == 0 && routing.decoded_bits) {
            routing.decoded_bits[static_cast<std::size_t>(blockIdx.x) * key_tiles + key_tile] = int8_requested;
        }
        
        for (int index = thread; index < kBlockN * padded_d; index += kThreads) {
            const int feature = index % padded_d;
            const std::size_t key = key_begin + index / padded_d;
            const bool valid = key < static_cast<std::size_t>(N) && feature < d;
            if (int8_requested) {
                const int packed = (feature / 16) * kBlockN * 16 + (index / padded_d) * 16 + feature % 16;
                k8_tile[packed] = valid ? quantized.k[key * d + feature] : 0;
            } else {
                k_tile[index] = valid ? K[key * d + feature] : __float2half(0.0f);
            }
            v_tile[index] = valid ? V[key * d + feature] : __float2half(0.0f);
        }
        __syncthreads();
        if (int8_requested) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, int> scores;
            wmma::fill_fragment(scores, 0);
            for (int feature = 0; feature < padded_d; feature += kTileK) {
                wmma::load_matrix_sync(a, q8_tile + (feature / 16) * kBlockM * 16, 16);
                wmma::load_matrix_sync(b, k8_tile + (feature / 16) * kBlockN * 16 + warp_key_begin * 16, 16);
                wmma::mma_sync(scores, a, b, scores);
            }
            wmma::store_matrix_sync(reinterpret_cast<int*>(weights) + warp_key_begin, scores,
                                    kBlockN, wmma::mem_row_major);
        } else {
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
        const float q_scale = int8_requested
            ? __half2float(quantized.q_scales[query_begin / kTileMixQQuantRows]) : 1.0f;
        const float k_scale = int8_requested
            ? __half2float(quantized.k_scales[key_begin / kTileMixKQuantRows]) : 1.0f;
        #pragma unroll
        for (int slot = 0; slot < kScoreSlots; ++slot) {
            const int key = row_lane + slot * kFusedRowThreads;
            float score = -CUDART_INF_F;
            if (query_valid && key_begin + key < static_cast<std::size_t>(N)) {
                const int index = local_query * kBlockN + key;
                if (int8_requested) {
                    const int dot = reinterpret_cast<const int*>(weights)[index];
                    if (quantized.score_dots) quantized.score_dots[query * N + key_begin + key] = dot;
                    score = static_cast<float>(dot) * q_scale * k_scale * score_scale;
                } else score = weights[index] * score_scale;
            }
            row_scores[slot] = score;
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

cudaError_t quantize_tilemix_qk(const __half* Q, const __half* K,
                               std::int8_t* Q8, std::int8_t* K8,
                               __half* q_scales, __half* k_scales,
                               int M, int N, int d, cudaStream_t stream) {
    if (!Q || !K || !Q8 || !K8 || !q_scales || !k_scales || M <= 0 || N <= 0 || d <= 0) {
        return cudaErrorInvalidValue;
    }
    if (d > kFusedMaxHeadDim) return cudaErrorNotSupported;
    quantize_blocks<kTileMixQQuantRows><<<(M - 1) / kTileMixQQuantRows + 1, 256, 0, stream>>>(Q, Q8, q_scales, M, d);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    quantize_blocks<kTileMixKQuantRows><<<(N - 1) / kTileMixKQuantRows + 1, 256, 0, stream>>>(K, K8, k_scales, N, d);
    return cudaGetLastError();
}

cudaError_t launch_fused_fp16_attention(const __half* Q, const __half* K, const __half* V,
                                       float* O, int M, int N, int d, float score_scale,
                                       cudaStream_t stream, const TileMixRouting& routing,
                                       const TileMixQuantized& quantized) {
    if (!Q || !K || !V || !O || M <= 0 || N <= 0 || d <= 0 || !std::isfinite(score_scale) ||
        routing.key_tiles_per_group <= 0 || (!routing.words && routing.word_count != 0)) {
        return cudaErrorInvalidValue;
    }
    if (d > kFusedMaxHeadDim) return cudaErrorNotSupported;
    const std::size_t words_per_row = tilemix_routing_words_per_row(N, routing.key_tiles_per_group);
    if (routing.words && routing.word_count < tilemix_routing_word_count(M, N, routing.key_tiles_per_group)) {
        return cudaErrorInvalidValue;
    }
    const bool has_quantized = quantized.q && quantized.k && quantized.q_scales && quantized.k_scales;
    const bool any_quantized = quantized.q || quantized.k || quantized.q_scales || quantized.k_scales || quantized.score_dots;
    if ((any_quantized && !has_quantized) || (routing.words && !has_quantized)) return cudaErrorInvalidValue;
    int device;
    cudaDeviceProp properties{};
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) return status;
    if (properties.major < 7 || (properties.major == 7 && properties.minor < 5)) return cudaErrorNotSupported;
    const unsigned query_tiles = static_cast<unsigned>((M - 1) / kBlockM + 1);
    const std::size_t shared_bytes = tilemix_shared_bytes(d, has_quantized);
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
    const std::size_t key_tiles = (static_cast<std::size_t>(N) - 1) / kBlockN + 1;
    fused_fp16_attention<<<query_tiles, kThreads, shared_bytes, stream>>>(
        Q, K, V, O, M, N, d, padded_d, score_scale, routing, words_per_row, key_tiles, quantized);
    return cudaGetLastError();
}
