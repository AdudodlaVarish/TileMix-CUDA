#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cassert>
#include <cstdint>

constexpr int kBlockM = 16;
constexpr int kBlockN = 32;
constexpr int kThreads = 256;
constexpr int kOutputsPerThread = kBlockM * kBlockN / kThreads;

__global__ void tiled_matmul(const __half* Q,
                             const __half* K,
                             float* S,
                             int M,
                             int N,
                             int d,
                             const uint64_t* routing_words) {

    assert(N >= 0 && N <= 64 * kBlockN);

    extern __shared__ __half shared[];
    __half* Q_tile = shared;
    __half* KT_tile = shared + static_cast<size_t>(kBlockM) * d;

    const int thread = threadIdx.x;
    const int query_begin = blockIdx.x * kBlockM;
    const uint64_t routing_word = routing_words[blockIdx.x];

    // Load values into Query Tile
    for (int index = thread; index < kBlockM * d; index += kThreads) {
        const int local_query = index / d;
        const int feature = index % d;
        const int query = query_begin + local_query;
        Q_tile[index] = query < M
                            ? Q[static_cast<size_t>(query) * d + feature]
                            : __float2half(0.0f);
    }
    __syncthreads();

    for (int key_begin = 0; key_begin < N; key_begin += kBlockN) {
        const int key_tile_index = key_begin / kBlockN;
        // Routing decision, 0: fp16 matmul 1: int8 matmul
        [[maybe_unused]] const bool use_int8 =
            ((routing_word >> key_tile_index) & uint64_t{1}) != 0;
        float sums[kOutputsPerThread] = {};
        
        // Load values into K^T Tile
        for (int index = thread; index < kBlockN * d; index += kThreads) {
            const int local_key = index / d;
            const int feature = index % d;
            const int key = key_begin + local_key;
            KT_tile[feature * (kBlockN + 1) + local_key] =
                key < N ? K[static_cast<size_t>(key) * d + feature] : __float2half(0.0f);
        }
        __syncthreads();

        for (int output = 0; output < kOutputsPerThread; ++output) {
            const int index = thread + output * kThreads;
            const int local_query = index / kBlockN;
            const int local_key = index % kBlockN;
            for (int feature = 0; feature < d; ++feature) {
                sums[output] = fmaf(
                    __half2float(Q_tile[local_query * d + feature]),
                    __half2float(KT_tile[feature * (kBlockN + 1) + local_key]),
                    sums[output]);
            }
        }

        __syncthreads();

        for (int output = 0; output < kOutputsPerThread; ++output) {
            const int index = thread + output * kThreads;
            const int query = query_begin + index / kBlockN;
            const int key = key_begin + index % kBlockN;
            if (query < M && key < N) {
                S[static_cast<size_t>(query) * N + key] = sums[output];
            }
        }
    }
}
