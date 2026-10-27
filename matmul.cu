#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "matmul_config.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "This kernel requires Tensor Cores (sm_70 or newer)."
#endif

namespace wmma = nvcuda::wmma;

__global__ void tiled_matmul(const __half* Q, const __half* K, float* S, int M, int N, int d, const uint64_t* routing_words) {

    assert(blockDim.x == kThreads &&
           blockDim.y == 1 && blockDim.z == 1);
    assert(M > 0 && N > 0 && d > 0);
    assert(N <= 64 * kBlockN);
    
    // Ceil(d / kTileK) * kTileK
    const int padded_d = ((d + kTileK - 1) / kTileK) * kTileK;

    extern __shared__ __align__(32) unsigned char shared_storage[];
    __half* Q_tile = reinterpret_cast<__half*>(shared_storage);
    __half* K_tile = Q_tile + kBlockM * padded_d;
    float* S_tile = reinterpret_cast<float*>(K_tile + kBlockN * padded_d);

    const int thread = threadIdx.x;
    const int warp = thread / kWarpSize;
    const int query_begin = blockIdx.x * kBlockM;
    // Each warp computes 16 key columns
    const int warp_key_begin = warp * 16;
    const uint64_t routing_word = routing_words[blockIdx.x];

    // Load Q tile
    for (int index = thread; index < kBlockM * padded_d; index += kThreads) {
        const int local_query = index / padded_d;
        const int feature = index % padded_d;
        const int query = query_begin + local_query;
        Q_tile[index] = query < M && feature < d ?
                        Q[static_cast<size_t>(query) * d + feature]
                        : __float2half(0.0f);
    }
    __syncthreads();

    // Loop over K tiles
    for (int key_begin = 0; key_begin < N; key_begin += kBlockN) {
        const int key_tile_index = key_begin / kBlockN;
        // Only FP16 currently
         const bool use_int8 =
            ((routing_word >> key_tile_index) & uint64_t{1}) != 0;
        
         for (int index = thread; index < kBlockN * padded_d; index += kThreads) {
            const int local_key = index / padded_d;
            const int feature = index % padded_d;
            const int key = key_begin + local_key;
            K_tile[index] = key < N && feature < d
                ? K[static_cast<size_t>(key) * d + feature]
                : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
        wmma::fill_fragment(accumulator, 0.0f);

        for (int feature_begin = 0; feature_begin < padded_d; feature_begin += kTileK) {
            wmma::load_matrix_sync(q_fragment, Q_tile + feature_begin, padded_d);
            wmma::load_matrix_sync(k_fragment, K_tile + warp_key_begin * padded_d + feature_begin, padded_d);
            wmma::mma_sync(accumulator, q_fragment, k_fragment, accumulator);
        }

        wmma::store_matrix_sync(S_tile + warp_key_begin, accumulator, kBlockN, wmma::mem_row_major);
        __syncthreads();

        for (int index = thread; index < kBlockM * kBlockN; index += kThreads) {
            const int query = query_begin + index / kBlockN;
            const int key = key_begin + index % kBlockN;
            if (query < M && key < N) {
                S[static_cast<size_t>(query) * N + key] = S_tile[index];
            }
        }

        __syncthreads();


         }
    }


