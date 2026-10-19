#include <cuda_runtime.h>

constexpr int kBlockM = 16;
constexpr int kBlockN = 32;
constexpr int kThreads = 256;
constexpr int kOutputsPerThread = kBlockM * kBlockN / kThreads;

__global__ void tiled_matmul(const float* Q,
                             const float* K,
                             float* S,
                             int M,
                             int N,
                             int d) {

    extern __shared__ float shared[];
    float* Q_tile = shared;
    float* KT_tile = shared + static_cast<size_t>(kBlockM) * d;

    const int thread = threadIdx.x;
    const int query_begin = blockIdx.x * kBlockM;

    for (int index = thread; index < kBlockM * d; index += kThreads) {
        const int local_query = index / d;
        const int feature = index % d;
        const int query = query_begin + local_query;
        Q_tile[index] = query < M
                            ? Q[static_cast<size_t>(query) * d + feature]
                            : 0.0f;
    }
    __syncthreads();

    for (int key_begin = 0; key_begin < N; key_begin += kBlockN) {
        float sums[kOutputsPerThread] = {};

        for (int index = thread; index < kBlockN * d; index += kThreads) {
            const int local_key = index / d;
            const int feature = index % d;
            const int key = key_begin + local_key;
            KT_tile[feature * (kBlockN + 1) + local_key] =
                key < N ? K[static_cast<size_t>(key) * d + feature] : 0.0f;
        }
        __syncthreads();

        for (int output = 0; output < kOutputsPerThread; ++output) {
            const int index = thread + output * kThreads;
            const int local_query = index / kBlockN;
            const int local_key = index % kBlockN;
            for (int feature = 0; feature < d; ++feature) {
                sums[output] += Q_tile[local_query * d + feature] *
                                KT_tile[feature * (kBlockN + 1) + local_key];
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
