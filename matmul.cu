#include <cuda_runtime.h>

// Query rows, key rows, and features per shared-memory chunk are independent.
constexpr int kBlockM = 16;
constexpr int kBlockN = 32;
constexpr int kBlockD = 32;
constexpr int kThreads = 256;
constexpr int kOutputsPerThread = kBlockM * kBlockN / kThreads;

// Computes S = Q * K^T using FP32 arithmetic (no attention scaling or softmax).
// All matrices are row-major: Q[M,d], K[N,d], and S[M,N].
// Required launch: block(256), grid(ceil(M / 16)), with other dimensions = 1.
// M and N must be positive for a launch; d may be zero, producing zeros.
// Q, K, and S must not overlap. Partial tiles are supported in all dimensions.
__global__ void tiled_matmul(const float* Q,
                             const float* K,
                             float* S,
                             int M,
                             int N,
                             int d) {
    __shared__ float Q_tile[kBlockM][kBlockD];
    // Store K transposed on chip so neighboring output columns read neighboring
    // words. The extra column prevents bank conflicts during the transpose load.
    __shared__ float KT_tile[kBlockD][kBlockN + 1];

    const int thread = threadIdx.x;
    const int query_begin = blockIdx.x * kBlockM;

    // A block owns one query-tile row and streams across all key tiles.
    // This is where a per-query-row routing mask will eventually be read.
    for (int key_begin = 0; key_begin < N; key_begin += kBlockN) {
        // The 16x32 score tile has 512 values: each of 256 threads owns two.
        float sums[kOutputsPerThread] = {};

        // Logical operands are Q[16,d] and K[32,d]. Chunking d bounds shared
        // memory use even when the full head dimension does not fit on chip.
        // shortcut: reload Q chunks for each key tile; consider retaining Q when
        // specializing this baseline for a fixed head dimension.
        for (int feature_begin = 0; feature_begin < d; feature_begin += kBlockD) {
            // Cooperatively load Q: each thread loads a subset of the tile.
            for (int index = thread; index < kBlockM * kBlockD; index += kThreads) {
                const int local_query = index / kBlockD;
                const int local_feature = index % kBlockD;
                const int query = query_begin + local_query;
                const int feature = feature_begin + local_feature;
                Q_tile[local_query][local_feature] =
                    query < M && feature < d
                        ? Q[static_cast<size_t>(query) * d + feature]
                        : 0.0f;
            }

            // K stays row-major in global memory. Transpose only the shared
            // chunk: K[key,feature] becomes KT_tile[feature,local_key].
            for (int index = thread; index < kBlockN * kBlockD; index += kThreads) {
                const int local_key = index / kBlockD;
                const int local_feature = index % kBlockD;
                const int key = key_begin + local_key;
                const int feature = feature_begin + local_feature;
                KT_tile[local_feature][local_key] =
                    key < N && feature < d
                        ? K[static_cast<size_t>(key) * d + feature]
                        : 0.0f;
            }

            // Every thread must finish loading before any thread reads a tile.
            // Edge threads still participate; returning early would break this.
            __syncthreads();

            for (int output = 0; output < kOutputsPerThread; ++output) {
                const int index = thread + output * kThreads;
                const int local_query = index / kBlockN;
                const int local_key = index % kBlockN;
                for (int feature = 0; feature < kBlockD; ++feature) {
                    sums[output] += Q_tile[local_query][feature] *
                                    KT_tile[feature][local_key];
                }
            }

            // Finish reading before anyone overwrites shared memory next time.
            __syncthreads();
        }

        // All feature chunks contribute to the same score tile. A routing bit
        // will select one precision for this whole tile, not for each d chunk.
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
