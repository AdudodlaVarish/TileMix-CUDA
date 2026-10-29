#include "pv.h"

#include <cassert>
#include <cmath>
#include <cstddef>

__global__ void attention_pv(const float* P, const __half* V, float* O,
                             int M, int N, int d) {
    assert(M > 0 && N > 0 && d > 0);
    assert(blockDim.x == kPVThreads && blockDim.y == 1 && blockDim.z == 1);
    const size_t count = static_cast<size_t>(M) * d;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t query = index / d;
        const size_t feature = index % d;
        float sum = 0.0f;
        for (int key = 0; key < N; ++key) {
            sum = fmaf(P[query * N + key],
                       __half2float(V[static_cast<size_t>(key) * d + feature]), sum);
        }
        O[index] = sum;
    }
}
