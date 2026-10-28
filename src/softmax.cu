#include "softmax.h"

#include <math_constants.h>

#include <cassert>
#include <cmath>
#include <cstddef>

__global__ void row_softmax(const float* S, float* P, int M, int N) {
    assert(M > 0 && N > 0);
    assert(blockDim.x == kSoftmaxThreads && blockDim.y == 1 && blockDim.z == 1);
    const int row = blockIdx.x;
    if (row >= M) return;  

    const int thread = threadIdx.x;
    const size_t row_begin = static_cast<size_t>(row) * N;
    __shared__ float reduction[kSoftmaxThreads];

    float maximum = -CUDART_INF_F;
    for (size_t col = thread; col < static_cast<size_t>(N); col += kSoftmaxThreads) {
        maximum = fmaxf(maximum, S[row_begin + col]);
    }
    reduction[thread] = maximum;
    __syncthreads();
    for (int stride = kSoftmaxThreads / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] = fmaxf(reduction[thread], reduction[thread + stride]);
        }
        __syncthreads();
    }
    // shared-memory broadcast, no bank conflict / contention
    const float row_maximum = reduction[0];
    // Reuse the same buffer so syncthreads to prevent data races
    __syncthreads();

    float sum = 0.0f;
    for (size_t col = thread; col < static_cast<size_t>(N); col += kSoftmaxThreads) {
        const float weight = expf(S[row_begin + col] - row_maximum);
        P[row_begin + col] = weight;
        sum += weight;
    }
    reduction[thread] = sum;
    __syncthreads();
    for (int stride = kSoftmaxThreads / 2; stride > 0; stride /= 2) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    const float denominator = reduction[0];
    for (size_t col = thread; col < static_cast<size_t>(N); col += kSoftmaxThreads) {
        P[row_begin + col] /= denominator;
    }
}
