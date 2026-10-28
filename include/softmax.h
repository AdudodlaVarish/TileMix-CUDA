#pragma once

#include <cuda_runtime.h>

inline constexpr int kSoftmaxThreads = 256;
static_assert(kSoftmaxThreads > 0 &&
              (kSoftmaxThreads & (kSoftmaxThreads - 1)) == 0 &&
              kSoftmaxThreads <= 1024,
              "Softmax reduction requires a power-of-two CTA size <= 1024.");

__global__ void row_softmax(const float* S, float* P, int M, int N);
