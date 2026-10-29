#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

inline constexpr int kPVThreads = 256;

// Contiguous, separate P[M,N] FP32, V[N,d] FP16, O[M,d] FP32 buffers.
// M,N,d > 0. Inputs and FP32 weighted sums must be finite. No masks or scaling
// are applied here: P already contains the softmax probabilities.
// Launch one-dimensional CTAs of kPVThreads threads; grid-stride output loop.
__global__ void attention_pv(const float* P, const __half* V, float* O,
                             int M, int N, int d);
