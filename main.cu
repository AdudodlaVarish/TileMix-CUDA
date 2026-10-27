#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "matmul_config.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

__global__ void tiled_matmul(const __half* Q,
                             const __half* K,
                             float* S,
                             int M,
                             int N,
                             int d,
                             const uint64_t* routing_words);

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        const cudaError_t error = (call);                               \
        if (error != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA error: %s\n",                    \
                         cudaGetErrorString(error));                     \
            return 1;                                                   \
        }                                                               \
    } while (0)
 
// Baseline
void reference_matmul(const __half* Q,
                      const __half* K,
                      float* S,
                      int M,
                      int N,
                      int d) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int feature = 0; feature < d; ++feature) {
                sum += static_cast<double>(__half2float(Q[static_cast<size_t>(row) * d + feature])) *
                       static_cast<double>(__half2float(K[static_cast<size_t>(col) * d + feature]));
            }
            S[static_cast<size_t>(row) * N + col] = static_cast<float>(sum);
        }
    }
}

int main() {
    constexpr int M = 1000;
    constexpr int N = 1003;
    constexpr int d = 128;
    constexpr int iterations = 100;
    static_assert(M > 0 && N > 0 && d > 0 && iterations > 0);
    static_assert(N <= 64 * kBlockN);

    const size_t Q_bytes = static_cast<size_t>(M) * d * sizeof(__half);
    const size_t K_bytes = static_cast<size_t>(N) * d * sizeof(__half);
    const size_t S_bytes = static_cast<size_t>(M) * N * sizeof(float);
    constexpr int padded_d = ((d + kTileK - 1) / kTileK) * kTileK;
    // Shared FP16 Q/K tiles followed by the FP32 score tile.
    const size_t shared_bytes = sizeof(__half) * static_cast<size_t>(padded_d) *
                                (kBlockM + kBlockN) +
                                sizeof(float) * kBlockM * kBlockN;
    const int query_tile_count = (M + kBlockM - 1) / kBlockM;
    const size_t routing_bytes = static_cast<size_t>(query_tile_count) * sizeof(uint64_t);

    int device = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (shared_bytes > properties.sharedMemPerBlockOptin) {
        std::fprintf(stderr, "Requested %zu shared bytes per block; device limit is %zu\n",
                     shared_bytes, properties.sharedMemPerBlockOptin);
        return 1;
    }
    if (shared_bytes > properties.sharedMemPerBlock) {
        CUDA_CHECK(cudaFuncSetAttribute(tiled_matmul,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
    }

    std::vector<__half> Q(static_cast<size_t>(M) * d);
    std::vector<__half> K(static_cast<size_t>(N) * d);
    std::vector<float> S(static_cast<size_t>(M) * N);
    std::vector<float> reference_S(static_cast<size_t>(M) * N);
    std::vector<uint64_t> routing_words(query_tile_count, uint64_t{0});
    
    // Initialize with random
    for (size_t i = 0; i < Q.size(); ++i) {
        Q[i] = __float2half(static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f);
    }
    for (size_t i = 0; i < K.size(); ++i) {
        K[i] = __float2half(static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125f);
    }

    reference_matmul(Q.data(), K.data(), reference_S.data(), M, N, d);

    __half* device_Q = nullptr;
    __half* device_K = nullptr;
    float* device_S = nullptr;
    uint64_t* device_routing_words = nullptr;

    CUDA_CHECK(cudaMalloc(&device_Q, Q_bytes));
    CUDA_CHECK(cudaMalloc(&device_K, K_bytes));
    CUDA_CHECK(cudaMalloc(&device_S, S_bytes));
    CUDA_CHECK(cudaMalloc(&device_routing_words, routing_bytes));

    CUDA_CHECK(cudaMemcpy(
        device_Q, Q.data(), Q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_K, K.data(), K_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_routing_words, routing_words.data(),
                          routing_bytes, cudaMemcpyHostToDevice));

    const dim3 block(kThreads);
    const dim3 grid(query_tile_count);

    tiled_matmul<<<grid, block, shared_bytes>>>(
        device_Q, device_K, device_S, M, N, d, device_routing_words);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        S.data(), device_S, S_bytes, cudaMemcpyDeviceToHost));

    constexpr float absolute_tolerance = 1.0e-4f;
    constexpr float relative_tolerance = 1.0e-4f;
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;

    for (size_t i = 0; i < S.size(); ++i) {
        const float absolute_error = std::fabs(S[i] - reference_S[i]);
        const float relative_error =
            absolute_error / std::fmax(std::fabs(reference_S[i]), absolute_tolerance);
        const float allowed_error =
            absolute_tolerance + relative_tolerance * std::fabs(reference_S[i]);

        max_absolute_error = std::fmax(max_absolute_error, absolute_error);
        max_relative_error = std::fmax(max_relative_error, relative_error);

        if (!std::isfinite(S[i]) || !std::isfinite(reference_S[i]) ||
            absolute_error > allowed_error) {
            std::fprintf(stderr,
                         "Incorrect result at %zu: got %.8f, expected %.8f, "
                         "error %.8f, allowed %.8f\n",
                         i, S[i], reference_S[i], absolute_error, allowed_error);
            return 1;
        }
    }

    std::printf("Correctness: PASS (max abs error %.8g, max rel error %.8g)\n",
                max_absolute_error, max_relative_error);
    std::printf("M=%d N=%d d=%d, shared memory per block: %zu bytes\n",
                M, N, d, shared_bytes);

    cudaEvent_t start, stop;
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        tiled_matmul<<<grid, block, shared_bytes>>>(
            device_Q, device_K, device_S, M, N, d, device_routing_words);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const double average_ms = elapsed_ms / iterations;
    const double operations = 2.0 * M * N * d;
    const double gflops = operations / (average_ms * 1.0e6);

    std::printf("Average kernel time: %.3f ms\n", average_ms);
    std::printf("Throughput: %.1f GFLOP/s\n", gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_Q));
    CUDA_CHECK(cudaFree(device_K));
    CUDA_CHECK(cudaFree(device_S));
    CUDA_CHECK(cudaFree(device_routing_words));

    return 0;
}
