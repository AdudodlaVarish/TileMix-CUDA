#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "matmul_config.h"
#include "matmul.h"
#include "attention_fused.h"
#include "pv.h"
#include "reference_attention.h"
#include "reference_softmax.h"
#include "softmax.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        const cudaError_t error = (call);                               \
        if (error != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA error: %s\n",                    \
                         cudaGetErrorString(error));                     \
            return 1;                                                   \
        }                                                               \
    } while (0)
 
// Independent FP64 oracle: S = (Q @ K.T) / sqrt(d), using rounded FP16 inputs.
void reference_matmul(const __half* Q,
                      const __half* K,
                      float* S,
                      int M,
                      int N,
                      int d) {
    const double score_scale = 1.0 / std::sqrt(static_cast<double>(d));
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int feature = 0; feature < d; ++feature) {
                sum += static_cast<double>(__half2float(Q[static_cast<size_t>(row) * d + feature])) *
                       static_cast<double>(__half2float(K[static_cast<size_t>(col) * d + feature]));
            }
            S[static_cast<size_t>(row) * N + col] = static_cast<float>(sum * score_scale);
        }
    }
}

int main(int argc, char** argv) {
    const bool benchmark_qk = argc == 2 && std::strcmp(argv[1], "--benchmark-qk") == 0;
    if (argc != 1 && !benchmark_qk) {
        std::fprintf(stderr, "Usage: %s [--benchmark-qk]\n", argv[0]);
        return 1;
    }
    constexpr int M = 1000;
    constexpr int N = 1003;
    constexpr int d = 128;
    constexpr int iterations = 100;
    static_assert(M > 0 && N > 0 && d > 0 && iterations > 0);
    static_assert(N <= 64 * kBlockN);

    const float score_scale = 1.0f / std::sqrt(static_cast<float>(d));
    const size_t Q_bytes = static_cast<size_t>(M) * d * sizeof(__half);
    const size_t K_bytes = static_cast<size_t>(N) * d * sizeof(__half);
    const size_t O_bytes = static_cast<size_t>(M) * d * sizeof(float);
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
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t required_bytes = Q_bytes + 2 * K_bytes + 2 * S_bytes + O_bytes + routing_bytes;
    if (required_bytes > free_bytes / 4) {
        std::fprintf(stderr, "Attention test needs %zu bytes; available VRAM is %zu bytes\n",
                     required_bytes, free_bytes);
        return 1;
    }
    std::printf("GPU=%s cc=%d.%d free_bytes=%zu total_bytes=%zu\n", properties.name,
                properties.major, properties.minor, free_bytes, total_bytes);
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
    std::vector<__half> V(static_cast<size_t>(N) * d);
    std::vector<float> O(static_cast<size_t>(M) * d);
    std::vector<double> reference_O(static_cast<size_t>(M) * d);
    std::vector<float> S(static_cast<size_t>(M) * N);
    std::vector<float> reference_S(static_cast<size_t>(M) * N);
    std::vector<float> P(static_cast<size_t>(M) * N);
    std::vector<float> reference_P(static_cast<size_t>(M) * N);
    std::vector<uint64_t> routing_words(query_tile_count, uint64_t{0});
    
    // Initialize with random
    for (size_t i = 0; i < Q.size(); ++i) {
        Q[i] = __float2half(static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f);
    }
    for (size_t i = 0; i < K.size(); ++i) {
        K[i] = __float2half(static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125f);
    }
    for (size_t i = 0; i < V.size(); ++i) {
        V[i] = __float2half(static_cast<float>(static_cast<int>(i % 23) - 11) * 0.1f);
    }

    reference_matmul(Q.data(), K.data(), reference_S.data(), M, N, d);

    __half* device_Q = nullptr;
    __half* device_K = nullptr;
    __half* device_V = nullptr;
    float* device_O = nullptr;
    float* device_S = nullptr;
    float* device_P = nullptr;
    uint64_t* device_routing_words = nullptr;

    CUDA_CHECK(cudaMalloc(&device_Q, Q_bytes));
    CUDA_CHECK(cudaMalloc(&device_K, K_bytes));
    CUDA_CHECK(cudaMalloc(&device_V, K_bytes));
    CUDA_CHECK(cudaMalloc(&device_O, O_bytes));
    CUDA_CHECK(cudaMalloc(&device_S, S_bytes));
    CUDA_CHECK(cudaMalloc(&device_P, S_bytes));
    CUDA_CHECK(cudaMalloc(&device_routing_words, routing_bytes));

    CUDA_CHECK(cudaMemcpy(
        device_Q, Q.data(), Q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_K, K.data(), K_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_V, V.data(), K_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_routing_words, routing_words.data(),
                          routing_bytes, cudaMemcpyHostToDevice));

    const dim3 block(kThreads);
    const dim3 grid(query_tile_count);

    tiled_matmul<<<grid, block, shared_bytes>>>(
        device_Q, device_K, device_S, M, N, d, score_scale, device_routing_words);
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
    std::printf("M=%d N=%d d=%d, score_scale=%.8g, shared memory per block: %zu bytes\n",
                M, N, d, score_scale, shared_bytes);

    reference_softmax(reference_S.data(), reference_P.data(), M, N);
    // NaN initialization makes missing probability stores fail validation.
    CUDA_CHECK(cudaMemset(device_P, 0xff, S_bytes));
    row_softmax<<<M, kSoftmaxThreads>>>(device_S, device_P, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(P.data(), device_P, S_bytes, cudaMemcpyDeviceToHost));

    constexpr float probability_atol = 1.0e-6f;
    constexpr float probability_rtol = 1.0e-4f;
    float probability_max_abs = 0.0f, probability_max_rel = 0.0f;
    double probability_squared_error = 0.0;
    for (size_t i = 0; i < P.size(); ++i) {
        const float error = std::fabs(P[i] - reference_P[i]);
        probability_max_abs = std::fmax(probability_max_abs, error);
        probability_max_rel = std::fmax(probability_max_rel,
            error / std::fmax(std::fabs(reference_P[i]), probability_atol));
        probability_squared_error += static_cast<double>(error) * error;
        if (!std::isfinite(P[i]) || P[i] < 0.0f || P[i] > 1.0f ||
            error > probability_atol + probability_rtol * std::fabs(reference_P[i])) {
            std::fprintf(stderr, "Softmax mismatch at %zu: got %.9g expected %.9g\n",
                         i, P[i], reference_P[i]);
            return 1;
        }
    }
    for (int row = 0; row < M; ++row) {
        double sum = 0.0;
        for (int col = 0; col < N; ++col) sum += P[static_cast<size_t>(row) * N + col];
        if (std::fabs(sum - 1.0) > 1.0e-5) {
            std::fprintf(stderr, "Softmax row %d sums to %.17g\n", row, sum);
            return 1;
        }
    }
    std::printf("Softmax: PASS (atol=1e-6 rtol=1e-4, max_abs_error=%.9g, "
                "max_rel_error=%.9g, rms_error=%.9g)\n", probability_max_abs,
                probability_max_rel, std::sqrt(probability_squared_error / P.size()));

    // Full FP64 attention oracle, independent of the CUDA S/P intermediates.
    reference_attention(Q.data(), K.data(), V.data(), reference_O.data(), M, N, d);
    CUDA_CHECK(cudaMemset(device_O, 0xff, O_bytes));
    const unsigned pv_blocks = static_cast<unsigned>((O.size() - 1) / kPVThreads + 1);
    attention_pv<<<pv_blocks, kPVThreads>>>(device_P, device_V, device_O, M, N, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O.data(), device_O, O_bytes, cudaMemcpyDeviceToHost));

    constexpr double output_atol = 1.0e-5, output_rtol = 1.0e-4;
    double output_max_abs = 0.0, output_max_rel = 0.0, output_squared_error = 0.0;
    for (size_t i = 0; i < O.size(); ++i) {
        const double error = std::fabs(O[i] - reference_O[i]);
        output_max_abs = std::fmax(output_max_abs, error);
        output_max_rel = std::fmax(output_max_rel,
            error / std::fmax(std::fabs(reference_O[i]), output_atol));
        output_squared_error += error * error;
        if (!std::isfinite(O[i]) || !std::isfinite(reference_O[i]) ||
            error > output_atol + output_rtol * std::fabs(reference_O[i])) {
            std::fprintf(stderr, "Attention mismatch at %zu: got %.9g expected %.17g\n",
                         i, O[i], reference_O[i]);
            return 1;
        }
    }
    std::printf("Unfused attention: PASS (atol=1e-5 rtol=1e-4, max_abs_error=%.9g, "
                "max_rel_error=%.9g, rms_error=%.9g)\n", output_max_abs, output_max_rel,
                std::sqrt(output_squared_error / O.size()));

    const std::vector<float> unfused_O = O;
    CUDA_CHECK(cudaMemset(device_O, 0xff, O_bytes));
    CUDA_CHECK(launch_fused_fp16_attention(device_Q, device_K, device_V, device_O,
                                          M, N, d, score_scale));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(O.data(), device_O, O_bytes, cudaMemcpyDeviceToHost));
    double fused_max_abs = 0.0, fused_max_rel = 0.0, fused_squared_error = 0.0;
    double max_unfused_difference = 0.0;
    for (size_t i = 0; i < O.size(); ++i) {
        const double error = std::fabs(O[i] - reference_O[i]);
        const double difference = std::fabs(static_cast<double>(O[i]) - unfused_O[i]);
        if (!std::isfinite(O[i]) || error > output_atol + output_rtol * std::fabs(reference_O[i]) ||
            difference > output_atol + output_rtol * std::fabs(unfused_O[i])) {
            std::fprintf(stderr, "Fused attention mismatch at %zu: got %.9g, "
                         "FP64 %.17g, unfused %.9g\n", i, O[i], reference_O[i], unfused_O[i]);
            return 1;
        }
        fused_max_abs = std::fmax(fused_max_abs, error);
        fused_max_rel = std::fmax(fused_max_rel, error / std::fmax(std::fabs(reference_O[i]), output_atol));
        fused_squared_error += error * error;
        max_unfused_difference = std::fmax(max_unfused_difference, difference);
    }
    std::printf("Fused FP16 attention: PASS (atol=1e-5 rtol=1e-4, max_abs_error=%.9g, "
                "max_rel_error=%.9g, rms_error=%.9g, max_unfused_difference=%.9g)\n",
                fused_max_abs, fused_max_rel, std::sqrt(fused_squared_error / O.size()),
                max_unfused_difference);

    // Correctness-only by default; timing requires an explicit request.
    if (benchmark_qk) {
        cudaEvent_t start, stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < iterations; ++i) {
            tiled_matmul<<<grid, block, shared_bytes>>>(
                device_Q, device_K, device_S, M, N, d, score_scale, device_routing_words);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        const double average_ms = elapsed_ms / iterations;
        const double operations = 2.0 * M * N * d;
        const double gflops = operations / (average_ms * 1.0e6);

        std::printf("Average scaled-QK kernel time: %.3f ms\n", average_ms);
        std::printf("Scaled-QK throughput: %.1f GFLOP/s\n", gflops);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    CUDA_CHECK(cudaFree(device_Q));
    CUDA_CHECK(cudaFree(device_K));
    CUDA_CHECK(cudaFree(device_V));
    CUDA_CHECK(cudaFree(device_O));
    CUDA_CHECK(cudaFree(device_S));
    CUDA_CHECK(cudaFree(device_P));
    CUDA_CHECK(cudaFree(device_routing_words));

    return 0;
}
