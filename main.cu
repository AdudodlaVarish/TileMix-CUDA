#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void tiled_matmul(const float* A,
                             const float* B,
                             float* C,
                             int M,
                             int N,
                             int K);

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        const cudaError_t error = (call);                               \
        if (error != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA error: %s\n",                    \
                         cudaGetErrorString(error));                     \
            return 1;                                                   \
        }                                                               \
    } while (0)

void reference_matmul(const float* A,
                      const float* B,
                      float* C,
                      int M,
                      int N,
                      int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[row * K + k]) *
                       static_cast<double>(B[k * N + col]);
            }
            C[row * N + col] = static_cast<float>(sum);
        }
    }
}

int main() {
    // Deliberately not divisible by 16, exercising partial edge tiles.
    constexpr int M = 1000;
    constexpr int N = 1003;
    constexpr int K = 997;
    constexpr int iterations = 100;

    const size_t A_bytes = static_cast<size_t>(M) * K * sizeof(float);
    const size_t B_bytes = static_cast<size_t>(K) * N * sizeof(float);
    const size_t C_bytes = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> B(static_cast<size_t>(K) * N);
    std::vector<float> C(static_cast<size_t>(M) * N);
    std::vector<float> reference_C(static_cast<size_t>(M) * N);

    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    for (size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125f;
    }

    reference_matmul(A.data(), B.data(), reference_C.data(), M, N, K);

    float* device_A = nullptr;
    float* device_B = nullptr;
    float* device_C = nullptr;

    CUDA_CHECK(cudaMalloc(&device_A, A_bytes));
    CUDA_CHECK(cudaMalloc(&device_B, B_bytes));
    CUDA_CHECK(cudaMalloc(&device_C, C_bytes));

    CUDA_CHECK(cudaMemcpy(
        device_A, A.data(), A_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_B, B.data(), B_bytes, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);
    const dim3 grid((N + block.x - 1) / block.x,
                    (M + block.y - 1) / block.y);

    // Warm up CUDA and verify one execution.
    tiled_matmul<<<grid, block>>>(device_A, device_B, device_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        C.data(), device_C, C_bytes, cudaMemcpyDeviceToHost));

    constexpr float absolute_tolerance = 1.0e-4f;
    constexpr float relative_tolerance = 1.0e-4f;
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;

    for (size_t i = 0; i < C.size(); ++i) {
        const float absolute_error = std::fabs(C[i] - reference_C[i]);
        const float relative_error =
            absolute_error / std::fmax(std::fabs(reference_C[i]), absolute_tolerance);
        const float allowed_error =
            absolute_tolerance + relative_tolerance * std::fabs(reference_C[i]);

        max_absolute_error = std::fmax(max_absolute_error, absolute_error);
        max_relative_error = std::fmax(max_relative_error, relative_error);

        if (absolute_error > allowed_error) {
            std::fprintf(stderr,
                         "Incorrect result at %zu: got %.8f, expected %.8f, "
                         "error %.8f, allowed %.8f\n",
                         i, C[i], reference_C[i], absolute_error, allowed_error);
            return 1;
        }
    }

    std::printf("Correctness: PASS (max abs error %.8g, max rel error %.8g)\n",
                max_absolute_error, max_relative_error);

    cudaEvent_t start, stop;
    
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        tiled_matmul<<<grid, block>>>(
            device_A, device_B, device_C, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const double average_ms = elapsed_ms / iterations;
    const double operations = 2.0 * M * N * K;
    const double gflops = operations / (average_ms * 1.0e6);

    std::printf("Average kernel time: %.3f ms\n", average_ms);
    std::printf("Throughput: %.1f GFLOP/s\n", gflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_A));
    CUDA_CHECK(cudaFree(device_B));
    CUDA_CHECK(cudaFree(device_C));

    return 0;
}
