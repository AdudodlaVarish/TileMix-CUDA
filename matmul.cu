#include <cuda_runtime.h>

constexpr int kTileSize = 16;

// M = row dim, N = col dim, K = contraction dim

__global__ void tiled_matmul(const float* A,
                             const float* B,
                             float* C,
                             int M,
                             int N,
                             int K) {
                                
    __shared__ float A_tile[kTileSize][kTileSize];
    __shared__ float B_tile[kTileSize][kTileSize];

    const int row = blockIdx.y * kTileSize + threadIdx.y;
    const int col = blockIdx.x * kTileSize + threadIdx.x;
    float sum = 0.0f;

    const int tile_count = (K + kTileSize - 1) / kTileSize;

    for (int tile = 0; tile < tile_count; ++tile) {
        const int A_col = tile * kTileSize + threadIdx.x;
        const int B_row = tile * kTileSize + threadIdx.y;

        A_tile[threadIdx.y][threadIdx.x] =
            row < M && A_col < K ? A[row * K + A_col] : 0.0f;
        B_tile[threadIdx.y][threadIdx.x] =
            B_row < K && col < N ? B[B_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
