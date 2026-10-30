#include "attention_fused.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(&data, count * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
}  // namespace

int main() {
    try {
        constexpr int M = 1000, N = 1003, d = 128;
        constexpr int group_size = 2;
        const std::size_t q_count = static_cast<std::size_t>(M) * d;
        const std::size_t kv_count = static_cast<std::size_t>(N) * d;
        const std::size_t word_count = tilemix_routing_word_count(M, N, group_size);
        const std::size_t bytes = sizeof(__half) * (q_count + 2 * kv_count)
                                + sizeof(float) * q_count + sizeof(std::uint64_t) * word_count;
        std::size_t free_bytes, total_bytes;
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (bytes > free_bytes / 4) throw std::runtime_error("inputs exceed available VRAM budget");

        std::vector<__half> q(q_count), k(kv_count), v(kv_count);
        std::vector<float> output(q_count);
        for (std::size_t i = 0; i < q.size(); ++i) q[i] = __float2half((static_cast<int>(i % 17) - 8) * 0.125f);
        for (std::size_t i = 0; i < k.size(); ++i) k[i] = __float2half((static_cast<int>(i % 19) - 9) * 0.125f);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = __float2half((static_cast<int>(i % 23) - 11) * 0.1f);

        DeviceBuffer<__half> dq(q_count), dk(kv_count), dv(kv_count);
        DeviceBuffer<float> result(q_count);
        const std::vector<std::uint64_t> mask(word_count, UINT64_C(0xaaaaaaaaaaaaaaaa));
        DeviceBuffer<std::uint64_t> routing_words(word_count);
        check(cudaMemcpy(dq.data, q.data(), q_count * sizeof(__half), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dk.data, k.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice));
        check(cudaMemcpy(dv.data, v.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice));
        check(cudaMemcpy(routing_words.data, mask.data(), word_count * sizeof(std::uint64_t), cudaMemcpyHostToDevice));
        const TileMixRouting routing{routing_words.data, word_count, group_size};
        check(launch_fused_fp16_attention(dq.data, dk.data, dv.data, result.data,
                                         M, N, d, 1.0f / std::sqrt(static_cast<float>(d)), nullptr, routing));
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(output.data(), result.data, q_count * sizeof(float), cudaMemcpyDeviceToHost));
        std::printf("FP16 attention: M=%d N=%d d=%d output[0]=%.9g\n", M, N, d, output[0]);
        std::printf("Routing: %d key tiles/group, %zu packed words; INT8 requests use FP16 fallback\n",
                    group_size, word_count);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
