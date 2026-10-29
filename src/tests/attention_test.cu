#include "matmul.h"
#include "attention_fused.h"
#include "matmul_config.h"
#include "pv.h"
#include "reference_attention.h"
#include "reference_pv.h"
#include "softmax.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T> struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(size_t count) { cuda_check(cudaMalloc(&data, count * sizeof(T))); }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

std::vector<__half> half_values(const std::vector<float>& values) {
    std::vector<__half> result;
    for (float value : values) result.push_back(__float2half(value));
    return result;
}

template <class T>
void compare(const std::vector<T>& actual, const std::vector<double>& expected,
             const char* label, double atol, double rtol) {
    if (actual.empty() || actual.size() != expected.size()) {
        throw std::runtime_error("output shape mismatch");
    }
    double max_abs = 0.0, max_rel = 0.0, squared = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            error > atol + rtol * std::fabs(expected[i])) {
            std::fprintf(stderr, "%s mismatch at %zu: got %.17g expected %.17g\n",
                         label, i, static_cast<double>(actual[i]), expected[i]);
            throw std::runtime_error("output comparison failed");
        }
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(expected[i]), atol));
        squared += error * error;
    }
    std::printf("PASS %s atol=%.1g rtol=%.1g max_abs_error=%.9g max_rel_error=%.9g "
                "rms_error=%.9g\n", label, atol, rtol, max_abs, max_rel,
                std::sqrt(squared / actual.size()));
}

void cpu_checks() {
    const auto values = half_values({2, -4, 6, 8, -2, 4});
    const float probabilities[] = {1, 0, 0, 0.25f, 0.5f, 0.25f};
    std::vector<double> result(4);
    reference_pv(probabilities, values.data(), result.data(), 2, 3, 2);
    compare(result, {2, -4, 3, 4}, "CPU analytic PV", 1e-12, 1e-12);

    const float zeros[] = {0, 0, 0, 0, 0, 0};
    reference_pv(zeros, values.data(), result.data(), 2, 3, 2);
    compare(result, {0, 0, 0, 0}, "CPU zero PV", 1e-12, 1e-12);

    const auto cancellation = half_values({60000, -60000, 2, -2});
    const float uniform[] = {0.25f, 0.25f, 0.25f, 0.25f};
    std::vector<double> scalar(1);
    reference_pv(uniform, cancellation.data(), scalar.data(), 1, 4, 1);
    compare(scalar, {0}, "CPU cancelling PV", 1e-12, 1e-12);

    const auto q_zero = half_values({0, 0, 0, 0});
    const auto keys = half_values({1, 2, 3, 4, 5, 6});
    reference_attention(q_zero.data(), keys.data(), values.data(), result.data(), 2, 3, 2);
    compare(result, {2, 8.0 / 3, 2, 8.0 / 3}, "CPU uniform attention", 1e-12, 1e-12);

    const auto one_q = half_values({1, 2, 3, -1, -2, -3});
    const auto one_k = half_values({4, 5, 6});
    const auto one_v = half_values({-2, 0, 1.5f});
    std::vector<double> single_key(6);
    reference_attention(one_q.data(), one_k.data(), one_v.data(), single_key.data(), 2, 1, 3);
    compare(single_key, {-2, 0, 1.5, -2, 0, 1.5}, "CPU single-key attention", 1e-12, 1e-12);

    const auto extreme_q = half_values({1000});
    const auto extreme_k = half_values({1000, -1000});
    const auto extreme_v = half_values({3, -7});
    reference_attention(extreme_q.data(), extreme_k.data(), extreme_v.data(), scalar.data(), 1, 2, 1);
    compare(scalar, {3}, "CPU extreme-logit attention", 1e-12, 1e-12);

    bool rejected_pv = false, rejected_attention = false;
    try { reference_pv(probabilities, values.data(), result.data(), 2, 0, 2); }
    catch (const std::invalid_argument&) { rejected_pv = true; }
    try { reference_attention(nullptr, keys.data(), values.data(), result.data(), 2, 3, 2); }
    catch (const std::invalid_argument&) { rejected_attention = true; }
    if (!rejected_pv || !rejected_attention) throw std::runtime_error("invalid input accepted");
    std::puts("PASS CPU invalid-input checks");
}

struct Shape { int m, n, d; const char* pattern; };

void gpu_case(const Shape& shape, const cudaDeviceProp& properties) {
    const int m = shape.m, n = shape.n, d = shape.d;
    const size_t q_count = static_cast<size_t>(m) * d;
    const size_t kv_count = static_cast<size_t>(n) * d;
    const size_t s_count = static_cast<size_t>(m) * n;
    const int query_tiles = (m + kBlockM - 1) / kBlockM;
    const int padded_d = ((d + kTileK - 1) / kTileK) * kTileK;
    const size_t shared_bytes = sizeof(__half) * static_cast<size_t>(padded_d) * (kBlockM + kBlockN)
                              + sizeof(float) * kBlockM * kBlockN;
    size_t free_bytes, total_bytes;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t required = sizeof(__half) * (q_count + 2 * kv_count)
                          + sizeof(float) * (2 * s_count + q_count)
                          + sizeof(uint64_t) * query_tiles;
    if (required > free_bytes / 4 || required > 64 * 1024 * 1024) {
        throw std::runtime_error("attention case exceeds available VRAM/test budget");
    }
    if (shared_bytes > std::max(properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin)) {
        throw std::runtime_error("attention case exceeds shared-memory limit");
    }
    if (shared_bytes > properties.sharedMemPerBlock) {
        cuda_check(cudaFuncSetAttribute(tiled_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(shared_bytes)));
    }
    std::mt19937 generator(m * 131 + n * 17 + d);
    const auto random_value = [&]() { return (static_cast<int>(generator() % 20001) - 10000) / 10000.0f; };
    std::vector<__half> q(q_count), k(kv_count), v(kv_count);
    std::vector<float> p(s_count), actual(q_count);
    std::vector<double> expected(q_count);
    for (auto& value : q) value = __float2half(random_value());
    for (auto& value : k) value = __float2half(random_value());
    for (auto& value : v) value = __float2half(random_value());
    for (int row = 0; row < m; ++row) {
        double denominator = 0.0;
        for (int key = 0; key < n; ++key) {
            p[static_cast<size_t>(row) * n + key] = 1 + generator() % 10000;
            denominator += p[static_cast<size_t>(row) * n + key];
        }
        for (int key = 0; key < n; ++key) p[static_cast<size_t>(row) * n + key] /= denominator;
    }
    const std::string pattern = shape.pattern;
    if (pattern == "uniform" || pattern == "cancellation") {
        std::fill(q.begin(), q.end(), __float2half(0.0f));
        std::fill(p.begin(), p.end(), 1.0f / n);
    }
    if (pattern == "zero-values") std::fill(v.begin(), v.end(), __float2half(0.0f));
    if (pattern == "cancellation") {
        for (int key = 0; key < n; ++key) {
            for (int feature = 0; feature < d; ++feature) {
                v[static_cast<size_t>(key) * d + feature] = __float2half(key % 2 ? -1.0f : 1.0f);
            }
        }
    }
    if (pattern == "one-hot" || pattern == "extreme") {
        std::fill(p.begin(), p.end(), 0.0f);
        for (int row = 0; row < m; ++row) p[static_cast<size_t>(row) * n] = 1.0f;
    }
    if (pattern == "extreme") {
        std::fill(q.begin(), q.end(), __float2half(1000.0f));
        std::fill(k.begin(), k.end(), __float2half(-1000.0f));
        std::fill(k.begin(), k.begin() + d, __float2half(1000.0f));
    }
    std::vector<uint64_t> routing(query_tiles, uint64_t{0});
    DeviceArray<__half> dq(q_count), dk(kv_count), dv(kv_count);
    DeviceArray<float> ds(s_count), dp(s_count), dout(q_count);
    DeviceArray<uint64_t> routes(query_tiles);
    cuda_check(cudaMemcpy(dq.data, q.data(), q_count * sizeof(__half), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dk.data, k.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dv.data, v.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(dp.data, p.data(), s_count * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(routes.data, routing.data(), routing.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    // Cap the test grid so larger outputs also exercise the grid-stride loop.
    const unsigned blocks = std::min(2u, static_cast<unsigned>((q_count - 1) / kPVThreads + 1));
    std::printf("Case B=1 H=1 M=%d N=%d d=%d pattern=%s contiguous QK/V=FP16 P/O=FP32 "
                "QK_tile=%dx%d PV_threads=%d\n", m, n, d, shape.pattern, kBlockM, kBlockN, kPVThreads);

    reference_pv(p.data(), v.data(), expected.data(), m, n, d);
    cuda_check(cudaMemset(dout.data, 0xff, q_count * sizeof(float)));
    attention_pv<<<blocks, kPVThreads>>>(dp.data, dv.data, dout.data, m, n, d);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaMemcpy(actual.data(), dout.data, q_count * sizeof(float), cudaMemcpyDeviceToHost));
    compare(actual, expected, "CUDA isolated PV", 1e-5, 1e-4);

    reference_attention(q.data(), k.data(), v.data(), expected.data(), m, n, d);
    cuda_check(cudaMemset(ds.data, 0xff, s_count * sizeof(float)));
    cuda_check(cudaMemset(dp.data, 0xff, s_count * sizeof(float)));
    cuda_check(cudaMemset(dout.data, 0xff, q_count * sizeof(float)));
    tiled_matmul<<<query_tiles, kThreads, shared_bytes>>>(
        dq.data, dk.data, ds.data, m, n, d, 1.0f / std::sqrt(static_cast<float>(d)), routes.data);
    cuda_check(cudaGetLastError());
    row_softmax<<<m, kSoftmaxThreads>>>(ds.data, dp.data, m, n);
    cuda_check(cudaGetLastError());
    attention_pv<<<blocks, kPVThreads>>>(dp.data, dv.data, dout.data, m, n, d);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaMemcpy(actual.data(), dout.data, q_count * sizeof(float), cudaMemcpyDeviceToHost));
    compare(actual, expected, "CUDA complete attention", 1e-5, 1e-4);

    const std::vector<double> unfused(actual.begin(), actual.end());
    cuda_check(cudaMemset(dout.data, 0xff, q_count * sizeof(float)));
    cuda_check(launch_fused_fp16_attention(dq.data, dk.data, dv.data, dout.data,
        m, n, d, 1.0f / std::sqrt(static_cast<float>(d))));
    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaMemcpy(actual.data(), dout.data, q_count * sizeof(float), cudaMemcpyDeviceToHost));
    compare(actual, expected, "CUDA fused attention vs FP64", 1e-5, 1e-4);
    compare(actual, unfused, "CUDA fused vs unfused attention", 1e-5, 1e-4);
}

void gpu_checks() {
    int device, driver, runtime;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device));
    cuda_check(cudaGetDeviceProperties(&properties, device));
    cuda_check(cudaDriverGetVersion(&driver));
    cuda_check(cudaRuntimeGetVersion(&runtime));
    size_t free_bytes, total_bytes;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::printf("GPU=%s cc=%d.%d free_bytes=%zu total_bytes=%zu driver=%d runtime=%d\n",
                properties.name, properties.major, properties.minor, free_bytes, total_bytes, driver, runtime);
    const Shape cases[] = {{1, 1, 1, "random"}, {3, 5, 7, "random"},
        {15, 31, 15, "random"}, {16, 32, 16, "random"}, {17, 33, 17, "random"},
        {35, 67, 65, "random"}, {64, 64, 32, "random"}, {33, 65, 128, "random"},
        {17, 33, 17, "uniform"}, {17, 33, 17, "zero-values"},
        {2, 32, 17, "cancellation"}, {3, 5, 7, "one-hot"}, {2, 3, 1, "extreme"}};
    for (const Shape& shape : cases) gpu_case(shape, properties);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu-only";
        if (argc != 1 && !cpu_only) throw std::invalid_argument("usage: attention_test [--cpu-only]");
        cpu_checks();
        if (!cpu_only) gpu_checks();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
