#include "attention_fused.h"
#include "matmul.h"
#include "pv.h"
#include "softmax.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
constexpr int kM = 1000, kN = 1003, kDim = 128;
constexpr int kWarmup = 20, kTrials = 11, kIterations = 100;
static_assert(kM > 0 && kN > 0 && kDim > 0 && kDim <= kFusedMaxHeadDim);
static_assert(kWarmup >= 0 && kTrials >= 3 && kTrials % 2 == 1 && kIterations > 0);

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) check(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

template <class Launch>
void measure(const char* name, int kernels_per_op, cudaStream_t stream, Launch launch) {
    Event start, stop;
    for (int i = 0; i < kWarmup; ++i) launch();
    check(cudaStreamSynchronize(stream));
    std::array<double, kTrials> samples{};
    for (double& sample : samples) {
        check(cudaEventRecord(start.value, stream));
        for (int i = 0; i < kIterations; ++i) launch();
        check(cudaEventRecord(stop.value, stream));
        check(cudaEventSynchronize(stop.value));
        float elapsed_ms = 0.0f;
        check(cudaEventElapsedTime(&elapsed_ms, start.value, stop.value));
        if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0f) {
            throw std::runtime_error("invalid CUDA-event interval");
        }
        sample = static_cast<double>(elapsed_ms) / kIterations;
    }
    std::sort(samples.begin(), samples.end());
    std::printf("%-12s %10d %12.6f %12.6f %12.6f\n", name, kernels_per_op,
                samples[kTrials / 2], samples[(kTrials - 1) / 10], samples[9 * (kTrials - 1) / 10]);
}
}  // namespace

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "all";
    if (argc > 2 || (mode != "all" && mode != "qk" && mode != "softmax" &&
                    mode != "pv" && mode != "unfused" && mode != "fused")) {
        std::fprintf(stderr, "Usage: %s [all|qk|softmax|pv|unfused|fused]\n", argv[0]);
        return mode == "--help" && argc == 2 ? 0 : 1;
    }
    try {
        const auto selected = [&](std::string_view name) { return mode == "all" || mode == name; };
        const bool workspace = mode != "fused";
        if (workspace && kN > 64 * kBlockN) {
            throw std::runtime_error("legacy QK path requires Lk <= 2048; fused mode has no routing-word limit");
        }
        const size_t q_count = static_cast<size_t>(kM) * kDim;
        const size_t kv_count = static_cast<size_t>(kN) * kDim;
        const size_t scores = workspace ? static_cast<size_t>(kM) * kN : 0;
        const unsigned query_tiles = (kM - 1) / kBlockM + 1;
        const size_t words = workspace ? query_tiles : 0;
        constexpr int padded_d = (kDim + kTileK - 1) / kTileK * kTileK;
        constexpr size_t qk_shared = sizeof(__half) * padded_d * (kBlockM + kBlockN)
                                   + sizeof(float) * kBlockM * kBlockN;
        const size_t bytes = sizeof(__half) * (q_count + 2 * kv_count)
                           + sizeof(float) * (q_count + 2 * scores) + sizeof(uint64_t) * words;
        int device, driver, runtime;
        cudaDeviceProp properties{};
        size_t free_bytes, total_bytes;
        check(cudaGetDevice(&device));
        check(cudaGetDeviceProperties(&properties, device));
        check(cudaMemGetInfo(&free_bytes, &total_bytes));
        check(cudaDriverGetVersion(&driver));
        check(cudaRuntimeGetVersion(&runtime));
        if (bytes > free_bytes / 4) throw std::runtime_error("benchmark exceeds available VRAM budget");
        if (workspace) {
            if (qk_shared > std::max(properties.sharedMemPerBlock, properties.sharedMemPerBlockOptin)) {
                throw std::runtime_error("QK tile exceeds device shared-memory limit");
            }
            if (qk_shared > properties.sharedMemPerBlock) {
                check(cudaFuncSetAttribute(tiled_matmul, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(qk_shared)));
            }
        }
        std::printf("GPU=%s cc=%d.%d free_bytes=%zu total_bytes=%zu allocated_bytes=%zu\n",
                    properties.name, properties.major, properties.minor, free_bytes, total_bytes, bytes);
        std::printf("CUDA driver_api=%d runtime=%d toolkit=%d.%d.%d host_compiler=%s\n",
                    driver, runtime, __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__, __VERSION__);
        std::printf("B=1 H=1 Lq=%d Lk=%d d=%d noncausal; Q/K/V=FP16 S/P/O=FP32; "
                    "score/state/PV_accum=FP32; "
                    "Q/K/V/O strides=(%d,1) S/P strides=(%d,1)\n", kM, kN, kDim, kDim, kN);
        std::printf("QK/fused tile=%dx%d threads=%d shared_bytes(QK/fused)=%zu/%zu; "
                    "softmax_threads=%d PV_threads=%d\n", kBlockM, kBlockN, kThreads,
                    qk_shared, fused_fp16_shared_bytes(kDim), kSoftmaxThreads, kPVThreads);
        std::printf("CUDA-event timing: warmup=%d trials=%d operations_per_trial=%d; "
                    "percentiles summarize trial means\n", kWarmup, kTrials, kIterations);
        std::puts("Scope: resident buffers; allocation/transfers/preparation excluded; host submission gaps included.\n"
                  "unfused = QK + softmax + PV, including intermediate S/P traffic. Run correctness tests separately.");

        std::vector<__half> q(q_count), k(kv_count), v(kv_count);
        for (size_t i = 0; i < q.size(); ++i) q[i] = __float2half((static_cast<int>(i % 17) - 8) * 0.125f);
        for (size_t i = 0; i < k.size(); ++i) k[i] = __float2half((static_cast<int>(i % 19) - 9) * 0.125f);
        for (size_t i = 0; i < v.size(); ++i) v[i] = __float2half((static_cast<int>(i % 23) - 11) * 0.1f);
        DeviceBuffer<__half> dq(q_count), dk(kv_count), dv(kv_count);
        DeviceBuffer<float> output(q_count), s(scores), p(scores);
        DeviceBuffer<uint64_t> routing(words);
        Stream stream;
        check(cudaMemcpyAsync(dq.data, q.data(), q_count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
        check(cudaMemcpyAsync(dk.data, k.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
        check(cudaMemcpyAsync(dv.data, v.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
        if (workspace) check(cudaMemsetAsync(routing.data, 0, words * sizeof(uint64_t), stream.value));

        const float scale = 1.0f / std::sqrt(static_cast<float>(kDim));
        const unsigned pv_blocks = static_cast<unsigned>((q_count - 1) / kPVThreads + 1);
        const auto qk = [&] {
            tiled_matmul<<<query_tiles, kThreads, qk_shared, stream.value>>>(
                dq.data, dk.data, s.data, kM, kN, kDim, scale, routing.data);
            check(cudaGetLastError());
        };
        const auto softmax = [&] {
            row_softmax<<<kM, kSoftmaxThreads, 0, stream.value>>>(s.data, p.data, kM, kN);
            check(cudaGetLastError());
        };
        const auto pv = [&] {
            attention_pv<<<pv_blocks, kPVThreads, 0, stream.value>>>(p.data, dv.data, output.data, kM, kN, kDim);
            check(cudaGetLastError());
        };
        // Standalone softmax/PV use the same QK-derived inputs as the pipeline.
        if (selected("softmax") || selected("pv")) {
            qk();
            if (selected("pv")) softmax();
        }
        check(cudaStreamSynchronize(stream.value));
        std::printf("%-12s %10s %12s %12s %12s\n", "operation", "kernels/op", "median_ms", "p10_ms", "p90_ms");
        if (selected("qk")) measure("qk", 1, stream.value, qk);
        if (selected("softmax")) measure("softmax", 1, stream.value, softmax);
        if (selected("pv")) measure("pv", 1, stream.value, pv);
        if (selected("unfused")) measure("unfused", 3, stream.value, [&] { qk(); softmax(); pv(); });
        if (selected("fused")) measure("fused", 1, stream.value, [&] {
            check(launch_fused_fp16_attention(dq.data, dk.data, dv.data, output.data,
                                              kM, kN, kDim, scale, stream.value));
        });
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
