#include "attention_fused.h"
#include "reference_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class T> struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(size_t count) { cuda_check(cudaMalloc(&data, count * sizeof(T))); }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    explicit Stream(bool default_stream = false) {
        if (!default_stream) cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Shape { int m, n, d; const char* pattern; bool default_stream = false; };
struct Inputs { std::vector<__half> q, k, v; };

Inputs inputs(const Shape& shape) {
    const int m = shape.m, n = shape.n, d = shape.d;
    Inputs result{std::vector<__half>(static_cast<size_t>(m) * d),
                  std::vector<__half>(static_cast<size_t>(n) * d),
                  std::vector<__half>(static_cast<size_t>(n) * d)};
    std::mt19937 generator(m * 131 + n * 17 + d);
    const auto random_value = [&]() {
        return __float2half((static_cast<int>(generator() % 20001) - 10000) / 10000.0f);
    };
    for (auto& x : result.q) x = random_value();
    for (auto& x : result.k) x = random_value();
    for (auto& x : result.v) x = random_value();
    const std::string pattern = shape.pattern;
    if (pattern == "zeros") {
        std::fill(result.q.begin(), result.q.end(), __float2half(0.0f));
        std::fill(result.k.begin(), result.k.end(), __float2half(0.0f));
        std::fill(result.v.begin(), result.v.end(), __float2half(0.0f));
        return result;
    }
    if (pattern == "random") return result;
    if (pattern == "zero-v") {
        std::fill(result.v.begin(), result.v.end(), __float2half(0.0f));
        return result;
    }
    if (pattern == "constant-v") {
        for (int key = 0; key < n; ++key) {
            for (int f = 0; f < d; ++f) {
                result.v[static_cast<size_t>(key) * d + f] = __float2half((f % 7 - 3) * 0.25f);
            }
        }
        return result;
    }
    if (pattern == "cancellation") {
        std::fill(result.q.begin(), result.q.end(), __float2half(0.0f));
        for (int key = 0; key < n; ++key) {
            for (int f = 0; f < d; ++f) {
                result.v[static_cast<size_t>(key) * d + f] = __float2half(key % 2 ? -1.0f : 1.0f);
            }
        }
        return result;
    }
    if (pattern == "large-dot") {
        std::fill(result.q.begin(), result.q.end(), __float2half(1000.0f));
        std::fill(result.k.begin(), result.k.end(), __float2half(-1000.0f));
        std::fill(result.k.begin(), result.k.begin() + d, __float2half(1000.0f));
        return result;
    }
    std::fill(result.q.begin(), result.q.end(), __float2half(0.0f));
    if (pattern == "uniform") return result;
    std::fill(result.k.begin(), result.k.end(), __float2half(0.0f));
    for (int query = 0; query < m; ++query) result.q[static_cast<size_t>(query) * d] = __float2half(1.0f);
    for (int key = 0; key < n; ++key) {
        const int tile = key / kBlockN;
        float score, value;
        if (pattern == "rising") {
            score = 4.0f * (tile - 1);
            value = tile == 0 ? 1.0f : (tile == 1 ? -1.0f : 2.0f);
        } else if (pattern == "falling") {
            score = 4.0f * (1 - tile);
            value = tile == 0 ? 1.0f : (tile == 1 ? -1.0f : 2.0f);
        } else if (pattern == "alternating") {
            const float scores[] = {0, 4, -2, 8, 10};
            const float values[] = {1, -1, 2, -2, 3};
            require(tile < 5, "alternating fixture supports five tiles");
            score = scores[tile];
            value = values[tile];
        } else if (pattern == "late-extreme") {
            score = key == n - 1 ? 1000.0f : 0.0f;
            value = key == n - 1 ? 3.0f : 1.0f;
        } else if (pattern == "negative") {
            score = -1000.0f;
            value = (key % 7 - 3) * 0.25f;
        } else throw std::invalid_argument("unknown fixture pattern");
        result.k[static_cast<size_t>(key) * d] = __float2half(score);
        for (int f = 0; f < d; ++f) result.v[static_cast<size_t>(key) * d + f] = __float2half(value);
    }
    return result;
}

template <class T>
void compare(const std::vector<T>& actual, const std::vector<double>& expected,
             const char* label, double atol, double rtol) {
    require(!actual.empty() && actual.size() == expected.size(), "output shape mismatch");
    double max_abs = 0.0, max_rel = 0.0, squared = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            error > atol + rtol * std::fabs(expected[i])) {
            std::fprintf(stderr, "%s mismatch at %zu: got %.17g expected %.17g\n",
                         label, i, static_cast<double>(actual[i]), expected[i]);
            throw std::runtime_error("TileMix FP16 comparison failed");
        }
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(expected[i]), atol));
        squared += error * error;
    }
    std::printf("PASS %s atol=%.1g rtol=%.1g max_abs_error=%.9g max_rel_error=%.9g rms_error=%.9g\n",
                label, atol, rtol, max_abs, max_rel, std::sqrt(squared / actual.size()));
}

void cpu_checks() {
    const __half x = __float2half(1.0f);
    float out;
    require(launch_fused_fp16_attention(nullptr, &x, &x, &out, 1, 1, 1, 1.0f) == cudaErrorInvalidValue,
            "null input accepted");
    require(launch_fused_fp16_attention(&x, nullptr, &x, &out, 1, 1, 1, 1.0f) == cudaErrorInvalidValue,
            "null K accepted");
    require(launch_fused_fp16_attention(&x, &x, nullptr, &out, 1, 1, 1, 1.0f) == cudaErrorInvalidValue,
            "null V accepted");
    require(launch_fused_fp16_attention(&x, &x, &x, nullptr, 1, 1, 1, 1.0f) == cudaErrorInvalidValue,
            "null O accepted");
    for (int dimension : {0, -1}) {
        require(launch_fused_fp16_attention(&x, &x, &x, &out, dimension, 1, 1, 1.0f) == cudaErrorInvalidValue,
                "invalid M accepted");
        require(launch_fused_fp16_attention(&x, &x, &x, &out, 1, dimension, 1, 1.0f) == cudaErrorInvalidValue,
                "invalid N accepted");
        require(launch_fused_fp16_attention(&x, &x, &x, &out, 1, 1, dimension, 1.0f) == cudaErrorInvalidValue,
                "invalid d accepted");
    }
    for (int d : {129, std::numeric_limits<int>::max()}) {
        require(launch_fused_fp16_attention(&x, &x, &x, &out, 1, 1, d, 1.0f) == cudaErrorNotSupported,
                "unsupported head dimension accepted");
    }
    for (float scale : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        require(launch_fused_fp16_attention(&x, &x, &x, &out, 1, 1, 1,
                scale) == cudaErrorInvalidValue, "invalid scale accepted");
    }
    require(fused_fp16_shared_bytes(1) == 4608 && fused_fp16_shared_bytes(17) == 7168 &&
            fused_fp16_shared_bytes(128) == 22528 && fused_fp16_shared_bytes(129) == 0,
            "incorrect shared-memory footprint");
    std::puts("PASS TileMix FP16 API validation and workspace layout (no CUDA initialization)");

    for (const char* pattern : {"rising", "falling", "alternating", "late-extreme", "negative"}) {
        const Shape shape{17, std::string(pattern) == "alternating" ? 129 : 65, 1, pattern};
        auto data = inputs(shape);
        std::vector<double> actual(shape.m);
        reference_attention(data.q.data(), data.k.data(), data.v.data(), actual.data(), shape.m, shape.n, shape.d);
        double expected;
        if (std::string(pattern) == "rising") {
            const double a = 32 * std::exp(-8.0), b = 32 * std::exp(-4.0);
            expected = (a - b + 2) / (a + b + 1);
        } else if (std::string(pattern) == "falling") {
            const double b = 32 * std::exp(-4.0), c = std::exp(-8.0);
            expected = (32 - b + 2 * c) / (32 + b + c);
        } else if (std::string(pattern) == "alternating") {
            const double a = 32 * std::exp(-10.0), b = 32 * std::exp(-6.0);
            const double c = 32 * std::exp(-12.0), e = 32 * std::exp(-2.0);
            expected = (a - b + 2 * c - 2 * e + 3) / (a + b + c + e + 1);
        } else if (std::string(pattern) == "late-extreme") expected = 3.0;
        else {
            double sum = 0.0;
            for (const auto& v : data.v) sum += __half2float(v);
            expected = sum / shape.n;
        }
        compare(actual, std::vector<double>(shape.m, expected), pattern, 1e-12, 1e-12);
    }
}

void gpu_case(const Shape& shape, std::ostream* fixtures) {
    const size_t count = static_cast<size_t>(shape.m) * shape.d;
    const size_t kv_count = static_cast<size_t>(shape.n) * shape.d;
    const size_t bytes = sizeof(__half) * (count + 2 * kv_count) + sizeof(float) * (count + 2);
    size_t free_bytes, total_bytes;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    require(bytes <= free_bytes / 4 && bytes <= 64 * 1024 * 1024, "case exceeds available VRAM/test budget");
    auto data = inputs(shape);
    std::vector<double> expected(count);
    reference_attention(data.q.data(), data.k.data(), data.v.data(), expected.data(), shape.m, shape.n, shape.d);
    constexpr float guard = -12345.25f;
    std::vector<float> guarded(count + 2, std::numeric_limits<float>::quiet_NaN());
    guarded.front() = guarded.back() = guard;
    DeviceArray<__half> q(count), k(kv_count), v(kv_count);
    DeviceArray<float> output(count + 2);
    Stream stream(shape.default_stream);
    // Copies and the kernel share the caller's stream.
    cuda_check(cudaMemcpyAsync(q.data, data.q.data(), count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(k.data, data.k.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(v.data, data.v.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(output.data, guarded.data(), guarded.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value));
    cuda_check(launch_fused_fp16_attention(q.data, k.data, v.data, output.data + 1,
        shape.m, shape.n, shape.d, 1.0f / std::sqrt(static_cast<float>(shape.d)), stream.value));
    cuda_check(cudaStreamSynchronize(stream.value));
    cuda_check(cudaMemcpy(guarded.data(), output.data, guarded.size() * sizeof(float), cudaMemcpyDeviceToHost));
    require(guarded.front() == guard && guarded.back() == guard, "output guard overwritten");
    std::printf("Case M=%d N=%d d=%d pattern=%s tile=%dx%d threads=%d shared_bytes=%zu "
                "stream=%s FP16_QKV FP32_state_O\n", shape.m, shape.n, shape.d, shape.pattern,
                kBlockM, kBlockN, kThreads, fused_fp16_shared_bytes(shape.d),
                shape.default_stream ? "default" : "nonblocking");
    compare(std::vector<float>(guarded.begin() + 1, guarded.end() - 1), expected,
            "TileMix FP16 vs CPU FP64", 1e-5, 1e-4);
    const auto unchanged = [](const __half* device, const std::vector<__half>& original) {
        std::vector<__half> actual(original.size());
        cuda_check(cudaMemcpy(actual.data(), device, actual.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        require(std::memcmp(actual.data(), original.data(), actual.size() * sizeof(__half)) == 0,
                "kernel modified a Q/K/V input");
    };
    unchanged(q.data, data.q);
    unchanged(k.data, data.k);
    unchanged(v.data, data.v);
    if (fixtures) {
        auto& out = *fixtures;
        out << std::setprecision(std::numeric_limits<float>::max_digits10)
            << "{\"m\":" << shape.m << ",\"n\":" << shape.n << ",\"d\":" << shape.d
            << ",\"pattern\":\"" << shape.pattern << "\"";
        const auto write_half = [&](const char* name, const std::vector<__half>& values) {
            out << ",\"" << name << "\":[";
            for (size_t i = 0; i < values.size(); ++i) {
                if (i) out << ',';
                out << __half2float(values[i]);
            }
            out << ']';
        };
        write_half("q", data.q);
        write_half("k", data.k);
        write_half("v", data.v);
        out << ",\"o\":[";
        for (size_t i = 1; i <= count; ++i) {
            if (i != 1) out << ',';
            out << guarded[i];
        }
        out << "]}\n";
        if (!out) throw std::runtime_error("failed to write JSON fixture");
    }
}

void score_scale_checks() {
    const __half q[] = {__float2half(1.0f)};
    const __half k[] = {__float2half(-1.0f), __float2half(0.0f), __float2half(1.0f)};
    const __half v[] = {__float2half(2.0f), __float2half(4.0f), __float2half(8.0f)};
    constexpr float guard = -12345.25f;
    float actual[] = {guard, std::numeric_limits<float>::quiet_NaN(), guard};
    DeviceArray<__half> dq(1), dk(3), dv(3);
    DeviceArray<float> output(3);
    Stream stream;
    cuda_check(cudaMemcpyAsync(dq.data, q, sizeof(q), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(dk.data, k, sizeof(k), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(dv.data, v, sizeof(v), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(output.data, actual, sizeof(actual), cudaMemcpyHostToDevice, stream.value));
    for (float scale : {1.0f, 0.0f, -1.0f, 0.5f}) {
        cuda_check(launch_fused_fp16_attention(dq.data, dk.data, dv.data, output.data + 1,
                                              1, 3, 1, scale, stream.value));
        cuda_check(cudaMemcpyAsync(actual, output.data, sizeof(actual), cudaMemcpyDeviceToHost, stream.value));
        cuda_check(cudaStreamSynchronize(stream.value));
        require(actual[0] == guard && actual[2] == guard, "output guard overwritten on reuse");
        const double a = std::exp(-static_cast<double>(scale)), b = std::exp(static_cast<double>(scale));
        const double expected = (2.0 * a + 4.0 + 8.0 * b) / (a + 1.0 + b);
        std::printf("Case score_scale=%.2g reused_device_buffers\n", scale);
        compare(std::vector<float>{actual[1]}, std::vector<double>{expected},
                "TileMix FP16 analytic scale / reuse", 1e-5, 1e-4);
    }
}

void gpu_checks(std::ostream* fixtures) {
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
    int reference_cases = 0;
    for (int d : {1, 15, 16, 17, 31, 32, 33, 49, 63, 64, 65, 81, 97, 113, 127, 128}) {
        gpu_case({3, 33, d, "random"}, fixtures);
        ++reference_cases;
    }
    const Shape cases[] = {{1, 1, 1, "random"}, {3, 5, 7, "random"},
        {15, 31, 15, "random"}, {16, 32, 16, "random"}, {17, 33, 17, "random"},
        {31, 63, 31, "random"}, {32, 64, 32, "random"}, {33, 65, 33, "random"},
        {35, 67, 65, "random"}, {17, 65, 128, "random"},
        {17, 1, 128, "random"}, {17, 33, 17, "random", true},
        {17, 65, 1, "rising"}, {17, 65, 1, "falling"}, {17, 129, 1, "alternating"},
        {17, 65, 1, "late-extreme"}, {17, 33, 1, "negative"},
        {17, 33, 17, "zeros"}, {17, 33, 17, "uniform"}, {17, 33, 17, "zero-v"},
        {17, 65, 17, "constant-v"}, {2, 32, 17, "cancellation"}, {17, 33, 1, "large-dot"},
        {3, 2051, 17, "random"}, {1, 4097, 128, "uniform"}};
    for (const auto& shape : cases) {
        gpu_case(shape, fixtures);
        ++reference_cases;
    }
    score_scale_checks();
    std::printf("PASS TileMix FP16 suite: %d FP64-reference cases + 4 analytic scale/reuse cases\n", reference_cases);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu-only";
        const bool export_json = argc == 3 && std::string(argv[1]) == "--export-json";
        if (argc != 1 && !cpu_only && !export_json) {
            throw std::invalid_argument("usage: tilemix_test [--cpu-only | --export-json FILE]");
        }
        std::ofstream fixtures;
        if (export_json) {
            fixtures.open(argv[2]);
            if (!fixtures) throw std::runtime_error("could not open JSON fixture file");
        }
        std::puts("TileMix test scope: current FP16-input / FP32-state path; INT8 and mixed routing are unimplemented.");
        cpu_checks();
        if (!cpu_only) gpu_checks(export_json ? &fixtures : nullptr);
        if (export_json) {
            fixtures.close();
            if (!fixtures) throw std::runtime_error("could not close JSON fixture file");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
