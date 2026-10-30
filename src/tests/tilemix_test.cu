#include "attention_fused.h"
#include "reference_attention.h"
#include "reference_tilemix.h"

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
    if (pattern == "block-outliers") {
        require(m >= 129 && n >= 129, "block-outlier fixture requires two quantization blocks");
        result.q[static_cast<size_t>(127) * d] = __float2half(8.0f);
        result.q[static_cast<size_t>(128) * d] = __float2half(0.125f);
        result.k[static_cast<size_t>(63) * d] = __float2half(-16.0f);
        result.k[static_cast<size_t>(128) * d] = __float2half(0.25f);
        return result;
    }
    if (pattern == "max-int8") {
        std::fill(result.q.begin(), result.q.end(), __float2half(1.0f));
        std::fill(result.k.begin(), result.k.end(), __float2half(1.0f));
        std::fill(result.k.begin(), result.k.begin() + d, __float2half(-1.0f));
        return result;
    }
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
    require(kTileMixQQuantRows == 128 && kTileMixKQuantRows == 64 &&
            tilemix_shared_bytes(128, true) == 28672 && tilemix_shared_bytes(17, true) == 8704,
            "incorrect quantization blocks or shared layout");
    std::int8_t byte = 0;
    __half scale_output;
    require(quantize_tilemix_qk(nullptr, &x, &byte, &byte, &scale_output, &scale_output,
                               1, 1, 1) == cudaErrorInvalidValue, "null quantizer input accepted");
    require(quantize_tilemix_qk(&x, &x, nullptr, &byte, &scale_output, &scale_output,
                               1, 1, 1) == cudaErrorInvalidValue, "null quantizer output accepted");
    require(quantize_tilemix_qk(&x, &x, &byte, &byte, nullptr, &scale_output,
                               1, 1, 1) == cudaErrorInvalidValue, "null scale output accepted");
    require(quantize_tilemix_qk(&x, &x, &byte, &byte, &scale_output, &scale_output,
                               0, 1, 1) == cudaErrorInvalidValue, "empty quantizer input accepted");
    require(quantize_tilemix_qk(&x, &x, &byte, &byte, &scale_output, &scale_output,
                               1, 1, 129) == cudaErrorNotSupported, "unsupported quantizer d accepted");
    const TileMixQuantized partial{&byte};
    require(launch_fused_fp16_attention(&x, &x, &x, &out, 1, 1, 1, 1.0f,
                                       nullptr, {}, partial) == cudaErrorInvalidValue,
            "partial quantized inputs accepted");
    require(tilemix_routing_word_count(17, 2048, 1) == 2 &&
            tilemix_routing_word_count(17, 2049, 1) == 4 &&
            tilemix_routing_word_count(17, 4097, 2) == 4 &&
            tilemix_routing_word_count(17, 4097, 1) == 6 &&
            tilemix_routing_word_count(17, std::numeric_limits<int>::max(),
                                      std::numeric_limits<int>::max()) == 2 &&
            tilemix_routing_word_count(0, 1, 1) == 0 &&
            tilemix_routing_word_count(1, 0, 1) == 0 &&
            tilemix_routing_word_count(1, 1, 0) == 0,
            "incorrect routing-mask footprint");
    for (int group : {0, -1}) {
        const TileMixRouting routing{nullptr, 0, group};
        require(launch_fused_fp16_attention(&x, &x, &x, &out, 1, 1, 1, 1.0f,
                                            nullptr, routing) == cudaErrorInvalidValue,
                "nonpositive group size accepted");
    }
    const std::uint64_t word = 0;
    require(launch_fused_fp16_attention(&x, &x, &x, &out, 1, 1, 1, 1.0f,
                                       nullptr, TileMixRouting{&word, 1, 1}) == cudaErrorInvalidValue,
            "routed attention accepted missing quantized operands");
    for (const TileMixRouting routing : {TileMixRouting{nullptr, 1, 1},
                                        TileMixRouting{&word, 0, 1},
                                        TileMixRouting{&word, 3, 1}}) {
        require(launch_fused_fp16_attention(&x, &x, &x, &out, 17, 2049, 1, 1.0f,
                                            nullptr, routing) == cudaErrorInvalidValue,
                "null or undersized routing mask accepted");
    }
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

template <class T>
void same_device_bytes(const T* device, const std::vector<T>& expected, const char* label) {
    std::vector<T> actual(expected.size());
    cuda_check(cudaMemcpy(actual.data(), device, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
    require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0, label);
}

void quantization_checks() {
    struct Case { int m, n, d; const char* pattern; };
    const Case cases[] = {{129, 65, 17, "zeros"}, {129, 65, 1, "rounding"},
        {129, 65, 3, "tiny"}, {3, 5, 128, "max-range"},
        {137, 73, 31, "random"}, {129, 131, 17, "block-outliers"}};
    for (const auto& test : cases) {
        const std::string pattern = test.pattern;
        auto data = inputs({test.m, test.n, test.d, pattern == "block-outliers" ? "block-outliers" : "random"});
        for (auto* values : {&data.q, &data.k}) {
            for (size_t i = 0; i < values->size(); ++i) {
                if (pattern == "zeros") (*values)[i] = __float2half(0.0f);
                if (pattern == "rounding") {
                    const float values_at_ties[] = {127, -127, 0.5f, -0.5f, 1.5f, -1.5f, 126.5f, -126.5f};
                    (*values)[i] = __float2half(values_at_ties[i % 8]);
                }
                if (pattern == "tiny") (*values)[i] = __float2half(std::ldexp(i % 2 ? -1.0f : 1.0f, -24));
                if (pattern == "max-range") (*values)[i] = __float2half(i % 2 ? -65504.0f : 65504.0f);
            }
        }
        const auto q_ref = reference_quantize(data.q, test.m, test.d, 128);
        const auto k_ref = reference_quantize(data.k, test.n, test.d, 64);
        if (pattern == "rounding") {
            const std::int8_t known[] = {127, -127, 1, -1, 2, -2, 127, -127};
            require(std::equal(std::begin(known), std::end(known), q_ref.values.begin()), "CPU rounding oracle failed ties");
        }
        const size_t bytes = 3 * (data.q.size() + data.k.size())
                           + 2 * (q_ref.scales.size() + k_ref.scales.size()) + 12;
        size_t free_bytes, total_bytes;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
        require(bytes <= free_bytes / 4 && bytes <= 64 * 1024 * 1024, "quantizer exceeds VRAM budget");
        std::vector<std::int8_t> q_expected(data.q.size() + 2, -128), k_expected(data.k.size() + 2, -128);
        std::copy(q_ref.values.begin(), q_ref.values.end(), q_expected.begin() + 1);
        std::copy(k_ref.values.begin(), k_ref.values.end(), k_expected.begin() + 1);
        std::vector<__half> qs_expected(q_ref.scales.size() + 2, __float2half(60000.0f));
        std::vector<__half> ks_expected(k_ref.scales.size() + 2, __float2half(60000.0f));
        std::copy(q_ref.scales.begin(), q_ref.scales.end(), qs_expected.begin() + 1);
        std::copy(k_ref.scales.begin(), k_ref.scales.end(), ks_expected.begin() + 1);
        DeviceArray<__half> q(data.q.size()), k(data.k.size()), qs(qs_expected.size()), ks(ks_expected.size());
        DeviceArray<std::int8_t> q8(q_expected.size()), k8(k_expected.size());
        Stream stream;
        cuda_check(cudaMemcpyAsync(q.data, data.q.data(), data.q.size() * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
        cuda_check(cudaMemcpyAsync(k.data, data.k.data(), data.k.size() * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
        cuda_check(cudaMemsetAsync(q8.data, 0x80, q_expected.size(), stream.value));
        cuda_check(cudaMemsetAsync(k8.data, 0x80, k_expected.size(), stream.value));
        const std::vector<__half> scale_guard(std::max(qs_expected.size(), ks_expected.size()), __float2half(60000.0f));
        cuda_check(cudaMemcpyAsync(qs.data, scale_guard.data(), qs_expected.size() * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
        cuda_check(cudaMemcpyAsync(ks.data, scale_guard.data(), ks_expected.size() * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
        cuda_check(quantize_tilemix_qk(q.data, k.data, q8.data + 1, k8.data + 1, qs.data + 1, ks.data + 1,
                                      test.m, test.n, test.d, stream.value));
        cuda_check(cudaStreamSynchronize(stream.value));
        same_device_bytes(q8.data, q_expected, "quantized Q bytes / guards mismatch");
        same_device_bytes(k8.data, k_expected, "quantized K bytes / guards mismatch");
        same_device_bytes(qs.data, qs_expected, "quantized Q scales / guards mismatch");
        same_device_bytes(ks.data, ks_expected, "quantized K scales / guards mismatch");
        same_device_bytes(q.data, data.q, "quantizer modified Q");
        same_device_bytes(k.data, data.k, "quantizer modified K");
        std::printf("PASS quantization M=%d N=%d d=%d pattern=%s exact bytes/scales/guards\n", test.m, test.n, test.d, test.pattern);
    }
}

void routing_case(const Shape& shape, int group_size, const char* pattern, bool trace = true) {
    const size_t query_tiles = (shape.m + kBlockM - 1) / kBlockM;
    const size_t key_tiles = (shape.n + kBlockN - 1) / kBlockN;
    const size_t groups = (key_tiles + static_cast<size_t>(group_size) - 1) / group_size;
    const size_t words_per_row = (groups + 63) / 64;
    std::printf("Routing M=%d N=%d d=%d g=%d words_per_row=%zu pattern=%s trace=%s\n",
                shape.m, shape.n, shape.d, group_size, words_per_row, pattern, trace ? "yes" : "no");
    const bool implicit = std::string(pattern) == "implicit-fp16";
    std::vector<std::uint64_t> words(query_tiles * words_per_row, ~std::uint64_t{0});
    std::vector<std::uint8_t> expected_bits(query_tiles * key_tiles);
    // Build logical group decisions first, then expand groups without n/g.
    // Mixed words differ across rows AND consecutive 64-group chunks.
    for (size_t row = 0; row < query_tiles; ++row) {
        size_t tile = 0;
        for (size_t group = 0; group < groups; ++group) {
            const bool bit = std::string(pattern) == "int8-requested" ||
                (std::string(pattern) == "mixed" && (group + row + group / 64) % 2 == 0);
            const std::uint64_t flag = std::uint64_t{1} << (group % 64);
            if (bit) words[row * words_per_row + group / 64] |= flag;
            else words[row * words_per_row + group / 64] &= ~flag;
            for (int repeat = 0; repeat < group_size && tile < key_tiles; ++repeat, ++tile) {
                expected_bits[row * key_tiles + tile] = bit;
            }
        }
    }
    require(words.size() == tilemix_routing_word_count(shape.m, shape.n, group_size),
            "routing layout disagrees with independent group packing");
    auto data = inputs(shape);
    const size_t count = data.q.size(), kv_count = data.k.size();
    const auto q8_ref = reference_quantize(data.q, shape.m, shape.d, 128);
    const auto k8_ref = reference_quantize(data.k, shape.n, shape.d, 64);
    const size_t score_count = static_cast<size_t>(shape.m) * shape.n;
    const size_t bytes = sizeof(__half) * (count + 2 * kv_count) + sizeof(float) * (count + 2)
                       + words.size() * sizeof(std::uint64_t) + expected_bits.size() + 2
                       + count + kv_count + sizeof(__half) * (q8_ref.scales.size() + k8_ref.scales.size())
                       + sizeof(std::int32_t) * (score_count + 2);
    size_t free_bytes, total_bytes;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
    require(bytes <= free_bytes / 4 && bytes <= 64 * 1024 * 1024, "routing case exceeds VRAM budget");
    std::vector<double> expected(count);
    reference_attention(data.q.data(), data.k.data(), data.v.data(), expected.data(), shape.m, shape.n, shape.d);
    constexpr float guard = -12345.25f;
    std::vector<float> actual(count + 2, std::numeric_limits<float>::quiet_NaN());
    actual.front() = actual.back() = guard;
    constexpr std::uint8_t trace_guard = 0xa5;
    std::vector<std::uint8_t> decoded(expected_bits.size() + 2, trace_guard);
    DeviceArray<__half> q(count), k(kv_count), v(kv_count);
    DeviceArray<float> output(actual.size());
    DeviceArray<std::uint64_t> mask(words.size());
    DeviceArray<std::uint8_t> bits(decoded.size());
    DeviceArray<std::int8_t> q8(count), k8(kv_count);
    DeviceArray<__half> q_scales(q8_ref.scales.size()), k_scales(k8_ref.scales.size());
    constexpr std::int32_t dot_guard = 0x2468ace;
    std::vector<std::int32_t> dots(score_count + 2, dot_guard);
    DeviceArray<std::int32_t> raw_dots(dots.size());
    Stream stream(shape.default_stream);
    cuda_check(cudaMemcpyAsync(q.data, data.q.data(), count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(k.data, data.k.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(v.data, data.v.data(), kv_count * sizeof(__half), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(mask.data, words.data(), words.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(output.data, actual.data(), actual.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(bits.data, decoded.data(), decoded.size(), cudaMemcpyHostToDevice, stream.value));
    cuda_check(cudaMemcpyAsync(raw_dots.data, dots.data(), dots.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream.value));
    cuda_check(quantize_tilemix_qk(q.data, k.data, q8.data, k8.data, q_scales.data, k_scales.data,
                                  shape.m, shape.n, shape.d, stream.value));
    const float scale = 1.0f / std::sqrt(static_cast<float>(shape.d));
    // All-FP16 masks must preserve the original GPU result exactly.
    std::vector<float> baseline(actual.size());
    cuda_check(launch_fused_fp16_attention(q.data, k.data, v.data, output.data + 1,
                                          shape.m, shape.n, shape.d, scale, stream.value));
    cuda_check(cudaMemcpyAsync(baseline.data(), output.data, baseline.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    cuda_check(cudaStreamSynchronize(stream.value));
    const TileMixRouting routing{implicit ? nullptr : mask.data, implicit ? 0 : words.size(), group_size,
                                trace ? bits.data + 1 : nullptr};
    const TileMixQuantized quantized{q8.data, k8.data, q_scales.data, k_scales.data, raw_dots.data + 1};
    cuda_check(launch_fused_fp16_attention(q.data, k.data, v.data, output.data + 1,
                                          shape.m, shape.n, shape.d, scale, stream.value, routing, quantized));
    cuda_check(cudaMemcpyAsync(actual.data(), output.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    cuda_check(cudaMemcpyAsync(decoded.data(), bits.data, decoded.size(), cudaMemcpyDeviceToHost, stream.value));
    cuda_check(cudaMemcpyAsync(dots.data(), raw_dots.data, dots.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream.value));
    cuda_check(cudaStreamSynchronize(stream.value));
    require(actual.front() == guard && actual.back() == guard, "routed output guard overwritten");
    if (std::none_of(expected_bits.begin(), expected_bits.end(), [](auto bit) { return bit != 0; })) {
        require(std::memcmp(actual.data(), baseline.data(), actual.size() * sizeof(float)) == 0,
                "all-FP16 routing changed baseline outputs");
    }
    require(dots.front() == dot_guard && dots.back() == dot_guard, "INT32 dot guard overwritten");
    for (int query = 0; query < shape.m; ++query) {
        for (int key = 0; key < shape.n; ++key) {
            const bool int8 = expected_bits[static_cast<size_t>(query / kBlockM) * key_tiles + key / kBlockN];
            const auto expected_dot = int8 ? reference_int8_dot(q8_ref, k8_ref, query, key, shape.d) : dot_guard;
            if (dots[static_cast<size_t>(query) * shape.n + key + 1] != expected_dot) {
                std::fprintf(stderr, "INT32 dot mismatch query=%d key=%d\n", query, key);
                throw std::runtime_error("signed INT8 WMMA dot mismatch");
            }
        }
    }
    require(decoded.front() == trace_guard && decoded.back() == trace_guard, "routing trace guard overwritten");
    for (size_t i = 0; i < expected_bits.size(); ++i) {
        const std::uint8_t expected_bit = trace ? expected_bits[i] : trace_guard;
        if (decoded[i + 1] != expected_bit) {
            std::fprintf(stderr, "Routing mismatch row=%zu tile=%zu g=%d: got %u expected %u\n",
                         i / key_tiles, i % key_tiles, group_size, decoded[i + 1], expected_bit);
            throw std::runtime_error("device routing-bit decoding failed");
        }
    }
    std::vector<std::uint64_t> mask_after(words.size());
    cuda_check(cudaMemcpy(mask_after.data(), mask.data, mask_after.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    require(mask_after == words, "routing mask modified by kernel");
    same_device_bytes(q8.data, q8_ref.values, "Q8 quantization/immutability mismatch");
    same_device_bytes(k8.data, k8_ref.values, "K8 quantization/immutability mismatch");
    same_device_bytes(q_scales.data, q8_ref.scales, "Q scale mismatch");
    same_device_bytes(k_scales.data, k8_ref.scales, "K scale mismatch");
    const auto mixed_expected = reference_tilemix(data.q, data.k, data.v, q8_ref, k8_ref, expected_bits,
                                                 shape.m, shape.n, shape.d, kBlockM, kBlockN, scale);
    compare(std::vector<float>(actual.begin() + 1, actual.end() - 1), mixed_expected,
            "mixed FP16/INT8 vs independent semantic oracle", 1e-5, 1e-4);
    double max_abs = 0.0, max_rel = 0.0, squared = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double error = std::fabs(actual[i + 1] - expected[i]);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(expected[i]), 1e-5));
        squared += error * error;
    }
    std::printf("Approximation vs dense FP16-input oracle (reported only): max_abs_error=%.9g "
                "max_rel_error=%.9g rms_error=%.9g\n", max_abs, max_rel, std::sqrt(squared / count));
}

void routing_checks() {
    struct Case { Shape shape; int group; };
    const Case cases[] = {{{35, 67, 17, "random"}, 2},
        {{17, 2048, 5, "random"}, 1}, {{17, 2049, 5, "random"}, 1},
        {{17, 4099, 5, "random"}, 2}, {{35, 163, 17, "random"}, 3},
        {{17, 65, 17, "random"}, std::numeric_limits<int>::max()},
        {{17, 4097, 3, "random"}, 1}, {{129, 131, 17, "block-outliers", true}, 1},
        {{17, 65, 17, "zeros"}, 2}, {{17, 65, 1, "rising"}, 1},
        {{17, 65, 1, "falling"}, 1}, {{17, 129, 1, "alternating"}, 1},
        {{17, 65, 1, "late-extreme"}, 1}, {{17, 33, 128, "max-int8"}, 1}};
    int checked = 0;
    for (const auto& test : cases) {
        for (const char* pattern : {"fp16", "int8-requested", "mixed"}) {
            routing_case(test.shape, test.group, pattern);
            ++checked;
        }
    }
    for (int d : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128}) {
        for (const char* pattern : {"fp16", "int8-requested", "mixed"}) {
            routing_case({3, 33, d, "random"}, 1, pattern);
            ++checked;
        }
    }
    routing_case({17, 2051, 5, "random"}, 1, "implicit-fp16");
    routing_case({35, 67, 17, "random"}, 2, "mixed", false);
    std::printf("PASS %d routing cases: exact signed INT8 dots, block scales, mixed online-softmax state and mask decoding\n", checked + 2);
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
    quantization_checks();
    routing_checks();
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
        std::puts("TileMix tests: FP16/INT8 score paths, 128/64-row Q/K quantization, shared FP32 softmax state.");
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
