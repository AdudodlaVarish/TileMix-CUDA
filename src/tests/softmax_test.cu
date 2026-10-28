#include "reference_softmax.h"
#include "softmax.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct DeviceArray {
    float* data = nullptr;
    explicit DeviceArray(size_t count) {
        cuda_check(cudaMalloc(&data, count * sizeof(float)));
    }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

void compare(const std::vector<float>& actual, const std::vector<double>& expected,
             int rows, int columns, const char* label, double atol, double rtol) {
    double max_abs = 0.0, max_rel = 0.0, squared = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || actual[i] < 0.0f || actual[i] > 1.0f ||
            error > atol + rtol * std::fabs(expected[i])) {
            std::fprintf(stderr, "%s mismatch at %zu: got %.9g expected %.17g\n",
                         label, i, actual[i], expected[i]);
            throw std::runtime_error("softmax comparison failed");
        }
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(expected[i]), atol));
        squared += error * error;
    }
    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (int col = 0; col < columns; ++col) sum += actual[static_cast<size_t>(row) * columns + col];
        if (std::fabs(sum - 1.0) > 1e-5) throw std::runtime_error("probability row sum differs from 1");
    }
    std::printf("PASS %s M=%d N=%d atol=%.1g rtol=%.1g max_abs_error=%.9g "
                "max_rel_error=%.9g rms_error=%.9g\n", label, rows, columns, atol, rtol,
                max_abs, max_rel, std::sqrt(squared / actual.size()));
}

void cpu_checks() {
    const auto check = [](const std::vector<float>& logits, const std::vector<double>& expected,
                          int rows, int columns, const char* label) {
        std::vector<float> actual(logits.size());
        reference_softmax(logits.data(), actual.data(), rows, columns);
        compare(actual, expected, rows, columns, label, 1e-6, 1e-6);
    };
    check({12345}, {1}, 1, 1, "CPU single key");
    check({0, 0, 0}, {1.0 / 3, 1.0 / 3, 1.0 / 3}, 1, 3, "CPU uniform zeros");
    check({0, std::log(3.0f)}, {0.25, 0.75}, 1, 2, "CPU analytic probabilities");
    const double first = 1.0 / (1.0 + std::exp(-1.0));
    check({10000, 9999, -10000}, {first, 1.0 - first, 0}, 1, 3, "CPU extreme logits");
    const double total = std::exp(-2.0) + std::exp(-1.0) + 1.0;
    const std::vector<double> shifted = {std::exp(-2.0) / total, std::exp(-1.0) / total, 1.0 / total,
                                       std::exp(-2.0) / total, std::exp(-1.0) / total, 1.0 / total};
    check({1, 2, 3, 1001, 1002, 1003}, shifted, 2, 3, "CPU row shift invariance");
    for (const float invalid : {std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()}) {
        float output;
        bool rejected = false;
        try { reference_softmax(&invalid, &output, 1, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("nonfinite logit was accepted");
    }
    float input = 0.0f, output;
    bool rejected = false;
    try { reference_softmax(&input, &output, 1, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("empty row was accepted");
    std::puts("PASS CPU invalid-input checks");
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
    std::printf("GPU=%s cc=%d.%d free_bytes=%zu total_bytes=%zu driver=%d runtime=%d "
                "FP32_S_P contiguous threads=%d\n", properties.name, properties.major,
                properties.minor, free_bytes, total_bytes, driver, runtime, kSoftmaxThreads);
    constexpr int rows = 3;
    std::mt19937 generator(29);
    for (const int columns : {1, 3, 31, 32, 33, 255, 256, 257, 1003, 2049}) {
        const size_t count = static_cast<size_t>(rows) * columns;
        const size_t bytes = count * sizeof(float);
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (2 * bytes > free_bytes / 4) throw std::runtime_error("insufficient available VRAM");
        std::vector<float> logits(count), expected(count), actual(count);
        for (int col = 0; col < columns; ++col) {
            logits[col] = 0.0f;
            logits[columns + col] = (static_cast<int>(generator() % 10001) - 5000) / 1000.0f;
            logits[2 * columns + col] = 10000.0f - col % 5;
        }
        if (columns == 3) {
            logits[2 * columns + 1] = 0.0f;
            logits[2 * columns + 2] = -10000.0f;
        }
        reference_softmax(logits.data(), expected.data(), rows, columns);
        DeviceArray device_scores(count), device_probabilities(count);
        cuda_check(cudaMemcpy(device_scores.data, logits.data(), bytes, cudaMemcpyHostToDevice));
        cuda_check(cudaMemset(device_probabilities.data, 0xff, bytes));
        row_softmax<<<rows, kSoftmaxThreads>>>(device_scores.data, device_probabilities.data,
                                             rows, columns);
        cuda_check(cudaGetLastError());
        cuda_check(cudaDeviceSynchronize());
        cuda_check(cudaMemcpy(actual.data(), device_probabilities.data, bytes, cudaMemcpyDeviceToHost));
        compare(actual, std::vector<double>(expected.begin(), expected.end()), rows, columns,
                "CUDA softmax", 1e-6, 1e-4);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool cpu_only = argc == 2 && std::string(argv[1]) == "--cpu-only";
        if (argc != 1 && !cpu_only) throw std::invalid_argument("usage: softmax_test [--cpu-only]");
        cpu_checks();
        if (!cpu_only) gpu_checks();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
