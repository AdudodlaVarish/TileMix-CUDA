#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

struct ReferenceQuantized {
    std::vector<std::int8_t> values;
    std::vector<__half> scales;
};

inline ReferenceQuantized reference_quantize(const std::vector<__half>& input,
                                             int rows, int d, int block_rows) {
    ReferenceQuantized result{std::vector<std::int8_t>(input.size()),
                             std::vector<__half>((rows - 1) / block_rows + 1)};
    for (int first = 0; first < rows; first += block_rows) {
        const int last = std::min(rows, first + block_rows);
        float maximum = 0.0f;
        for (int row = first; row < last; ++row) {
            for (int f = 0; f < d; ++f) {
                maximum = std::max(maximum, std::fabs(__half2float(input[static_cast<size_t>(row) * d + f])));
            }
        }
        const float scale = maximum / 127.0f;
        result.scales[first / block_rows] = __float2half_rn(scale);
        for (int row = first; row < last; ++row) {
            for (int f = 0; f < d; ++f) {
                const size_t index = static_cast<size_t>(row) * d + f;
                const float x = scale == 0.0f ? 0.0f : __half2float(input[index]) / scale;

                const float shifted = x < 0.0f ? x - 0.5f : x + 0.5f;
                result.values[index] = static_cast<std::int8_t>(std::clamp(static_cast<int>(shifted), -127, 127));
            }
        }
    }
    return result;
}

inline std::int32_t reference_int8_dot(const ReferenceQuantized& q, const ReferenceQuantized& k,
                                      int query, int key, int d) {
    std::int64_t sum = 0;
    for (int f = 0; f < d; ++f) {
        sum += static_cast<int>(q.values[static_cast<size_t>(query) * d + f])
             * static_cast<int>(k.values[static_cast<size_t>(key) * d + f]);
    }
    return static_cast<std::int32_t>(sum);  // d<=128 bounds magnitude by 2,064,512.
}

inline std::vector<double> reference_tilemix(const std::vector<__half>& q,
                                            const std::vector<__half>& k,
                                            const std::vector<__half>& v,
                                            const ReferenceQuantized& q8,
                                            const ReferenceQuantized& k8,
                                            const std::vector<std::uint8_t>& routes,
                                            int M, int N, int d, int block_m, int block_n,
                                            float score_scale) {
    const int key_tiles = (N + block_n - 1) / block_n;
    std::vector<double> result(static_cast<size_t>(M) * d), scores(N);
    for (int query = 0; query < M; ++query) {
        for (int key = 0; key < N; ++key) {
            double dot = 0.0;
            if (routes[static_cast<size_t>(query / block_m) * key_tiles + key / block_n]) {
                dot = reference_int8_dot(q8, k8, query, key, d);
                dot *= static_cast<double>(__half2float(q8.scales[query / 128]));
                dot *= static_cast<double>(__half2float(k8.scales[key / 64]));
            } else {
                for (int f = 0; f < d; ++f) {
                    dot += static_cast<double>(__half2float(q[static_cast<size_t>(query) * d + f]))
                         * static_cast<double>(__half2float(k[static_cast<size_t>(key) * d + f]));
                }
            }
            scores[key] = dot * static_cast<double>(score_scale);
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double denominator = 0.0;
        for (double& score : scores) { score = std::exp(score - maximum); denominator += score; }
        for (int key = 0; key < N; ++key) {
            const double probability = scores[key] / denominator;
            for (int f = 0; f < d; ++f) {
                result[static_cast<size_t>(query) * d + f] += probability
                    * __half2float(v[static_cast<size_t>(key) * d + f]);
            }
        }
    }
    return result;
}
