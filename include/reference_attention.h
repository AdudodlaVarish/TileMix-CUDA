#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

inline void reference_attention(const __half* Q, const __half* K, const __half* V,
                                double* O, int M, int N, int d) {
    if (!Q || !K || !V || !O || M <= 0 || N <= 0 || d <= 0) {
        throw std::invalid_argument("attention requires nonempty Q, K, V, and O");
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    std::vector<double> weights(N), output_row(d);
    for (int query = 0; query < M; ++query) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (int key = 0; key < N; ++key) {
            double dot = 0.0;
            for (int feature = 0; feature < d; ++feature) {
                const double q = __half2float(Q[static_cast<std::size_t>(query) * d + feature]);
                const double k = __half2float(K[static_cast<std::size_t>(key) * d + feature]);
                if (!std::isfinite(q) || !std::isfinite(k)) {
                    throw std::invalid_argument("attention requires finite Q/K");
                }
                dot += q * k;
            }
            weights[key] = dot * scale;
            maximum = std::max(maximum, weights[key]);
        }
        double denominator = 0.0;
        for (double& weight : weights) {
            weight = std::exp(weight - maximum);
            denominator += weight;
        }
        std::fill(output_row.begin(), output_row.end(), 0.0);
        for (int key = 0; key < N; ++key) {
            const double probability = weights[key] / denominator;
            for (int feature = 0; feature < d; ++feature) {
                const double value = __half2float(V[static_cast<std::size_t>(key) * d + feature]);
                if (!std::isfinite(value)) throw std::invalid_argument("attention requires finite V");
                output_row[feature] += probability * value;
            }
        }
        for (int feature = 0; feature < d; ++feature) {
            O[static_cast<std::size_t>(query) * d + feature] = output_row[feature];
        }
    }
}
