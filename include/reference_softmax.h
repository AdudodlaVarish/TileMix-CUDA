#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

inline void reference_softmax(const float* scores, float* probabilities, int M, int N) {
    if (!scores || !probabilities || M <= 0 || N <= 0) {
        throw std::invalid_argument("softmax requires nonempty score/probability matrices");
    }
    for (int row = 0; row < M; ++row) {
        const std::size_t begin = static_cast<std::size_t>(row) * N;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < N; ++col) {
            const double score = scores[begin + col];
            if (!std::isfinite(score)) throw std::invalid_argument("softmax requires finite logits");
            maximum = std::max(maximum, score);
        }
        double denominator = 0.0;
        for (int col = 0; col < N; ++col) {
            denominator += std::exp(static_cast<double>(scores[begin + col]) - maximum);
        }
        for (int col = 0; col < N; ++col) {
            probabilities[begin + col] = static_cast<float>(
                std::exp(static_cast<double>(scores[begin + col]) - maximum) / denominator);
        }
    }
}
