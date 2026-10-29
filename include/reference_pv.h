#pragma once

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

inline void reference_pv(const float* P, const __half* V, double* O,
                         int M, int N, int d) {
    if (!P || !V || !O || M <= 0 || N <= 0 || d <= 0) {
        throw std::invalid_argument("PV requires nonempty P, V, and O matrices");
    }
    std::vector<double> row(d);
    for (int query = 0; query < M; ++query) {
        std::fill(row.begin(), row.end(), 0.0);
        for (int key = 0; key < N; ++key) {
            const double weight = P[static_cast<std::size_t>(query) * N + key];
            if (!std::isfinite(weight)) throw std::invalid_argument("PV requires finite P");
            for (int feature = 0; feature < d; ++feature) {
                const double value = __half2float(V[static_cast<std::size_t>(key) * d + feature]);
                if (!std::isfinite(value)) throw std::invalid_argument("PV requires finite V");
                row[feature] += weight * value;
            }
        }
        for (int feature = 0; feature < d; ++feature) {
            O[static_cast<std::size_t>(query) * d + feature] = row[feature];
        }
    }
}
