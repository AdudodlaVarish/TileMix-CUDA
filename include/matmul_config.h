#pragma once

inline constexpr int kBlockM = 16;
inline constexpr int kBlockN = 32;
inline constexpr int kTileK = 16;
inline constexpr int kThreads = 64;
inline constexpr int kWarpSize = 32;

static_assert(kBlockM == 16 && kTileK == 16,
              "The kernel uses 16x16x16 WMMA fragments.");
static_assert(kBlockN > 0 && kBlockN % 16 == 0,
              "Each warp must own 16 key columns.");
static_assert(kWarpSize == 32 && kThreads == (kBlockN / 16) * kWarpSize,
              "Launch one warp per 16 key columns.");
