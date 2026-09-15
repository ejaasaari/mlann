#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace mlann_detail {

inline uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Reservoir sampling with an independent random generator.
inline void sample_unique(int n, int k, std::vector<uint32_t> &reservoir) {
  std::random_device rd;
  std::minstd_rand generator(rd());

  reservoir.resize(k);
  std::iota(reservoir.begin(), reservoir.end(), 0);

  for (int i = k; i < n; ++i) {
    std::uniform_int_distribution<int> distribution(0, i);
    int j = distribution(generator);

    if (j < k) {
      reservoir[j] = i;
    }
  }
}

inline std::vector<uint32_t> sample_unique(int n, int k) {
  std::vector<uint32_t> reservoir;
  sample_unique(n, k, reservoir);
  return reservoir;
}

// Sample distinct row offsets without scanning large nodes.
template <typename Generator>
inline std::vector<int> sample(int n, int k, Generator &rng) {
  std::vector<int> result;
  result.reserve(k);
  if (k < n / 4) {
    std::unordered_set<int> selected;
    selected.reserve(k);
    for (int i = n - k; i < n; ++i) {
      int j = std::uniform_int_distribution<int>(0, i)(rng);
      if (!selected.insert(j).second) {
        selected.insert(i);
        j = i;
      }
      result.push_back(j);
    }
  } else {
    result.resize(n);
    std::iota(result.begin(), result.end(), 0);
    for (int i = 0; i < k; ++i) {
      std::swap(result[i], result[std::uniform_int_distribution<int>(i, n - 1)(rng)]);
    }
    result.resize(k);
  }
  return result;
}

}  // namespace mlann_detail
