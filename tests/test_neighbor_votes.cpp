#include "../cpp/neighbor-query.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int main() {
  std::minstd_rand random(17);
  for (int count : {0, 1, 15, 16, 17, 31, 32, 33, 64, 257}) {
    std::vector<uint32_t> labels(count);
    std::iota(labels.begin(), labels.end(), 0);
    std::shuffle(labels.begin(), labels.end(), random);
    std::vector<float> weights(count);
    for (int i = 0; i < count; ++i) weights[i] = float(1 + random() % 100) / 10000;
    for (float threshold : {-1e8f, -1.f, 0.f, .001f, .01f, 1.f}) {
      std::vector<float> reference(300), actual(300);
      std::vector<uint32_t> expected, elected;
      for (int tree = 0; tree < 4; ++tree) {
        for (int i = 0; i < count; ++i) {
          if ((reference[labels[i]] += weights[i]) >= threshold) {
            expected.push_back(labels[i]);
            reference[labels[i]] = -9999999.f;
          }
        }
        mlann_detail::accumulate_neighbor_votes(labels, weights, actual.data(), threshold, elected);
        assert(reference == actual);
        assert(expected == elected);
        std::reverse(labels.begin(), labels.end());
      }
    }
  }
  std::cout << "SIMD/scalar vote values and election order match across leaves and tails\n";
}
