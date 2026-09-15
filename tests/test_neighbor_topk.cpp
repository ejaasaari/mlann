#include "../cpp/neighbor-query.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int main() {
  using namespace mlann_detail;
  std::minstd_rand rng(17);
  std::uniform_real_distribution<float> random(-1, 1);
  for (size_t dim : {1, 7, 16, 17, 63, 64, 65, 200, 203}) {
    std::vector<float> corpus(1000 * dim), query(dim);
    for (auto &x : corpus) x = random(rng);
    for (auto &x : query) x = random(rng);
    std::vector<uint32_t> ids(1000);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    for (size_t count : {0, 1, 3, 4, 7, 99, 100, 101, 401, 1000}) {
      for (auto metric : {OneToManyMetric::IP, OneToManyMetric::L2}) {
        std::vector<float> scores(count), by_id(1000);
        compute_neighbor_one_to_many(query.data(), corpus.data(), dim, ids.data(), count,
                                     metric, scores.data());
        for (size_t i = 0; i < count; ++i) by_id[ids[i]] = scores[i];
        std::sort(scores.begin(), scores.end(), [metric](float a, float b) {
          return metric == OneToManyMetric::IP ? a > b : a < b;
        });
        for (size_t k : {0, 1, 10, 100, 1200}) {
          const size_t keep = std::min(k, count);
          std::vector<ScoredCandidate> output(keep + 1, {123.f, 0xdeadbeefU});
          compute_neighbor_topk(query.data(), corpus.data(), dim, ids.data(), count, k,
                                 metric, output.data());
          std::vector<bool> seen(1000, false);
          for (size_t i = 0; i < keep; ++i) {
            assert(output[i].score == scores[i]);
            assert(output[i].label < 1000 && !seen[output[i].label]);
            assert(std::find(ids.begin(), ids.begin() + count, output[i].label) != ids.begin() + count);
            assert(output[i].score == by_id[output[i].label]);
            seen[output[i].label] = true;
          }
          assert(output[keep].score == 123.f && output[keep].label == 0xdeadbeefU);
        }
      }
    }
    // Every candidate ties. Any distinct input IDs are valid, and no output
    // beyond the requested k records may be touched.
    std::fill(query.begin(), query.end(), 0.f);
    ScoredCandidate tied[11]; tied[10] = {123.f, 0xdeadbeefU};
    compute_neighbor_topk(query.data(), corpus.data(), dim, ids.data(), 1000, 10,
                           OneToManyMetric::IP, tied);
    for (size_t i = 0; i < 10; ++i) assert(tied[i].score == 0.f);
    assert(tied[10].score == 123.f && tied[10].label == 0xdeadbeefU);
  }
  std::cout << "Exact top-k scores and valid IDs match full scoring; tails, metrics, k and output bounds pass\n";
}
