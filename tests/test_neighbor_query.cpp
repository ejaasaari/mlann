#include "../cpp/neighbor-query.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  struct Scored { float score; uint32_t label; };
  using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  for (int d : {1, 7, 16, 17, 63, 64, 65, 200, 203}) {
    for (int n : {0, 1, 3, 4, 15, 16, 31, 32, 64, 257}) {
      Matrix corpus = Matrix::Random(n, d);
      Eigen::VectorXf query = Eigen::VectorXf::Random(d);
      std::vector<uint32_t> ids(n);
      std::iota(ids.begin(), ids.end(), 0);
      std::reverse(ids.begin(), ids.end());
      for (float scale : {0.f, 1e-15f, 1.f, 1e15f}) {
        Matrix data = corpus * scale;
        Eigen::VectorXf q = query * scale;
        for (auto metric : {mlann_detail::OneToManyMetric::IP, mlann_detail::OneToManyMetric::L2}) {
          std::vector<float> expected(n), actual(n);
          std::vector<Scored> strided(n);
          for (int i = 0; i < n; ++i) strided[i].label = ids[i];
          mlann_detail::compute_one_to_many(q.data(), data.data(), d, ids.data(), n, metric, expected.data());
          mlann_detail::compute_neighbor_one_to_many(q.data(), data.data(), d, ids.data(), n, metric, actual.data());
          mlann_detail::compute_neighbor_scores(q.data(), data.data(), d, ids.data(), n, metric,
              {reinterpret_cast<unsigned char *>(strided.data()), sizeof(Scored)});
          assert(expected == actual);
          for (int i = 0; i < n; ++i) {
            assert(expected[i] == strided[i].score);
            assert(strided[i].label == ids[i]);
          }
        }
      }
    }
  }
  std::cout << "Native and strided Neighbor scores exactly match original IP/L2 kernels; dimensions, tails and scales pass\n";
}
