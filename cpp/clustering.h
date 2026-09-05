#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "mlann.h"

namespace mlann_detail {

// L2 subset of LoRANN's clustering.h: sampled Lloyd iterations, blocked
// assignment, and symmetric splitting of empty clusters. The seed controls
// sampling and ties are resolved by centroid ID.
class KMeans {
 public:
  KMeans(int n_clusters, int iterations = 25, int samples_per_cluster = 256,
         uint32_t seed = 42)
      : n_clusters_(n_clusters), iterations_(iterations),
        samples_per_cluster_(samples_per_cluster), seed_(seed) {
    if (n_clusters <= 0 || iterations <= 0)
      throw std::invalid_argument("n_lists and iterations must be positive");
  }

  RowMatrix train(const Eigen::Ref<const RowMatrix> &data) const {
    if (data.rows() < n_clusters_ || data.cols() == 0 || !data.allFinite())
      throw std::invalid_argument("Clustering needs finite data and n_vectors >= n_lists");
    std::mt19937 generator(seed_);
    std::vector<int> order(data.rows());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), generator);
    const int sample_size = samples_per_cluster_ <= 0 ? int(data.rows()) :
        int(std::min<int64_t>(data.rows(), int64_t(samples_per_cluster_) * n_clusters_));
    RowMatrix sample(sample_size, data.cols());
    for (int i = 0; i < sample_size; ++i) sample.row(i) = data.row(order[i]);
    RowMatrix centroids = sample.topRows(n_clusters_);
    for (int iteration = 0; iteration < iterations_; ++iteration) {
      const auto assignments = assign(sample, centroids);
      std::vector<int> sizes(n_clusters_, 0);
      centroids.setZero();
      for (int i = 0; i < sample_size; ++i) {
        centroids.row(assignments[i]) += sample.row(i);
        ++sizes[assignments[i]];
      }
      for (int c = 0; c < n_clusters_; ++c)
        if (sizes[c]) centroids.row(c) /= float(sizes[c]);
      for (int c = 0; c < n_clusters_; ++c) {
        if (sizes[c]) continue;
        // Largest donor avoids rejection loops for identical points or n == L.
        const int donor = int(std::max_element(sizes.begin(), sizes.end()) - sizes.begin());
        centroids.row(c) = centroids.row(donor);
        for (int d = 0; d < data.cols(); ++d) {
          const float delta = (d % 2 ? -1.0f : 1.0f) *
                              std::max(std::abs(centroids(donor, d)), 1e-6f) / 1024.0f;
          centroids(c, d) += delta;
          centroids(donor, d) -= delta;
        }
        sizes[c] = sizes[donor] / 2;
        sizes[donor] -= sizes[c];
      }
    }
    return centroids;
  }

  static std::vector<int> assign(const Eigen::Ref<const RowMatrix> &data,
                                  const RowMatrix &centroids) {
    std::vector<int> assignments(data.rows());
    const Eigen::VectorXf norms = centroids.rowwise().squaredNorm();
    // Bound the temporary similarity matrix per worker, as in LoRANN.
    const int block = std::clamp(int((1 << 20) / (sizeof(float) * centroids.rows())), 128, 1024);
#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int begin = 0; begin < data.rows(); begin += block) {
      const int rows = std::min(block, int(data.rows()) - begin);
      RowMatrix dots = data.middleRows(begin, rows) * centroids.transpose();
      for (int i = 0; i < rows; ++i) {
        Eigen::Index closest;
        (norms.array() - 2.0f * dots.row(i).transpose().array()).minCoeff(&closest);
        assignments[begin + i] = int(closest);
      }
    }
    return assignments;
  }

 private:
  int n_clusters_, iterations_, samples_per_cluster_;
  uint32_t seed_;
};

}  // namespace mlann_detail
