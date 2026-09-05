#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "clustering.h"

// Both variants route queries by squared L2 to the same kind of flat partition.
// Scoring candidates can independently use L2 or IP (normalize for cosine).
class IVF : public MLANN {
 public:
  IVF(const float *data, int n, int dim, bool vector_labels = false)
      : MLANN(data, n, dim), vector_labels_(vector_labels) {}

  void build(int n_lists, const Eigen::Ref<const UIntRowMatrix> &knn,
             const Eigen::Ref<const RowMatrix> &train, int iterations = 25,
             int samples_per_cluster = 256, uint32_t seed = 42) {
    validate_training(knn, train);
    auto centroids = mlann_detail::KMeans(n_lists, iterations, samples_per_cluster, seed).train(corpus);
    build_from_centroids(centroids, knn, train);
  }

  // Also permits exactly the same fixed partition to be reused across variants.
  void build_from_centroids(const RowMatrix &centroids,
                            const Eigen::Ref<const UIntRowMatrix> &knn,
                            const Eigen::Ref<const RowMatrix> &train) {
    validate_training(knn, train);
    if (centroids.rows() <= 0 || centroids.rows() > n_corpus ||
        centroids.cols() != dim || !centroids.allFinite())
      throw std::invalid_argument("Invalid centroids");
    centroids_ = centroids;
    const int count = centroids.rows();
    const auto database_cells = mlann_detail::KMeans::assign(corpus, centroids_);
    lists_.assign(count, {});
    for (int i = 0; i < n_corpus; ++i) lists_[database_cells[i]].push_back(i);
    if (!vector_labels_) {
      n_trees = 1;
      return;
    }
    const auto query_cells = mlann_detail::KMeans::assign(train, centroids_);
    std::vector<std::vector<int>> cell_queries(count);
    for (int i = 0; i < train.rows(); ++i) cell_queries[query_cells[i]].push_back(i);
    postings_.assign(count, {});
    cell_sizes_.resize(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int cell = 0; cell < count; ++cell) {
      cell_sizes_[cell] = cell_queries[cell].size();
      std::unordered_map<uint32_t, uint64_t> frequencies;
      for (int row : cell_queries[cell]) {
        for (int col = 0; col < knn.cols(); ++col) {
          const uint32_t id = knn(row, col);
          ++frequencies[id];
        }
      }
      std::vector<Posting> postings;
      postings.reserve(frequencies.size());
      for (const auto &item : frequencies)
        postings.push_back({item.first, float(double(item.second) / cell_sizes_[cell])});
      miniselect::pdqsort_branchless(postings.begin(), postings.end(), [](const Posting &a, const Posting &b) {
        return a.score != b.score ? a.score > b.score : a.id < b.id;
      });
      auto &list = postings_[cell];
      list.ids.reserve(postings.size());
      list.scores.reserve(postings.size());
      for (const auto &posting : postings) {
        list.ids.push_back(posting.id);
        list.scores.push_back(posting.score);
      }
    }
    n_trees = 1;  // MLANN::empty() also represents the built state for flat indexes.
  }

  // MLANN compatibility: IVF1 uses a probability threshold; ordinary IVF uses nprobe.
  void query(const float *data, int k, float parameter, int *out, Distance dist = L2,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const override {
    if (vector_labels_) {
      search(data, k, -1, parameter, out, dist, out_distances, out_n_elected);
    } else {
      if (!std::isfinite(parameter) || parameter < 1 || parameter > centroids_.rows() ||
          std::floor(parameter) != parameter)
        throw std::invalid_argument("nprobe must be an integer in [1, n_lists]");
      search(data, k, int(parameter), 0, out, dist, out_distances, out_n_elected);
    }
  }

  // budget is B for IVF1 (-1 means threshold mode), and nprobe for ordinary IVF.
  void search(const float *data, int k, int budget, float threshold,
              int *out, Distance dist = L2, float *out_distances = nullptr,
              int *out_n_elected = nullptr) const {
    if (empty()) throw std::logic_error("Cannot query an unbuilt index");
    if (k <= 0 || k > n_corpus || (dist != L2 && dist != IP))
      throw std::invalid_argument("Invalid k or distance");
    if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
      throw std::invalid_argument("IVF1 threshold must be in [0, 1]");
    if ((vector_labels_ && budget < -1) ||
        (!vector_labels_ && (budget < 1 || budget > centroids_.rows())))
      throw std::invalid_argument("Invalid candidate budget or nprobe");
    const Eigen::Map<const Eigen::RowVectorXf> q(data, dim);
    if (vector_labels_) {
      int cell = 0;
      float best_distance = squared_euclidean(data, centroids_.row(0).data(), dim);
      for (int c = 1; c < centroids_.rows(); ++c) {
        const float distance = squared_euclidean(data, centroids_.row(c).data(), dim);
        if (distance < best_distance) {
          best_distance = distance;
          cell = c;
        }
      }
      const auto &postings = postings_[cell];
      const uint32_t *ids;
      size_t size;
      if (cell_sizes_[cell] == 0) {
        // No conditional estimate exists: use this cell's ordinary membership.
        const auto &fallback = lists_[cell];
        size = budget < 0 ? fallback.size() : std::min(fallback.size(), size_t(budget));
        ids = fallback.data();
      } else {
        // Scores descend: lower_bound finds the first score <= threshold.
        size = budget < 0 ? size_t(std::lower_bound(
            postings.scores.begin(), postings.scores.end(), threshold,
            [](float score, float cutoff) { return score > cutoff; }) - postings.scores.begin())
            : std::min(postings.ids.size(), size_t(budget));
        ids = postings.ids.data();
      }
      if (out_n_elected) *out_n_elected = size;
      exact_knn(q, k, ids, size, out, dist, out_distances);
      return;
    }
    static thread_local std::vector<float> distances;
    distances.resize(centroids_.rows());
    for (int c = 0; c < centroids_.rows(); ++c)
      distances[c] = squared_euclidean(data, centroids_.row(c).data(), dim);
    static thread_local std::vector<uint32_t> candidates;
    candidates.clear();
    {
      static thread_local std::vector<uint32_t> probes;
      probes.resize(centroids_.rows());
      std::iota(probes.begin(), probes.end(), 0);
      miniselect::pdqpartial_sort_branchless(probes.begin(), probes.begin() + budget, probes.end(),
                        [&](uint32_t a, uint32_t b) {
        return distances[a] != distances[b] ? distances[a] < distances[b] : a < b;
      });
      probes.resize(budget);
      size_t size = 0;
      for (uint32_t id : probes) size += lists_[id].size();
      candidates.reserve(size);
      for (uint32_t id : probes)
        candidates.insert(candidates.end(), lists_[id].begin(), lists_[id].end());
    }
    if (out_n_elected) *out_n_elected = candidates.size();
    exact_knn(q, k, candidates, out, dist, out_distances);
  }

  struct Posting { uint32_t id; float score; };
  struct PostingList {
    std::vector<uint32_t> ids;
    std::vector<float> scores;
  };
  const RowMatrix &centroids() const { return centroids_; }
  const std::vector<PostingList> &postings() const { return postings_; }
  const std::vector<std::vector<uint32_t>> &lists() const { return lists_; }
  const std::vector<int> &cell_sizes() const { return cell_sizes_; }
  bool vector_labels() const { return vector_labels_; }

 private:
  void validate_training(const Eigen::Ref<const UIntRowMatrix> &knn,
                         const Eigen::Ref<const RowMatrix> &train) const {
    if (!empty()) throw std::logic_error("The index has already been built");
    if (train.rows() == 0 || train.cols() != dim || knn.rows() != train.rows() ||
        knn.cols() == 0 || !train.allFinite())
      throw std::invalid_argument("Invalid training query/neighbor shapes or values");
    std::vector<uint32_t> row(knn.cols());
    for (int i = 0; i < knn.rows(); ++i) {
      for (int j = 0; j < knn.cols(); ++j) {
        if (knn(i, j) >= uint32_t(n_corpus))
          throw std::invalid_argument("Training neighbor ID outside the database");
        row[j] = knn(i, j);
      }
      miniselect::pdqsort_branchless(row.begin(), row.end());
      if (std::adjacent_find(row.begin(), row.end()) != row.end())
        throw std::invalid_argument("Training neighbors must be sets (no duplicate IDs)");
    }
  }

  bool vector_labels_;
  RowMatrix centroids_;
  std::vector<std::vector<uint32_t>> lists_;
  std::vector<PostingList> postings_;
  std::vector<int> cell_sizes_;
};

class IVF1 : public IVF {
 public:
  IVF1(const float *data, int n, int dim) : IVF(data, n, dim, true) {}
};
