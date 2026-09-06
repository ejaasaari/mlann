#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "clustering.h"

// IVF uses one k-means partition; IVF1 ensembles random Voronoi partitions.
// Scoring candidates can independently use L2 or IP (normalize for cosine).
class IVF : public MLANN {
 public:
  IVF(const float *data, int n, int dim, bool vector_labels = false)
      : MLANN(data, n, dim), vector_labels_(vector_labels) {}

  void build(int n_lists, const Eigen::Ref<const UIntRowMatrix> &knn,
             const Eigen::Ref<const RowMatrix> &train, int iterations = 25,
             int samples_per_cluster = 256, uint32_t seed = 42, int n_clusterings = 1) {
    validate_training(knn, train);
    if (n_lists <= 0 || n_lists > n_corpus || n_clusterings <= 0 || !corpus.allFinite())
      throw std::invalid_argument("Require finite corpus, 1 <= n_lists <= n_vectors, and positive n_clusterings");
    if (!vector_labels_ && n_clusterings != 1)
      throw std::invalid_argument("Ordinary IVF uses one clustering");
    std::vector<Partition> partitions(n_clusterings);
    for (int t = 0; t < n_clusterings; ++t) {
      RowMatrix centroids;
      if (vector_labels_) {
        // Independent uniform samples without replacement. Never update the
        // sampled corpus vectors: each partition only performs assignment.
        std::seed_seq seeds{seed, uint32_t(t)};
        std::mt19937 generator(seeds);
        std::vector<int> order(n_corpus);
        std::iota(order.begin(), order.end(), 0);
        centroids.resize(n_lists, dim);
        for (int c = 0; c < n_lists; ++c) {
          std::uniform_int_distribution<int> pick(c, n_corpus - 1);
          std::swap(order[c], order[pick(generator)]);
          centroids.row(c) = corpus.row(order[c]);
        }
      } else {
        centroids = mlann_detail::KMeans(n_lists, iterations, samples_per_cluster, seed).train(corpus);
      }
      build_partition(partitions[t], centroids, knn, train);
    }
    partitions_ = std::move(partitions);
    n_trees = n_clusterings;
  }

  // Also permits exactly the same fixed partition to be reused across variants.
  void build_from_centroids(const RowMatrix &centroids,
                            const Eigen::Ref<const UIntRowMatrix> &knn,
                            const Eigen::Ref<const RowMatrix> &train) {
    build_from_centroids(std::vector<RowMatrix>{centroids}, knn, train);
  }

  void build_from_centroids(const std::vector<RowMatrix> &centroids,
                            const Eigen::Ref<const UIntRowMatrix> &knn,
                            const Eigen::Ref<const RowMatrix> &train) {
    validate_training(knn, train);
    if (centroids.empty() || (!vector_labels_ && centroids.size() != 1) ||
        centroids.size() > size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Invalid number of clusterings");
    for (const auto &matrix : centroids)
      if (matrix.rows() <= 0 || matrix.rows() > n_corpus ||
          matrix.cols() != dim || !matrix.allFinite())
        throw std::invalid_argument("Invalid centroids");
    std::vector<Partition> partitions(centroids.size());
    for (size_t t = 0; t < centroids.size(); ++t)
      build_partition(partitions[t], centroids[t], knn, train);
    partitions_ = std::move(partitions);
    n_trees = int(partitions_.size());
  }

  struct Posting { uint32_t id; float score; };
  struct PostingList {
    std::vector<uint32_t> ids;
    std::vector<float> scores;
  };
  struct Partition {
    RowMatrix centroids;
    std::vector<std::vector<uint32_t>> lists;
    std::vector<PostingList> postings;
    std::vector<int> cell_sizes;
  };

 private:
  void build_partition(Partition &partition, const RowMatrix &centroids,
                       const Eigen::Ref<const UIntRowMatrix> &knn,
                       const Eigen::Ref<const RowMatrix> &train) const {
    partition.centroids = centroids;
    const int count = centroids.rows();
    const auto database_cells = mlann_detail::KMeans::assign(corpus, centroids);
    auto &lists = partition.lists;
    lists.resize(count);
    for (int i = 0; i < n_corpus; ++i) lists[database_cells[i]].push_back(i);
    if (!vector_labels_) return;
    const auto query_cells = mlann_detail::KMeans::assign(train, centroids);
    std::vector<std::vector<int>> cell_queries(count);
    for (int i = 0; i < train.rows(); ++i) cell_queries[query_cells[i]].push_back(i);
    partition.postings.resize(count);
    partition.cell_sizes.resize(count);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int cell = 0; cell < count; ++cell) {
      partition.cell_sizes[cell] = cell_queries[cell].size();
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
        postings.push_back({item.first, float(double(item.second) / partition.cell_sizes[cell])});
      miniselect::pdqsort_branchless(postings.begin(), postings.end(), [](const Posting &a, const Posting &b) {
        return a.score != b.score ? a.score > b.score : a.id < b.id;
      });
      auto &list = partition.postings[cell];
      list.ids.reserve(postings.size());
      list.scores.reserve(postings.size());
      for (const auto &posting : postings) {
        list.ids.push_back(posting.id);
        list.scores.push_back(posting.score);
      }
    }
  }

 public:
  // MLANN compatibility: IVF1 uses a probability threshold; ordinary IVF uses nprobe.
  void query(const float *data, int k, float parameter, int *out, Distance dist = L2,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const override {
    if (vector_labels_) {
      search(data, k, -1, parameter, out, dist, out_distances, out_n_elected);
    } else {
      if (empty()) throw std::logic_error("Cannot query an unbuilt index");
      if (!std::isfinite(parameter) || parameter < 1 || parameter > centroids().rows() ||
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
        (!vector_labels_ && (budget < 1 || budget > centroids().rows())))
      throw std::invalid_argument("Invalid candidate budget or nprobe");
    const Eigen::Map<const Eigen::RowVectorXf> q(data, dim);
    if (vector_labels_ && n_trees == 1) {
      const auto &partition = partitions_[0];
      const int cell = nearest_cell(data, partition.centroids);
      const auto &postings = partition.postings[cell];
      const uint32_t *ids;
      size_t size;
      if (partition.cell_sizes[cell] == 0) {
        // No conditional estimate exists: use this cell's ordinary membership.
        const auto &fallback = partition.lists[cell];
        size = budget < 0 ? (threshold < 1 ? fallback.size() : 0)
                          : std::min(fallback.size(), size_t(budget));
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
    if (vector_labels_) {
      search_ensemble(data, q, k, budget, threshold, out, dist, out_distances, out_n_elected);
      return;
    }
    const auto &centroids_ = partitions_[0].centroids;
    const auto &lists_ = partitions_[0].lists;
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

  const RowMatrix &centroids(size_t t = 0) const { return partitions_.at(t).centroids; }
  const std::vector<PostingList> &postings(size_t t = 0) const { return partitions_.at(t).postings; }
  const std::vector<std::vector<uint32_t>> &lists(size_t t = 0) const { return partitions_.at(t).lists; }
  const std::vector<int> &cell_sizes(size_t t = 0) const { return partitions_.at(t).cell_sizes; }
  int n_clusterings() const { return n_trees; }
  bool vector_labels() const { return vector_labels_; }

 private:
  int nearest_cell(const float *data, const RowMatrix &centroids) const {
    static thread_local std::vector<uint32_t> ids;
    static thread_local std::vector<float> distances;
    if (ids.size() != size_t(centroids.rows())) {
      ids.resize(centroids.rows());
      std::iota(ids.begin(), ids.end(), 0);
    }
    distances.resize(centroids.rows());
    mlann_detail::compute_one_to_many(data, centroids.data(), size_t(dim), ids.data(),
        ids.size(), mlann_detail::OneToManyMetric::L2, distances.data());
    return int(std::min_element(distances.begin(), distances.end()) - distances.begin());
  }

  void search_ensemble(const float *data, const Eigen::Map<const Eigen::RowVectorXf> &q,
                       int k, int budget, float threshold, int *out, Distance dist,
                       float *out_distances, int *out_n_elected) const {
    static thread_local std::vector<float> scores;
    static thread_local std::vector<uint32_t> candidates;
    scores.assign(n_corpus, 0.0f);
    candidates.clear();
    for (const auto &partition : partitions_) {
      const int cell = nearest_cell(data, partition.centroids);
      const auto add = [&](uint32_t id, float score) {
        if (scores[id] == 0) candidates.push_back(id);
        scores[id] += score;
      };
      if (partition.cell_sizes[cell] == 0) {
        // An untrained cell contributes a geometric membership vote.
        for (uint32_t id : partition.lists[cell]) add(id, 1.0f);
      } else {
        const auto &postings = partition.postings[cell];
        for (size_t i = 0; i < postings.ids.size(); ++i)
          add(postings.ids[i], postings.scores[i]);
      }
    }
    if (budget < 0) {
      // Average the conditional probabilities so thresholds do not scale with T.
      const float cutoff = threshold * n_trees;
      candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
          [&](uint32_t id) { return scores[id] <= cutoff; }), candidates.end());
    } else {
      const size_t size = std::min(candidates.size(), size_t(budget));
      miniselect::pdqpartial_sort_branchless(candidates.begin(), candidates.begin() + size,
          candidates.end(), [&](uint32_t a, uint32_t b) {
            return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
          });
      candidates.resize(size);
    }
    if (out_n_elected) *out_n_elected = candidates.size();
    exact_knn(q, k, candidates, out, dist, out_distances);
  }

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
  std::vector<Partition> partitions_;
};

class IVF1 : public IVF {
 public:
  IVF1(const float *data, int n, int dim) : IVF(data, n, dim, true) {}
};
