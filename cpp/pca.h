#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "detail/huge-buffer.h"
#include "detail/neighbor-query.h"
#include "mlann.h"

// Median-split PCA forest. Sparse PCA samples coordinates with replacement;
// PCAFull fits on at most 300 rows and uses every coordinate.
class RFPCA : public MLANN {
 public:
  RFPCA(const float *corpus_, int n_corpus_, int dim_) : RFPCA(corpus_, n_corpus_, dim_, false) {}

  void grow(int n_trees_, int depth_, const Eigen::Ref<const UIntRowMatrix> &knn,
            const Eigen::Ref<const RowMatrix> &train, float density_ = -1.0, int b_ = 1) override {
    grow_impl(n_trees_, depth_, knn, train, density_, b_, false);
  }

  // Partition the corpus itself; leaves hold corpus IDs with implicit unit votes.
  void grow_unsupervised(int n_trees_, int depth_, float density_ = -1.0) override {
    grow_impl(n_trees_, depth_, UIntRowMatrix(), corpus, density_, 1, true);
  }

 private:
  void grow_impl(int n_trees_, int depth_, const Eigen::Ref<const UIntRowMatrix> &knn,
                 const Eigen::Ref<const RowMatrix> &train, float density_, int b_,
                 bool unsupervised) {
    if (!empty()) {
      throw std::logic_error("The index has already been grown.");
    }
    if (n_trees_ <= 0) {
      throw std::out_of_range("The number of trees must be positive.");
    }
    const int n_train = train.rows();
    if (depth_ <= 0 || depth_ > std::log2(n_train) || depth_ > 29) {
      throw std::out_of_range(
          "The depth must belong to the set {1, ... , min(log2(n_train), 29)}.");
    }
    if (dim <= 0 || n_corpus <= 0 || train.cols() != dim || b_ < 1 ||
        (unsupervised && !train.allFinite()) ||
        (!unsupervised &&
         (knn.rows() != n_train || knn.cols() < 1 || knn.maxCoeff() >= uint32_t(n_corpus)))) {
      throw std::invalid_argument("Invalid forest data or dimensions.");
    }
    const float requested_density =
        full_dimensions ? 1.f : (density_ < 0 ? float(1.0 / std::sqrt(dim)) : density_);
    if (!std::isfinite(requested_density) || requested_density < 0.f || requested_density > 1.f) {
      throw std::invalid_argument("Density must belong to [0, 1].");
    }

    corpus_leaves = unsupervised;
    n_trees = n_trees_;
    depth = depth_;
    n_inner_nodes = (1 << depth) - 1;
    n_leaves = 1 << depth;
    n_array = 1 << (depth + 1);
    b = b_;
    density = requested_density;
    support = static_cast<int>(density * dim);
    const Eigen::Index node_count = Eigen::Index(n_inner_nodes) * n_trees;

    split_points.resize(n_inner_nodes, n_trees);
    projections.resize(node_count, support);
    // Full-support nodes share implicit coordinates 0..dim-1.
    if (!full_dimensions) projection_dims.resize(node_count, support);
    labels_all.resize(n_trees);
    if (!corpus_leaves) votes_all.resize(n_trees);

#pragma omp parallel
    {
      TreeScratch scratch(corpus_leaves ? 0 : n_corpus, n_train);
#pragma omp for schedule(dynamic, 1)
      for (int tree = 0; tree < n_trees; ++tree) {
        labels_all[tree].resize(n_leaves);
        if (!corpus_leaves) votes_all[tree].resize(n_leaves);
        std::iota(scratch.rows.begin(), scratch.rows.end(), 0);

        std::random_device rd;
        std::minstd_rand generator(rd());
        initialize_projections(tree, generator);
        grow_subtree(scratch.rows.begin(), scratch.rows.end(), 0, 0, tree, knn, train, generator,
                     scratch);
      }
    }
    mlann_detail::promote_existing_corpus_pages(corpus.data(),
                                                size_t(corpus.size()) * sizeof(float));
  }

 public:
  void query(const float *data, int k, float vote_threshold, int *out, Distance dist = L2,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const override {
    static thread_local mlann_detail::HugeBuffer<float> votes_total;
    static thread_local std::vector<uint32_t> elected;
    votes_total.resize(n_corpus);
    std::fill_n(votes_total.data(), n_corpus, 0.f);
    elected.clear();

    std::array<int, routing_batch_size> leaves;
    for (int first = 0; first < n_trees; first += routing_batch_size) {
      const int count = std::min(routing_batch_size, n_trees - first);
      route_batch(data, first, count, leaves.data());
      for (int t = 0; t < count; ++t) {
        const int leaf = leaves[t];
        if (corpus_leaves) {
          mlann_detail::accumulate_unit_votes(labels_all[first + t][leaf], votes_total.data(),
                                              vote_threshold, elected);
        } else {
          mlann_detail::accumulate_neighbor_votes(labels_all[first + t][leaf],
                                                  votes_all[first + t][leaf], votes_total.data(),
                                                  vote_threshold, elected);
        }
      }
    }
    if (out_n_elected) *out_n_elected = elected.size();
    exact_knn(Eigen::Map<const Eigen::RowVectorXf>(data, dim), k, elected, out, dist, out_distances,
              mlann_detail::compute_neighbor_scores, mlann_detail::compute_neighbor_topk);
  }

 protected:
  RFPCA(const float *corpus_, int n_corpus_, int dim_, bool full_dimensions_)
      : MLANN(corpus_, n_corpus_, dim_), full_dimensions(full_dimensions_) {}

 private:
  using IndexIterator = std::vector<int>::iterator;
  static constexpr int routing_batch_size = 64;
  bool corpus_leaves = false;
  static constexpr int fit_sample = 300;
  const bool full_dimensions;
  int support = 0;
  RowMatrix projections;
  UIntRowMatrix projection_dims;

  struct TreeScratch {
    std::vector<int> rows;
    std::vector<int> votes;
    std::vector<uint32_t> touched_ids;
    std::vector<int> sampled_rows;
    std::vector<float> row_scores;
    Eigen::MatrixXf points, fit, centered, covariance;
    Eigen::VectorXf direction, last, projected, scores;

    TreeScratch(int corpus_size, int train_size)
        : rows(train_size), votes(corpus_size, 0), row_scores(train_size) {
      sampled_rows.reserve(fit_sample);
      scores.resize(train_size);
    }
  };

  Eigen::Index projection_row(int tree, int node) const {
    return Eigen::Index(tree) * n_inner_nodes + node;
  }

  void initialize_projections(int tree, std::minstd_rand &generator) {
    std::uniform_int_distribution<int> coordinate(0, dim - 1);
    std::normal_distribution<float> normal(0, 1);
    // Preserve per-tree random draw order, including initialization of every node.
    for (int node = 0; node < n_inner_nodes; ++node) {
      const auto row = projection_row(tree, node);
      if (!full_dimensions) {
        for (int j = 0; j < support; ++j) projection_dims(row, j) = coordinate(generator);
      }
      for (int j = 0; j < support; ++j) projections(row, j) = normal(generator);
    }
  }

  void gather_points(IndexIterator begin, int count, Eigen::Index row,
                     const Eigen::Ref<const RowMatrix> &train, Eigen::MatrixXf &output) const {
    if (output.rows() != support || output.cols() < count) output.resize(support, count);
    for (int i = 0; i < count; ++i) {
      const float *point = train.row(begin[i]).data();
      float *column = output.col(i).data();
      if (full_dimensions) {
        std::copy_n(point, support, column);
      } else {
        for (int j = 0; j < support; ++j) column[j] = point[projection_dims(row, j)];
      }
    }
  }

  void fit_projection(IndexIterator begin, IndexIterator end, Eigen::Index row,
                      const Eigen::Ref<const RowMatrix> &train, std::minstd_rand &generator,
                      TreeScratch &scratch) {
    const int count = end - begin;
    if (support == 0) {
      for (auto it = begin; it != end; ++it) scratch.row_scores[*it] = 0.f;
      return;
    }
    auto &direction = scratch.direction;
    direction = projections.row(row).transpose();
    direction /= direction.norm();
    gather_points(begin, count, row, train, scratch.points);

    const float *fit_data = scratch.points.data();
    int fit_count = count;
    if (full_dimensions && count > fit_sample) {
      scratch.sampled_rows.clear();
      std::sample(begin, end, std::back_inserter(scratch.sampled_rows), fit_sample, generator);
      gather_points(scratch.sampled_rows.begin(), fit_sample, row, train, scratch.fit);
      fit_data = scratch.fit.data();
      fit_count = fit_sample;
    }
    const Eigen::Map<const Eigen::MatrixXf> fit(fit_data, support, fit_count);
    const float scale = 1. / (fit_count - 1);
    if (scratch.centered.rows() != support || scratch.centered.cols() < fit_count) {
      scratch.centered.resize(support, fit_count);
    }
    Eigen::Map<Eigen::MatrixXf> centered(scratch.centered.data(), support, fit_count);
    centered = fit.colwise() - fit.rowwise().mean();
    if (!full_dimensions) {
      scratch.covariance = 2 * 0.01 * scale * (centered * centered.transpose());
    }
    direction /= direction.norm();
    for (int iteration = 0; iteration < 20; ++iteration) {
      scratch.last = direction;
      if (full_dimensions) {
        if (scratch.projected.size() < fit_count) scratch.projected.resize(fit_count);
        auto projected = scratch.projected.head(fit_count);
        projected = centered.transpose() * direction;
        direction += (0.02f * scale) * (centered * projected);
      } else {
        direction += scratch.covariance * direction;
      }
      direction /= direction.norm();
      if ((direction - scratch.last).cwiseAbs().mean() < 0.01) break;
    }
    scratch.scores.head(count) = direction.transpose() * scratch.points.leftCols(count);
    projections.row(row) = direction.transpose();
    for (int i = 0; i < count; ++i) scratch.row_scores[begin[i]] = scratch.scores[i];
  }

  void make_leaf(IndexIterator begin, IndexIterator end, int tree, int leaf,
                 const Eigen::Ref<const UIntRowMatrix> &knn, TreeScratch &scratch) {
    if (corpus_leaves) {
      labels_all[tree][leaf].assign(begin, end);
      return;
    }
    scratch.touched_ids.clear();
    scratch.touched_ids.reserve(std::min<size_t>(n_corpus, size_t(end - begin) * knn.cols()));
    for (auto it = begin; it != end; ++it) {
      const uint32_t *labels = knn.row(*it).data();
      for (int j = 0; j < knn.cols(); ++j) {
        const auto label = labels[j];
        if (scratch.votes[label]++ == 0) scratch.touched_ids.push_back(label);
      }
    }
    auto &labels = labels_all[tree][leaf];
    auto &votes = votes_all[tree][leaf];
    labels.reserve(scratch.touched_ids.size());
    votes.reserve(scratch.touched_ids.size());
    for (const auto label : scratch.touched_ids) {
      const int count = scratch.votes[label];
      scratch.votes[label] = 0;
      if (count >= b) {
        labels.push_back(label);
        // PCA uses raw counts, unlike RF's normalized leaf votes.
        votes.push_back(static_cast<float>(count));
      }
    }
  }

  void grow_subtree(IndexIterator begin, IndexIterator end, int level, int node, int tree,
                    const Eigen::Ref<const UIntRowMatrix> &knn,
                    const Eigen::Ref<const RowMatrix> &train, std::minstd_rand &generator,
                    TreeScratch &scratch) {
    if (level == depth) {
      make_leaf(begin, end, tree, node - n_inner_nodes, knn, scratch);
      return;
    }
    const int count = end - begin;
    const auto mid = end - count / 2;
    fit_projection(begin, end, projection_row(tree, node), train, generator, scratch);
    const auto less = [&](int left, int right) {
      return scratch.row_scores[left] < scratch.row_scores[right];
    };
    miniselect::pdqselect_branchless(begin, begin + count / 2, end, less);
    if (count % 2) {
      split_points(node, tree) = scratch.row_scores[*(mid - 1)];
    } else {
      const auto left = std::max_element(begin, mid, less);
      split_points(node, tree) = (scratch.row_scores[*mid] + scratch.row_scores[*left]) / 2.0;
    }
    grow_subtree(begin, mid, level + 1, 2 * node + 1, tree, knn, train, generator, scratch);
    grow_subtree(mid, end, level + 1, 2 * node + 2, tree, knn, train, generator, scratch);
  }

  void route_batch(const float *query, int first, int count, int *leaves) const {
    std::array<int, routing_batch_size> nodes{};
    for (int level = 0; level < depth; ++level) {
      for (int t = 0; t < count; ++t) {
        const auto row = projection_row(first + t, nodes[t]);
        float score = 0.f;
        // Keep the sparse projection's accumulation order, including repeated coordinates.
        if (full_dimensions) {
          for (int j = 0; j < support; ++j) score += query[j] * projections(row, j);
        } else {
          for (int j = 0; j < support; ++j)
            score += query[projection_dims(row, j)] * projections(row, j);
        }
        nodes[t] = 2 * nodes[t] + (score <= split_points(nodes[t], first + t) ? 1 : 2);
      }
    }
    for (int t = 0; t < count; ++t) leaves[t] = nodes[t] - n_inner_nodes;
  }
};

class PCAFull : public RFPCA {
 public:
  PCAFull(const float *corpus_, int n_corpus_, int dim_) : RFPCA(corpus_, n_corpus_, dim_, true) {}
};
