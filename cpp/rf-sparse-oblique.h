#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlann.h"

/**
 * A supervised random forest whose nodes consider both ordinary axis-aligned
 * splits and sparse oblique splits.  The oblique directions are proposed from
 * random-sign sketches of the training labels, but every proposed threshold is
 * scored with the same full-label entropy objective as RFClass.
 */
class SparseObliqueRF : public MLANN {
 public:
  SparseObliqueRF(const float *corpus_, int n_corpus_, int dim_)
      : MLANN(corpus_, n_corpus_, dim_) {}

  void configure(int sketch_dim_, int oblique_candidates_, int oblique_sparsity_) {
    if (!empty()) {
      throw std::logic_error("The index has already been grown.");
    }
    if (sketch_dim_ < 1 || sketch_dim_ > 64) {
      throw std::out_of_range("The label sketch dimension must be in [1, 64].");
    }
    if (oblique_candidates_ < 1 || oblique_candidates_ > 8) {
      throw std::out_of_range("The number of oblique candidates must be in [1, 8].");
    }
    if (oblique_sparsity_ < 2 || oblique_sparsity_ > kMaxObliqueFeatures) {
      throw std::out_of_range("Oblique split sparsity must be in [2, 4].");
    }
    sketch_dim = sketch_dim_;
    oblique_candidates = oblique_candidates_;
    oblique_sparsity = oblique_sparsity_;
  }

  void grow(int n_trees_, int depth_, const Eigen::Ref<const UIntRowMatrix> &knn_,
            const Eigen::Ref<const RowMatrix> &train_, float density_, int b_, int sketch_dim_,
            int oblique_candidates_, int oblique_sparsity_) {
    configure(sketch_dim_, oblique_candidates_, oblique_sparsity_);
    grow(n_trees_, depth_, knn_, train_, density_, b_);
  }

  void grow(int n_trees_, int depth_, const Eigen::Ref<const UIntRowMatrix> &knn_,
            const Eigen::Ref<const RowMatrix> &train_, float density_ = -1.0F,
            int b_ = 1) override {
    if (!empty()) {
      throw std::logic_error("The index has already been grown.");
    }
    if (n_trees_ <= 0) {
      throw std::out_of_range("The number of trees must be positive.");
    }
    if (train_.rows() == 0 || train_.cols() != dim || knn_.rows() != train_.rows() ||
        knn_.cols() == 0) {
      throw std::invalid_argument(
          "Training vectors and neighbor labels must be non-empty, row-aligned, and match the "
          "corpus dimension.");
    }

    const int n_train = train_.rows();
    if (depth_ <= 0 || depth_ > std::log2(n_train) || depth_ > 29) {
      throw std::out_of_range(
          "The depth must belong to the set {1, ... , min(log2(n_train), 29)}.");
    }
    if (b_ <= 0) {
      throw std::out_of_range("The leaf vote threshold must be positive.");
    }
    if (density_ == 0.0F || density_ > 1.0F) {
      throw std::out_of_range("Density must be in (0, 1] or negative for auto.");
    }

    n_trees = n_trees_;
    depth = depth_;
    n_inner_nodes = (1 << depth_) - 1;
    n_leaves = 1 << depth_;
    b = b_;
    n_pool = n_trees_ * depth_;
    n_array = 1 << (depth_ + 1);
    if (density_ < 0.0F) {
      density = 1.0F / std::sqrt(static_cast<float>(dim));
    } else {
      density = density_;
    }

    const Eigen::Map<const UIntRowMatrix> knn(knn_.data(), knn_.rows(), knn_.cols());
    const Eigen::Map<const RowMatrix> train(train_.data(), train_.rows(), train_.cols());

    nodes_all.assign(n_trees, std::vector<StoredNode>(n_inner_nodes));
    oblique_directions_all = std::vector<std::vector<Direction>>(n_trees);
    labels_all = std::vector<std::vector<std::vector<uint32_t>>>(n_trees);
    votes_all = std::vector<std::vector<std::vector<float>>>(n_trees);

    const SketchMatrix sketches = precompute_label_sketches(knn);
    const auto random_dims_all = generate_random_dimensions();

    const int n = knn.rows();
    log2_tbl.resize(n + 1);
    t_tbl.resize(n + 1);
    log2_tbl[0] = 0.0F;
    for (int i = 1; i <= n; ++i) log2_tbl[i] = std::log2(static_cast<float>(i));
    for (int i = 0; i <= n; ++i) t_tbl[i] = i * log2_tbl[i];
    for (int i = n; i > 0; --i) t_tbl[i] -= t_tbl[i - 1];

#pragma omp parallel
    {
      SplitScratch scratch;
      scratch.ensure_corpus(n_corpus);
      std::vector<int> indices(n_train);

#pragma omp for schedule(dynamic, 1)
      for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
        labels_all[n_tree] = std::vector<std::vector<uint32_t>>(n_leaves);
        votes_all[n_tree] = std::vector<std::vector<float>>(n_leaves);
        std::iota(indices.begin(), indices.end(), 0);
        std::minstd_rand sample_generator(tree_seed(n_tree, 0x9e3779b9U));

        grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, labels_all[n_tree],
                     votes_all[n_tree], train, knn, sketches, random_dims_all[n_tree], scratch,
                     sample_generator);
      }
    }
  }

  void query(const float *data, int k, float vote_threshold, int *out, Distance dist = L2,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const override {
    const Eigen::Map<const Eigen::RowVectorXf> q(data, dim);

    std::vector<int> found_leaves(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      int idx_tree = 0;
      int tree_level = 0;
      for (; tree_level < depth; ++tree_level) {
        const StoredNode &node = nodes_all[n_tree][idx_tree];
        if (node.split == kNoSplit) break;

        float projection;
        if ((node.split & kObliqueMask) == 0) {
          projection = q(node.split);
        } else {
          const Direction &direction =
              oblique_directions_all[n_tree][node.split & kObliqueIndexMask];
          projection = 0.0F;
          for (int j = 0; j < direction.size; ++j) {
            projection += direction.weights[j] * q(direction.features[j]);
          }
        }

        const int idx_left = 2 * idx_tree + 1;
        idx_tree = projection <= node.threshold ? idx_left : idx_left + 1;
      }
      const int levels_to_leaf = depth - tree_level;
      found_leaves[n_tree] =
          (1 << levels_to_leaf) * (idx_tree + 1) - 1 - n_inner_nodes;
    }

    std::vector<uint32_t> elected;
    Eigen::VectorXf votes_total = Eigen::VectorXf::Zero(n_corpus);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      const int leaf_idx = found_leaves[n_tree];
      const std::vector<uint32_t> &labels = labels_all[n_tree][leaf_idx];
      const std::vector<float> &votes = votes_all[n_tree][leaf_idx];
      for (std::size_t i = 0; i < labels.size(); ++i) {
        if ((votes_total(labels[i]) += votes[i]) >= vote_threshold) {
          elected.push_back(labels[i]);
          votes_total(labels[i]) = -9999999.0F;
        }
      }
    }

    if (out_n_elected) *out_n_elected = static_cast<int>(elected.size());
    exact_knn(q, k, elected, out, dist, out_distances);
  }

  std::pair<std::size_t, std::size_t> split_counts() const {
    std::size_t axis = 0;
    std::size_t oblique = 0;
    for (const auto &tree : nodes_all) {
      for (const auto &node : tree) {
        if (node.split == kNoSplit) {
          continue;
        }
        if ((node.split & kObliqueMask) == 0) {
          ++axis;
        } else {
          ++oblique;
        }
      }
    }
    return {axis, oblique};
  }

 private:
  static constexpr int kMaxObliqueFeatures = 4;
  static constexpr int kNodeSubsample = 200;
  static constexpr float kGainTolerance = 0.001F;
  static constexpr std::uint64_t kSketchSeed = 0x6a09e667f3bcc909ULL;
  static constexpr uint32_t kObliqueMask = uint32_t{1} << 31;
  static constexpr uint32_t kObliqueIndexMask = ~kObliqueMask;
  static constexpr uint32_t kNoSplit = std::numeric_limits<uint32_t>::max();

  using SketchMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using SignMatrix =
      Eigen::Matrix<std::int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  struct Direction {
    std::array<uint32_t, kMaxObliqueFeatures> features{};
    std::array<float, kMaxObliqueFeatures> weights{};
    std::uint8_t size = 0;
  };

  struct StoredNode {
    float threshold = 0.0F;
    // A feature id, or the high bit plus an index into the tree's oblique side table.
    uint32_t split = kNoSplit;
  };

  static_assert(sizeof(StoredNode) == 8, "Sparse-oblique routing nodes must stay compact.");

  struct TrainedSplit {
    Direction direction;
    float threshold = 0.0F;
    float gain = 0.0F;
  };

  struct SplitEntry {
    float key;
    int index;
  };

  struct SplitScratch {
    std::vector<int> votes;
    std::vector<int> compact_votes;
    std::vector<uint16_t> compact_votes_16;
    std::vector<int> ids;
    std::vector<uint32_t> local;
    std::vector<SplitEntry> order;
    std::vector<float> left_entropy;
    std::vector<uint32_t> sampled_labels;
    std::vector<uint32_t> touched_ids;
    std::vector<int> coefficient_order;
    Eigen::MatrixXf sampled_x;
    Eigen::MatrixXf sampled_z;
    Eigen::MatrixXf cross_covariance;

    void ensure_corpus(std::size_t n_corpus) {
      if (votes.size() != n_corpus) votes.assign(n_corpus, 0);
    }

    void ensure_samples(int n) {
      if (static_cast<int>(ids.size()) < n) {
        ids.resize(n);
        order.resize(n);
        left_entropy.resize(n);
      }
    }
  };

  static uint32_t tree_seed(int tree, uint32_t salt) {
    uint32_t value = static_cast<uint32_t>(tree) + salt;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
  }

  template <typename Generator>
  static void sample_unique_offsets(int n, int k, Generator &generator,
                                    std::vector<uint32_t> &sample) {
    sample.resize(k);
    std::iota(sample.begin(), sample.end(), 0U);
    for (int i = k; i < n; ++i) {
      std::uniform_int_distribution<int> distribution(0, i);
      const int j = distribution(generator);
      if (j < k) sample[j] = static_cast<uint32_t>(i);
    }
  }

  SketchMatrix precompute_label_sketches(const Eigen::Ref<const UIntRowMatrix> &knn) const {
    SignMatrix label_sketches(n_corpus, sketch_dim);
    std::mt19937_64 generator(kSketchSeed);
    for (int label = 0; label < n_corpus; ++label) {
      const std::uint64_t random_bits = generator();
      for (int j = 0; j < sketch_dim; ++j) {
        label_sketches(label, j) =
            (random_bits & (std::uint64_t{1} << j)) != 0 ? std::int8_t{1} : std::int8_t{-1};
      }
    }

    SketchMatrix sketches = SketchMatrix::Zero(knn.rows(), sketch_dim);
    int invalid_label = 0;
#pragma omp parallel for schedule(static) reduction(| : invalid_label)
    for (int row = 0; row < knn.rows(); ++row) {
      float *output = sketches.row(row).data();
      for (int neighbor = 0; neighbor < knn.cols(); ++neighbor) {
        const uint32_t label = knn(row, neighbor);
        if (label >= static_cast<uint32_t>(n_corpus)) {
          invalid_label = 1;
          continue;
        }
        const std::int8_t *signs = label_sketches.row(label).data();
        for (int j = 0; j < sketch_dim; ++j) output[j] += static_cast<float>(signs[j]);
      }
    }
    if (invalid_label != 0) {
      throw std::out_of_range("A training neighbor label is outside the corpus.");
    }
    return sketches;
  }

  std::vector<std::vector<std::vector<uint32_t>>> generate_random_dimensions() const {
    const int n_random_dimensions =
        std::max(1, std::min(dim, static_cast<int>(density * static_cast<float>(dim))));
    std::vector<std::vector<std::vector<uint32_t>>> dimensions(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      std::minstd_rand generator(tree_seed(n_tree, 0x243f6a88U));
      dimensions[n_tree].reserve(depth);
      for (int tree_level = 0; tree_level < depth; ++tree_level) {
        std::vector<uint32_t> selected;
        sample_unique_offsets(dim, n_random_dimensions, generator, selected);
        dimensions[n_tree].push_back(std::move(selected));
      }
    }
    return dimensions;
  }

  std::vector<Direction> propose_oblique_directions(
      int n, const std::vector<uint32_t> &selected_dimensions,
      const Eigen::Ref<const RowMatrix> &train, const Eigen::Ref<const SketchMatrix> &sketches,
      SplitScratch &scratch) const {
    const int n_dimensions = static_cast<int>(selected_dimensions.size());
    if (n < 2 || n_dimensions < 2) return {};

    scratch.sampled_x.resize(n, n_dimensions);
    scratch.sampled_z.resize(n, sketch_dim);
    for (int row = 0; row < n; ++row) {
      const int source_row = scratch.ids[row];
      for (int column = 0; column < n_dimensions; ++column) {
        scratch.sampled_x(row, column) = train(source_row, selected_dimensions[column]);
      }
      scratch.sampled_z.row(row) = sketches.row(source_row);
    }

    const Eigen::RowVectorXf mean_x = scratch.sampled_x.colwise().mean();
    const Eigen::RowVectorXf mean_z = scratch.sampled_z.colwise().mean();
    scratch.cross_covariance.noalias() = scratch.sampled_x.transpose() * scratch.sampled_z;
    scratch.cross_covariance -=
        static_cast<float>(n) * (mean_x.transpose() * mean_z);

    Eigen::JacobiSVD<Eigen::MatrixXf> solver(scratch.cross_covariance, Eigen::ComputeThinU);
    if (solver.info() != Eigen::Success) return {};

    const Eigen::VectorXf &singular_values = solver.singularValues();
    const float largest = singular_values(0);
    if (!(largest > 0.0F) || !std::isfinite(largest)) return {};

    const int max_candidates = std::min(
        {oblique_candidates, static_cast<int>(singular_values.size()), n - 1});
    const float minimum_singular_value = std::numeric_limits<float>::epsilon() *
                                         std::max(n_dimensions, sketch_dim) * largest;
    std::vector<Direction> directions;
    directions.reserve(max_candidates);
    scratch.coefficient_order.resize(n_dimensions);

    for (int candidate = 0; candidate < max_candidates; ++candidate) {
      if (singular_values(candidate) < minimum_singular_value) break;
      const Eigen::VectorXf vector = solver.matrixU().col(candidate);

      std::iota(scratch.coefficient_order.begin(), scratch.coefficient_order.end(), 0);
      const int target_size = std::min(oblique_sparsity, n_dimensions);
      std::partial_sort(
          scratch.coefficient_order.begin(),
          scratch.coefficient_order.begin() + target_size, scratch.coefficient_order.end(),
          [&vector](int left, int right) {
            return std::abs(vector(left)) > std::abs(vector(right));
          });

      Direction direction;
      float squared_norm = 0.0F;
      const float largest_coefficient = std::abs(vector(scratch.coefficient_order[0]));
      for (int j = 0; j < target_size; ++j) {
        const int coordinate = scratch.coefficient_order[j];
        const float weight = vector(coordinate);
        if (std::abs(weight) <= largest_coefficient * 1.0e-6F) continue;
        direction.features[direction.size] = selected_dimensions[coordinate];
        direction.weights[direction.size] = weight;
        squared_norm += weight * weight;
        ++direction.size;
      }
      if (direction.size < 2 || !(squared_norm > 0.0F)) continue;
      const float inverse_norm = 1.0F / std::sqrt(squared_norm);
      for (int j = 0; j < direction.size; ++j) direction.weights[j] *= inverse_norm;
      directions.push_back(direction);
    }
    return directions;
  }

  template <typename CompactVotes, typename Projection>
  void evaluate_candidate(int n, int k_build, const Direction &direction,
                          const Projection &projection, CompactVotes &compact_votes,
                          SplitScratch &scratch, TrainedSplit &best) const {
    auto &order = scratch.order;
    auto &left_entropy = scratch.left_entropy;
    const auto &sampled_labels = scratch.sampled_labels;

    for (int i = 0; i < n; ++i) order[i].key = projection(order[i].index);
    miniselect::pdqsort_branchless(
        order.begin(), order.begin() + n,
        [](const SplitEntry &left, const SplitEntry &right) { return left.key < right.key; });

    float entropy = 0.0F;
    for (int position = 0; position < n; ++position) {
      const uint32_t *labels =
          sampled_labels.data() + static_cast<std::size_t>(order[position].index) * k_build;
      for (int j = 0; j < k_build; ++j) {
        const int label = static_cast<int>(labels[j]);
        const int votes = ++compact_votes[label];
        entropy += t_tbl[votes];
      }
      left_entropy[position] =
          k_build * log2_tbl[position + 1] - entropy / static_cast<float>(position + 1);
    }

    const float base = left_entropy[n - 1];
    for (int position = 0; position < n - 1; ++position) {
      const uint32_t *labels =
          sampled_labels.data() + static_cast<std::size_t>(order[position].index) * k_build;
      for (int j = 0; j < k_build; ++j) {
        const int label = static_cast<int>(labels[j]);
        const int votes = --compact_votes[label];
        entropy -= t_tbl[votes + 1];
      }

      const int remaining = n - position - 1;
      const float right_entropy =
          k_build * log2_tbl[remaining] - entropy / static_cast<float>(remaining);
      const float left_value = order[position].key;
      const float right_value = order[position + 1].key;
      if (left_value == right_value) continue;

      const float left_weight =
          (position + 1) * (1.0F / n) * left_entropy[position];
      const float right_weight = remaining * (1.0F / n) * right_entropy;
      const float gain = base - (left_weight + right_weight);
      if (gain > best.gain + kGainTolerance) {
        best.gain = gain;
        best.direction = direction;
        best.threshold = 0.5F * (left_value + right_value);
      }
    }

    const uint32_t *last_labels =
        sampled_labels.data() + static_cast<std::size_t>(order[n - 1].index) * k_build;
    for (int j = 0; j < k_build; ++j) --compact_votes[last_labels[j]];
  }

  template <typename Generator>
  TrainedSplit split(const std::vector<int>::iterator &begin,
                     const std::vector<int>::iterator &end,
                     const std::vector<uint32_t> &selected_dimensions,
                     const Eigen::Ref<const RowMatrix> &train,
                     const Eigen::Ref<const UIntRowMatrix> &knn,
                     const Eigen::Ref<const SketchMatrix> &sketches, SplitScratch &scratch,
                     Generator &sample_generator) const {
    const int node_size = static_cast<int>(end - begin);
    TrainedSplit best;
    if (node_size <= 1) return best;

    int n = node_size;
    if (kNodeSubsample > 0 && kNodeSubsample < node_size) {
      n = kNodeSubsample;
      sample_unique_offsets(node_size, n, sample_generator, scratch.local);
    } else {
      scratch.local.resize(n);
      std::iota(scratch.local.begin(), scratch.local.end(), 0U);
    }

    scratch.ensure_corpus(n_corpus);
    scratch.ensure_samples(n);
    for (int i = 0; i < n; ++i) {
      scratch.ids[i] = *(begin + scratch.local[i]);
      scratch.order[i].index = i;
    }

    const int k_build = knn.cols();
    const std::size_t n_sampled_labels =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(k_build);
    scratch.sampled_labels.resize(n_sampled_labels);
    scratch.touched_ids.clear();
    scratch.touched_ids.reserve(std::min(n_sampled_labels, static_cast<std::size_t>(n_corpus)));

    int n_labels = 0;
    for (int i = 0; i < n; ++i) {
      const uint32_t *source = knn.row(scratch.ids[i]).data();
      uint32_t *destination =
          scratch.sampled_labels.data() + static_cast<std::size_t>(i) * k_build;
      for (int j = 0; j < k_build; ++j) {
        const uint32_t label = source[j];
        int &dense_id = scratch.votes[label];
        if (dense_id == 0) {
          dense_id = ++n_labels;
          scratch.touched_ids.push_back(label);
        }
        destination[j] = static_cast<uint32_t>(dense_id - 1);
      }
    }
    for (const uint32_t label : scratch.touched_ids) scratch.votes[label] = 0;

    const auto oblique_directions =
        propose_oblique_directions(n, selected_dimensions, train, sketches, scratch);
    const float *train_data = train.data();
    const int n_columns = train.cols();

    const auto evaluate_all = [&](auto &compact_votes) {
      for (const uint32_t feature : selected_dimensions) {
        Direction axis_direction;
        axis_direction.size = 1;
        axis_direction.features[0] = feature;
        evaluate_candidate(
            n, k_build, axis_direction,
            [&](int sample_index) {
              const std::size_t offset =
                  static_cast<std::size_t>(scratch.ids[sample_index]) * n_columns + feature;
              return train_data[offset];
            },
            compact_votes, scratch, best);
      }

      for (const Direction &direction : oblique_directions) {
        evaluate_candidate(
            n, k_build, direction,
            [&](int sample_index) {
              const std::size_t row_offset =
                  static_cast<std::size_t>(scratch.ids[sample_index]) * n_columns;
              float projection = 0.0F;
              for (int j = 0; j < direction.size; ++j) {
                projection += direction.weights[j] *
                              train_data[row_offset + direction.features[j]];
              }
              return projection;
            },
            compact_votes, scratch, best);
      }
    };

    if (n_sampled_labels <= std::numeric_limits<uint16_t>::max()) {
      scratch.compact_votes_16.resize(n_labels);
      evaluate_all(scratch.compact_votes_16);
    } else {
      scratch.compact_votes.resize(n_labels);
      evaluate_all(scratch.compact_votes);
    }
    return best;
  }

  std::pair<std::vector<uint32_t>, std::vector<float>> count_votes(
      std::vector<int>::iterator leaf_begin, std::vector<int>::iterator leaf_end,
      const Eigen::Ref<const UIntRowMatrix> &knn, SplitScratch &scratch) const {
    const int k_build = knn.cols();
    const std::size_t leaf_size = static_cast<std::size_t>(leaf_end - leaf_begin);
    const std::size_t max_labels = leaf_size * static_cast<std::size_t>(k_build);
    scratch.touched_ids.clear();
    scratch.touched_ids.reserve(std::min(max_labels, static_cast<std::size_t>(n_corpus)));

    for (auto row = leaf_begin; row != leaf_end; ++row) {
      const uint32_t *labels = knn.row(*row).data();
      for (int j = 0; j < k_build; ++j) {
        const uint32_t label = labels[j];
        if (scratch.votes[label]++ == 0) scratch.touched_ids.push_back(label);
      }
    }

    std::vector<uint32_t> labels;
    std::vector<float> votes;
    labels.reserve(scratch.touched_ids.size());
    votes.reserve(scratch.touched_ids.size());
    int n_votes = 0;
    for (const uint32_t label : scratch.touched_ids) {
      const int count = scratch.votes[label];
      scratch.votes[label] = 0;
      if (count >= b) {
        labels.push_back(label);
        votes.push_back(static_cast<float>(count));
        n_votes += count;
      }
    }

    if (!votes.empty()) {
      const float inverse_total =
          1.0F / (static_cast<float>(n_votes) * static_cast<float>(n_trees));
      for (float &vote : votes) vote *= inverse_total;
    }
    return {std::move(labels), std::move(votes)};
  }

  static float project(const float *data, int columns, int row, const Direction &direction) {
    if (direction.size == 1) {
      return data[static_cast<std::size_t>(row) * columns + direction.features[0]];
    }
    const std::size_t row_offset = static_cast<std::size_t>(row) * columns;
    float projection = 0.0F;
    for (int j = 0; j < direction.size; ++j) {
      projection += direction.weights[j] * data[row_offset + direction.features[j]];
    }
    return projection;
  }

  template <typename Generator>
  void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                    int tree_level, int node_index, int n_tree,
                    std::vector<std::vector<uint32_t>> &labels_tree,
                    std::vector<std::vector<float>> &votes_tree,
                    const Eigen::Ref<const RowMatrix> &train,
                    const Eigen::Ref<const UIntRowMatrix> &knn,
                    const Eigen::Ref<const SketchMatrix> &sketches,
                    const std::vector<std::vector<uint32_t>> &random_dimensions,
                    SplitScratch &scratch, Generator &sample_generator) {
    if (tree_level == depth) {
      const int leaf_index = node_index - n_inner_nodes;
      auto result = count_votes(begin, end, knn, scratch);
      labels_tree[leaf_index] = std::move(result.first);
      votes_tree[leaf_index] = std::move(result.second);
      return;
    }

    const TrainedSplit trained =
        split(begin, end, random_dimensions[tree_level], train, knn, sketches, scratch,
              sample_generator);
    if (trained.direction.size == 0) {
      const int levels_to_leaf = depth - tree_level;
      const int leaf_index =
          (1 << levels_to_leaf) * (node_index + 1) - 1 - n_inner_nodes;
      auto result = count_votes(begin, end, knn, scratch);
      labels_tree[leaf_index] = std::move(result.first);
      votes_tree[leaf_index] = std::move(result.second);
      return;
    }

    const float *data = train.data();
    const int columns = train.cols();
    const Direction direction = trained.direction;
    const float threshold = trained.threshold;
    auto middle = std::partition(begin, end, [&](int row) {
      return project(data, columns, row, direction) <= threshold;
    });
    if (middle == begin || middle == end) {
      const int levels_to_leaf = depth - tree_level;
      const int leaf_index =
          (1 << levels_to_leaf) * (node_index + 1) - 1 - n_inner_nodes;
      auto result = count_votes(begin, end, knn, scratch);
      labels_tree[leaf_index] = std::move(result.first);
      votes_tree[leaf_index] = std::move(result.second);
      return;
    }

    StoredNode stored;
    stored.threshold = threshold;
    if (direction.size == 1) {
      stored.split = direction.features[0];
    } else {
      const std::size_t oblique_index = oblique_directions_all[n_tree].size();
      if (oblique_index >= kObliqueIndexMask) {
        throw std::overflow_error("Too many sparse oblique directions in one tree.");
      }
      oblique_directions_all[n_tree].push_back(direction);
      stored.split = kObliqueMask | static_cast<uint32_t>(oblique_index);
    }
    nodes_all[n_tree][node_index] = stored;
    const int left = 2 * node_index + 1;
    const int right = left + 1;
    grow_subtree(begin, middle, tree_level + 1, left, n_tree, labels_tree, votes_tree, train, knn,
                 sketches, random_dimensions, scratch, sample_generator);
    grow_subtree(middle, end, tree_level + 1, right, n_tree, labels_tree, votes_tree, train, knn,
                 sketches, random_dimensions, scratch, sample_generator);
  }

  std::vector<float> log2_tbl;
  std::vector<float> t_tbl;
  std::vector<std::vector<StoredNode>> nodes_all;
  std::vector<std::vector<Direction>> oblique_directions_all;
  int sketch_dim = 16;
  int oblique_candidates = 3;
  int oblique_sparsity = 4;
};
