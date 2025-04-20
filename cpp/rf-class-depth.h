#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "mlann.h"

class RFClass : public MLANN {
 public:
  RFClass(const float *corpus_, int n_corpus_, int dim_) : MLANN(corpus_, n_corpus_, dim_) {}

  void grow(int n_trees_, int depth_, const Eigen::Ref<const UIntRowMatrix> &knn_,
            const Eigen::Ref<const RowMatrix> &train_, float density_ = -1.0, int b_ = 1) {
    if (!empty()) {
      throw std::logic_error("The index has already been grown.");
    }

    if (n_trees_ <= 0) {
      throw std::out_of_range("The number of trees must be positive.");
    }

    int n_train = train_.rows();

    n_trees = n_trees_;
    depth = depth_;
    n_inner_nodes = (1 << depth_) - 1;
    n_leaves = 1 << depth_;
    b = b_;
    n_pool = n_trees_ * depth_;
    n_array = 1 << (depth_ + 1);

    if (density_ < 0) {
      density = 1.0 / std::sqrt(dim);
    } else {
      density = density_;
    }

    const Eigen::Map<const UIntRowMatrix> knn(knn_.data(), knn_.rows(), knn_.cols());
    const Eigen::Map<const RowMatrix> train(train_.data(), train_.rows(), train_.cols());

    split_points = Eigen::MatrixXf(n_array, n_trees);
    split_dimensions = Eigen::MatrixXi(n_array, n_trees);
    labels_all = std::vector<std::vector<std::vector<int>>>(n_trees);
    votes_all = std::vector<std::vector<std::vector<float>>>(n_trees);

    std::random_device gen;
    std::mt19937 r = std::mt19937(gen());

    const auto random_dims_all = generate_random_directions(r);

#pragma omp parallel for
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      labels_all[n_tree] = std::vector<std::vector<int>>(n_leaves);
      votes_all[n_tree] = std::vector<std::vector<float>>(n_leaves);

      std::random_device gen;
      std::mt19937 rand_gen(gen());
      std::vector<int> indices(n_train);
      std::iota(indices.begin(), indices.end(), 0);
      grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, labels_all[n_tree],
                   votes_all[n_tree], train, knn, random_dims_all[n_tree], rand_gen, n_subsample);
    }
  }

  void query(const float *data, int k, float vote_threshold, int *out, Distance dist = L2,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const {
    const Eigen::Map<const Eigen::RowVectorXf> q(data, dim);

    std::vector<int> found_leaves(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      int idx_tree = 0;
      int d = 0;
      for (; d < depth; ++d) {
        const int idx_left = 2 * idx_tree + 1;
        const int idx_right = idx_left + 1;
        const float split_point = split_points(idx_tree, n_tree);
        const int split_dimension = split_dimensions(idx_tree, n_tree);
        if (split_dimension == -1) {  // if branch stops before maximum depth
          break;
        }
        if (q(split_dimension) <= split_point) {
          idx_tree = idx_left;
        } else {
          idx_tree = idx_right;
        }
      }
      const int levels2leaf = depth - d;
      found_leaves[n_tree] = (1 << levels2leaf) * (idx_tree + 1) - 1 - n_inner_nodes;
    }

    std::vector<int> elected;
    Eigen::VectorXf votes_total = Eigen::VectorXf::Zero(n_corpus);

    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      int leaf_idx = found_leaves[n_tree];
      const std::vector<int> &labels = labels_all[n_tree][leaf_idx];
      const std::vector<float> &votes = votes_all[n_tree][leaf_idx];
      int n_labels = labels.size();
      for (int i = 0; i < n_labels; ++i) {
        if ((votes_total(labels[i]) += votes[i]) >= vote_threshold) {
          elected.push_back(labels[i]);
          votes_total(labels[i]) = -9999999;
        }
      }
    }

    if (out_n_elected) *out_n_elected = elected.size();

    exact_knn(q, k, elected, out, dist, out_distances);
  }

 private:
  static std::tuple<int, float, float> split(const std::vector<int>::iterator &begin,
                                             const std::vector<int>::iterator &end,
                                             const std::vector<int> &random_dims,
                                             const Eigen::Ref<const RowMatrix> &train,
                                             const Eigen::Ref<const UIntRowMatrix> &knn, float tol,
                                             int n_corpus, std::mt19937 &r, int n_subsample) {
    int n = end - begin;
    int max_dim = -1;
    float max_gain = 0, max_split = 0;

    if (n <= 1) return std::make_tuple(max_dim, max_split, max_gain);

    std::vector<int> indices;
    if (n_subsample > 0 && n_subsample < n) {
      std::vector<int> indices_org(n);
      n = n_subsample;
      std::iota(indices_org.begin(), indices_org.end(), 0);
      std::shuffle(indices_org.begin(), indices_org.end(), r);
      indices = std::vector<int>(indices_org.begin(), indices_org.begin() + n_subsample);
    } else {
      indices = std::vector<int>(n);
      std::iota(indices.begin(), indices.end(), 0);
    }

    Eigen::VectorXf left_entropies(n), right_entropies(n);
    const float *data = train.data();
    const int cols = train.cols();
    const int k_build = knn.cols();

    for (const auto &d : random_dims) {
      std::sort(indices.begin(), indices.end(), [data, cols, d, begin](int i1, int i2) {
        return data[*(begin + i1) * cols + d] < data[*(begin + i2) * cols + d];
      });

      std::vector<int> votes(n_corpus, 0);

      float entropy = 0;
      for (int ii = 0; ii < n; ++ii) {
        const int i = indices[ii];
        const Eigen::Matrix<uint32_t, 1, Eigen::Dynamic> knn_crnt = knn.row(*(begin + i));
        for (int j = 0; j < k_build; ++j) {
          int v = ++votes[knn_crnt(j)];
          if (v > 1) entropy -= (v - 1) * log2(v - 1);
          entropy += v * log2(v);
        }
        left_entropies[ii] = k_build * log2(ii + 1) - entropy / (ii + 1);
      }

      for (int ii = 0; ii < n - 1; ++ii) {
        const int i = indices[ii];
        const Eigen::Matrix<uint32_t, 1, Eigen::Dynamic> knn_crnt = knn.row(*(begin + i));
        for (int j = 0; j < k_build; ++j) {
          int v = --votes[knn_crnt(j)];
          entropy -= (v + 1) * log2(v + 1);
          if (v) entropy += v * log2(v);
        }
        right_entropies[ii] = k_build * log2(n - ii - 1) - entropy / (n - ii - 1);
      }
      right_entropies[n - 1] = 0;

      for (int ii = 0; ii < n - 1; ++ii) {
        const int i = indices[ii];
        if (train(*(begin + i), d) == train(*(begin + indices[ii + 1]), d)) continue;
        float left = static_cast<float>(ii + 1) / n * left_entropies[ii];
        float right = static_cast<float>(n - ii - 1) / n * right_entropies[ii];
        float gain = left_entropies[n - 1] - (left + right);
        if (gain > max_gain + tol) {
          max_gain = gain;
          max_dim = d;
          max_split = (train(*(begin + i), d) + train(*(begin + indices[ii + 1]), d)) / 2.0;
        }
      }
    }
    return std::make_tuple(max_dim, max_split, max_gain);
  }

  std::vector<std::vector<std::vector<int>>> generate_random_directions(std::mt19937 &r) {
    const int n_random_dim = density * dim;
    std::vector<std::vector<std::vector<int>>> dims_all(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      for (int tree_level = 0; tree_level < depth; ++tree_level) {
        std::vector<int> dims(dim);
        std::iota(dims.begin(), dims.end(), 0);
        std::shuffle(dims.begin(), dims.end(), r);
        dims_all[n_tree].push_back(std::vector<int>(dims.begin(), dims.begin() + n_random_dim));
      }
    }
    return dims_all;
  }

  std::pair<std::vector<int>, std::vector<float>> count_votes(
      std::vector<int>::iterator leaf_begin, std::vector<int>::iterator leaf_end,
      const Eigen::Ref<const UIntRowMatrix> &knn) {
    int k_build = knn.cols();
    std::unordered_map<int, int> votes;
    for (auto it = leaf_begin; it != leaf_end; ++it) {
      const Eigen::Matrix<uint32_t, 1, Eigen::Dynamic> knn_crnt = knn.row(*it);
      for (int j = 0; j < k_build; ++j) ++votes[knn_crnt(j)];
    }

    std::vector<int> out_labels;
    std::vector<float> out_votes;

    int n_votes = 0;
    for (const auto &v : votes)
      if (v.second >= b) {
        out_labels.push_back(v.first);
        out_votes.push_back(v.second);
        n_votes += v.second;
      }

    for (int i = 0; i < out_votes.size(); ++i) out_votes[i] /= (n_votes * n_trees);

    return {out_labels, out_votes};
  }

  void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                    int tree_level, int i, int n_tree, std::vector<std::vector<int>> &labels_tree,
                    std::vector<std::vector<float>> &votes_tree,
                    const Eigen::Ref<const RowMatrix> &train,
                    const Eigen::Ref<const UIntRowMatrix> &knn,
                    const std::vector<std::vector<int>> &random_dims, std::mt19937 &r,
                    int n_subsample) {
    if (tree_level == depth) {
      const int index_leaf = i - n_inner_nodes;
      const auto ret = count_votes(begin, end, knn);
      labels_tree[index_leaf] = ret.first;
      votes_tree[index_leaf] = ret.second;
      return;
    }

    const auto s =
        split(begin, end, random_dims[tree_level], train, knn, tol, n_corpus, r, n_subsample);
    const int max_dim = std::get<0>(s);
    const float max_split = std::get<1>(s);

    if (max_dim == -1) {
      split_dimensions(i, n_tree) = -1;
      const int levels2leaf = depth - tree_level;
      const int index_leaf = (1 << levels2leaf) * (i + 1) - 1 - n_inner_nodes;
      const auto ret = count_votes(begin, end, knn);
      labels_tree[index_leaf] = ret.first;
      votes_tree[index_leaf] = ret.second;
      return;
    }

    const float *data = train.data();
    const int cols = train.cols();
    auto mid = std::partition(begin, end, [data, cols, max_dim, max_split](const int em) {
      return data[em * cols + max_dim] <= max_split;
    });

    split_points(i, n_tree) = max_split;
    split_dimensions(i, n_tree) = max_dim;

    const int idx_left = 2 * i + 1;
    const int idx_right = idx_left + 1;
    grow_subtree(begin, mid, tree_level + 1, idx_left, n_tree, labels_tree, votes_tree, train, knn,
                 random_dims, r, n_subsample);
    grow_subtree(mid, end, tree_level + 1, idx_right, n_tree, labels_tree, votes_tree, train, knn,
                 random_dims, r, n_subsample);
  }

  int n_subsample = 100;
  float tol = 0.001;
};
