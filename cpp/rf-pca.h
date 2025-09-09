#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlann.h"

class RFPCA : public MLANN {
 public:
  RFPCA(const float *corpus_, int n_corpus_, int dim_) : MLANN(corpus_, n_corpus_, dim_) {}

  void grow(int n_trees_, int depth_, const Eigen::Ref<const UIntRowMatrix> &knn_,
            const Eigen::Ref<const RowMatrix> &train_, float density_ = -1.0, int b_ = 1) {
    if (!empty()) {
      throw std::logic_error("The index has already been grown.");
    }

    if (n_trees_ <= 0) {
      throw std::out_of_range("The number of trees must be positive.");
    }

    int n_train = train_.rows();
    if (depth_ <= 0 || depth_ > std::log2(n_train)) {
      throw std::out_of_range("The depth must belong to the set {1, ... , log2(n_train)}.");
    }

    n_trees = n_trees_;
    depth = depth_;
    n_inner_nodes = (1 << depth_) - 1;
    n_leaves = 1 << depth_;
    b = b_;
    n_pool = n_inner_nodes * n_trees;
    n_array = 1 << (depth_ + 1);

    if (density_ < 0) {
      density = 1.0 / std::sqrt(dim);
    } else {
      density = density_;
    }

    const Eigen::Map<const UIntRowMatrix> knn(knn_.data(), knn_.rows(), knn_.cols());
    const Eigen::Map<const RowMatrix> train(train_.data(), train_.rows(), train_.cols());

    split_points = Eigen::MatrixXf(n_array, n_trees);
    labels_all = std::vector<std::vector<std::vector<uint32_t>>>(n_trees);
    votes_all = std::vector<std::vector<std::vector<float>>>(n_trees);

    _random_dims = std::vector<Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>>();
    _random_vectors = std::vector<Eigen::VectorXf>();
    _random_dims.reserve(n_pool);
    _random_vectors.reserve(n_pool);

    int tgt = static_cast<int>(density * dim);
    for (int i = 0; i < n_pool; ++i) {
      _random_dims.emplace_back(tgt);
      _random_vectors.emplace_back(tgt);
    }

#pragma omp parallel for
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      labels_all[n_tree] = std::vector<std::vector<uint32_t>>(n_leaves);
      votes_all[n_tree] = std::vector<std::vector<float>>(n_leaves);
      std::vector<int> indices(n_train);
      std::iota(indices.begin(), indices.end(), 0);

      std::random_device rd;
      std::minstd_rand gen(rd());
      std::uniform_int_distribution<int> uni_dist(0, dim - 1);
      std::normal_distribution<float> norm_dist(0, 1);

      for (int i = n_inner_nodes * n_tree; i < n_inner_nodes * (n_tree + 1); ++i) {
        std::generate(_random_dims[i].data(), _random_dims[i].data() + tgt,
                      [&uni_dist, &gen] { return uni_dist(gen); });
        std::generate(_random_vectors[i].data(), _random_vectors[i].data() + tgt,
                      [&norm_dist, &gen] { return norm_dist(gen); });
      }

      grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, labels_all[n_tree],
                   votes_all[n_tree], knn, train);
    }
  }

  void query(const float *data, int k, float vote_threshold, int *out, Distance dist = L2,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const {
    const Eigen::Map<const Eigen::RowVectorXf> q(data, dim);

    std::vector<int> found_leaves(n_trees);
    const int tgt = static_cast<int>(density * dim);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      int idx_tree = 0;
      const int jj = n_tree * ((1 << depth) - 1);
      for (int d = 0; d < depth; ++d) {
        float proj = 0;
        const uint32_t *x = _random_dims[jj + idx_tree].data();
        const float *y = _random_vectors[jj + idx_tree].data();
        for (int i = tgt; i--; ++x, ++y) proj += q[*x] * *y;
        if (proj <= split_points(idx_tree, n_tree)) {
          idx_tree = 2 * idx_tree + 1;
        } else {
          idx_tree = 2 * idx_tree + 2;
        }
      }
      found_leaves[n_tree] = idx_tree - n_inner_nodes;
    }

    std::vector<uint32_t> elected;
    Eigen::VectorXf votes_total = Eigen::VectorXf::Zero(n_corpus);

    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      int leaf_idx = found_leaves[n_tree];
      const std::vector<uint32_t> &labels = labels_all[n_tree][leaf_idx];
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
  std::pair<std::vector<uint32_t>, std::vector<float>> count_votes(
      std::vector<int>::iterator leaf_begin, std::vector<int>::iterator leaf_end,
      const Eigen::Ref<const UIntRowMatrix> &knn) {
    const int k_build = knn.cols();
    const size_t L = static_cast<size_t>(leaf_end - leaf_begin);
    const size_t M = L * static_cast<size_t>(k_build);

    std::unordered_map<uint32_t, int> votes;
    votes.reserve(M);

    for (auto it = leaf_begin; it != leaf_end; ++it) {
      const int col_idx = *it;
      auto col = knn.row(col_idx);
      for (int j = 0; j < k_build; ++j) {
        const uint32_t id = col(j);
        auto [p, inserted] = votes.try_emplace(id, 0);
        ++p->second;
      }
    }

    std::vector<uint32_t> out_labels;
    std::vector<float> out_votes;
    out_labels.reserve(votes.size());
    out_votes.reserve(votes.size());

    for (const auto &kv : votes) {
      const int cnt = kv.second;
      if (cnt >= b) {
        out_labels.push_back(kv.first);
        out_votes.push_back(static_cast<float>(cnt));
      }
    }

    return {std::move(out_labels), std::move(out_votes)};
  }

  void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                    int tree_level, int i, int n_tree,
                    std::vector<std::vector<uint32_t>> &labels_tree,
                    std::vector<std::vector<float>> &votes_tree, const UIntRowMatrix &knn,
                    const RowMatrix &train) {
    int n = end - begin;
    int idx_left = 2 * i + 1;
    int idx_right = idx_left + 1;
    auto mid = end - n / 2;

    if (tree_level == depth) {
      int index_leaf = i - n_inner_nodes;
      auto ret = count_votes(begin, end, knn);
      labels_tree[index_leaf] = ret.first;
      votes_tree[index_leaf] = ret.second;
      return;
    }

    {
      Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> dims =
          _random_dims[n_tree * ((1 << depth) - 1) + i];
      Eigen::VectorXf rv = _random_vectors[n_tree * ((1 << depth) - 1) + i];
      rv /= rv.norm();

      Eigen::MatrixXf tmp = train(Eigen::Map<Eigen::VectorXi>(&*begin, n), dims).transpose();
      float isz = 1. / (n - 1);
      Eigen::MatrixXf centered = tmp.colwise() - tmp.rowwise().mean();
      Eigen::MatrixXf cov = 2 * 0.01 * isz * (centered * centered.transpose());

      rv /= rv.norm();
      for (int i = 0; i < 20; ++i) {
        Eigen::VectorXf last = rv;
        rv += cov * rv;
        rv /= rv.norm();
        if ((rv - last).cwiseAbs().mean() < 0.01) break;
      }

      Eigen::VectorXf data = rv.transpose() * tmp;
      _random_vectors[n_tree * ((1 << depth) - 1) + i] = rv;

      std::unordered_map<int, int> inv_idx;
      for (int i = 0; i < n; ++i) {
        inv_idx[*(begin + i)] = i;
      }

      miniselect::pdqselect_branchless(begin, begin + n / 2, end,
                                       [&data, &inv_idx](const int i1, const int i2) {
                                         return data[inv_idx[i1]] < data[inv_idx[i2]];
                                       });

      if (n % 2) {
        split_points(i, n_tree) = data[inv_idx[*(mid - 1)]];
      } else {
        auto left_it = std::max_element(begin, mid, [&data, &inv_idx](const int i1, const int i2) {
          return data[inv_idx[i1]] < data[inv_idx[i2]];
        });
        split_points(i, n_tree) = (data[inv_idx[*mid]] + data[inv_idx[*left_it]]) / 2.0;
      }
    }

    grow_subtree(begin, mid, tree_level + 1, idx_left, n_tree, labels_tree, votes_tree, knn, train);
    grow_subtree(mid, end, tree_level + 1, idx_right, n_tree, labels_tree, votes_tree, knn, train);
  }

  std::vector<Eigen::Matrix<uint32_t, Eigen::Dynamic, 1>> _random_dims;
  std::vector<Eigen::VectorXf> _random_vectors;
};
