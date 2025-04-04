#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlann.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> IntRowMatrix;

class RFRP : public MLANN {
 public:
  RFRP(const float *corpus_, int n_corpus_, int dim_) : MLANN(corpus_, n_corpus_, dim_) {}

  void grow(int n_trees_, int depth_, const Eigen::Ref<const IntRowMatrix> &knn_,
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

    if (density_ < -1.0001 || density_ > 1.0001 || (density_ > -0.9999 && density_ < -0.0001)) {
      throw std::out_of_range("The density must be on the interval (0,1].");
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

    const Eigen::Map<const IntRowMatrix> knn(knn_.data(), knn_.rows(), knn_.cols());
    const Eigen::Map<const RowMatrix> train(train_.data(), train_.rows(), train_.cols());

    density < 1 ? build_sparse_random_matrix(sparse_random_matrix, n_pool, dim, density)
                : build_dense_random_matrix(dense_random_matrix, n_pool, dim);

    split_points = Eigen::MatrixXf(n_array, n_trees);
    labels_all = std::vector<std::vector<std::vector<int>>>(n_trees);
    votes_all = std::vector<std::vector<std::vector<float>>>(n_trees);

#pragma omp parallel for
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      labels_all[n_tree] = std::vector<std::vector<int>>(n_leaves);
      votes_all[n_tree] = std::vector<std::vector<float>>(n_leaves);
      Eigen::MatrixXf tree_projections;

      if (density < 1)
        tree_projections.noalias() = sparse_random_matrix.middleRows(n_tree * depth, depth) * train;
      else
        tree_projections.noalias() = dense_random_matrix.middleRows(n_tree * depth, depth) * train;

      std::vector<int> indices(n_train);
      std::iota(indices.begin(), indices.end(), 0);

      grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, tree_projections,
                   labels_all[n_tree], votes_all[n_tree], knn);
    }
  }

  void query(const float *data, int k, float vote_threshold, int *out,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const {
    if (k <= 0 || k > n_corpus) {
      throw std::out_of_range("k must belong to the set {1, ..., n_corpus}.");
    }

    if (vote_threshold <= 0) {
      throw std::out_of_range("vote_threshold must be positive");
    }

    if (empty()) {
      throw std::logic_error("The index must be built before making queries.");
    }

    const Eigen::Map<const Eigen::VectorXf> q(data, dim);

    Eigen::VectorXf projected_query(n_pool);
    if (density < 1)
      projected_query.noalias() = sparse_random_matrix * q;
    else
      projected_query.noalias() = dense_random_matrix * q;

    std::vector<int> found_leaves(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      int idx_tree = 0;
      for (int d = 0; d < depth; ++d) {
        const int j = n_tree * depth + d;
        const int idx_left = 2 * idx_tree + 1;
        const int idx_right = idx_left + 1;
        const float split_point = split_points(idx_tree, n_tree);
        if (projected_query(j) <= split_point) {
          idx_tree = idx_left;
        } else {
          idx_tree = idx_right;
        }
      }
      found_leaves[n_tree] = idx_tree - n_inner_nodes;
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
          votes_total(labels[i]) = -100000.;
        }
      }
    }

    if (out_n_elected) *out_n_elected = elected.size();

    exact_knn(q, k, elected, out, out_distances);
  }

 private:
  std::pair<std::vector<int>, std::vector<float>> count_votes(std::vector<int>::iterator leaf_begin,
                                                              std::vector<int>::iterator leaf_end,
                                                              const IntRowMatrix &knn) {
    int k_build = knn.cols();
    std::unordered_map<int, float> votes;
    for (auto it = leaf_begin; it != leaf_end; ++it) {
      const Eigen::VectorXi knn_crnt = knn.row(*it);
      for (int j = 0; j < k_build; ++j) votes[knn_crnt(j)] += 1.0;
    }

    std::vector<int> out_labels;
    std::vector<float> out_votes;

    for (const auto &v : votes)
      if (v.second >= b) {
        out_labels.push_back(v.first);
        out_votes.push_back(v.second);
      }

    return {out_labels, out_votes};
  }

  /**
   * Builds a single random projection tree. The tree is constructed by recursively
   * projecting the data on a random vector and splitting into two by the median.
   */
  void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                    int tree_level, int i, int n_tree, const Eigen::MatrixXf &tree_projections,
                    std::vector<std::vector<int>> &labels_tree,
                    std::vector<std::vector<float>> &votes_tree,
                    const Eigen::Map<const IntRowMatrix> &knn) {
    int n = end - begin;
    int idx_left = 2 * i + 1;
    int idx_right = idx_left + 1;

    if (tree_level == depth) {
      int index_leaf = i - n_inner_nodes;
      auto ret = count_votes(begin, end, knn);
      labels_tree[index_leaf] = ret.first;
      votes_tree[index_leaf] = ret.second;
      return;
    }

    miniselect::pdqselect_branchless(
        begin, begin + n / 2, end, [&tree_projections, tree_level](int i1, int i2) {
          return tree_projections(tree_level, i1) < tree_projections(tree_level, i2);
        });
    auto mid = end - n / 2;

    if (n % 2) {
      split_points(i, n_tree) = tree_projections(tree_level, *(mid - 1));
    } else {
      auto left_it = std::max_element(begin, mid, [&tree_projections, tree_level](int i1, int i2) {
        return tree_projections(tree_level, i1) < tree_projections(tree_level, i2);
      });
      split_points(i, n_tree) =
          (tree_projections(tree_level, *mid) + tree_projections(tree_level, *left_it)) / 2.0;
    }

    grow_subtree(begin, mid, tree_level + 1, idx_left, n_tree, tree_projections, labels_tree,
                 votes_tree, knn);
    grow_subtree(mid, end, tree_level + 1, idx_right, n_tree, tree_projections, labels_tree,
                 votes_tree, knn);
  }

  /**
   * Builds a random sparse matrix for use in random projection. The components of
   * the matrix are drawn from the distribution
   *
   *       0 w.p. 1 - a
   * N(0, 1) w.p. a
   *
   * where a = density.
   */
  static void build_sparse_random_matrix(
      Eigen::SparseMatrix<float, Eigen::RowMajor> &sparse_random_matrix, int n_row, int n_col,
      float density) {
    sparse_random_matrix = Eigen::SparseMatrix<float, Eigen::RowMajor>(n_row, n_col);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uni_dist(0, 1);
    std::normal_distribution<float> norm_dist(0, 1);

    std::vector<Eigen::Triplet<float>> triplets;
    for (int j = 0; j < n_row; ++j) {
      for (int i = 0; i < n_col; ++i) {
        if (uni_dist(gen) > density) continue;
        triplets.push_back(Eigen::Triplet<float>(j, i, norm_dist(gen)));
      }
    }

    sparse_random_matrix.setFromTriplets(triplets.begin(), triplets.end());
    sparse_random_matrix.makeCompressed();
  }

  /*
   * Builds a random dense matrix for use in random projection. The components of
   * the matrix are drawn from the standard normal distribution.
   */
  static void build_dense_random_matrix(
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &dense_random_matrix,
      int n_row, int n_col) {
    dense_random_matrix =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(n_row, n_col);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dist(0, 1);

    std::generate(dense_random_matrix.data(), dense_random_matrix.data() + n_row * n_col,
                  [&normal_dist, &gen] { return normal_dist(gen); });
  }

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      dense_random_matrix;  // random vectors needed for all the RP-trees
  Eigen::SparseMatrix<float, Eigen::RowMajor>
      sparse_random_matrix;  // random vectors needed for all the RP-trees
};