#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "mlann.h"

std::vector<int> sample_unique(int n, int k) {
  std::random_device rd;
  std::minstd_rand generator(rd());

  std::vector<int> reservoir(k);
  std::iota(reservoir.begin(), reservoir.end(), 0);

  for (int i = k; i < n; ++i) {
    std::uniform_int_distribution<int> distribution(0, i);
    int j = distribution(generator);

    if (j < k) {
      reservoir[j] = i;
    }
  }

  return reservoir;
}

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
    std::minstd_rand r = std::minstd_rand(gen());

    const auto random_dims_all = generate_random_directions(r);

    const int n = knn.rows();
    log2_tbl = std::vector<float>(n + 1);
    t_tbl = std::vector<float>(n + 1);

    log2_tbl[0] = 0.f;
    for (int i = 1; i <= n; ++i) log2_tbl[i] = std::log2(float(i));
    for (int i = 0; i <= n; ++i) t_tbl[i] = i * log2_tbl[i];

#pragma omp parallel for
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      labels_all[n_tree] = std::vector<std::vector<int>>(n_leaves);
      votes_all[n_tree] = std::vector<std::vector<float>>(n_leaves);

      std::random_device gen;
      std::minstd_rand rand_gen(gen());
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
  std::vector<float> log2_tbl;
  std::vector<float> t_tbl;

  std::tuple<int, float, float> split(const std::vector<int>::iterator &begin,
                                      const std::vector<int>::iterator &end,
                                      const std::vector<int> &random_dims,
                                      const Eigen::Ref<const RowMatrix> &train,
                                      const Eigen::Ref<const UIntRowMatrix> &knn, float tol,
                                      int n_corpus, std::minstd_rand &r, int n_subsample) {
    int n = int(end - begin);
    int max_dim = -1;
    float max_gain = 0.f, max_split = 0.f;
    if (n <= 1) return std::make_tuple(max_dim, max_split, max_gain);

    std::vector<int> local;
    if (n_subsample > 0 && n_subsample < n) {
      local = sample_unique(n, n_subsample);
      n = n_subsample;
    } else {
      local.resize(n);
      std::iota(local.begin(), local.end(), 0);
    }

    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i) ids[i] = *(begin + local[i]);

    std::vector<float> left_ent(n), right_ent(n), keys(n);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);

    std::vector<int> stamp(n_corpus, -1);
    std::vector<int> votes(n_corpus);
    int epoch = 0;

    const float *train_data = train.data();
    const int ncols = train.cols();
    const int k_build = knn.cols();

    for (int d : random_dims) {
      for (int i = 0; i < n; ++i) keys[i] = train_data[ids[i] * ncols + d];

      miniselect::pdqsort_branchless(order.begin(), order.end(),
                                     [&](int a, int b) { return keys[a] < keys[b]; });

      ++epoch;

      float entropy = 0.f;
      for (int pos = 0; pos < n; ++pos) {
        const int row = ids[order[pos]];
        const uint32_t *knn_ptr = knn.row(row).data();
        for (int j = 0; j < k_build; ++j) {
          const int gid = int(knn_ptr[j]);
          if (stamp[gid] != epoch) {
            stamp[gid] = epoch;
            votes[gid] = 0;
          }
          const int v = ++votes[gid];
          // entropy += v*log2(v) - (v-1)*log2(v-1)
          entropy += t_tbl[v] - t_tbl[v - 1];
        }
        left_ent[pos] = k_build * log2_tbl[pos + 1] - entropy / float(pos + 1);
      }

      for (int pos = 0; pos < n - 1; ++pos) {
        const int row = ids[order[pos]];
        const uint32_t *knn_ptr = knn.row(row).data();
        for (int j = 0; j < k_build; ++j) {
          const int gid = int(knn_ptr[j]);
          const int v = --votes[gid];
          // entropy += v*log2(v) - (v+1)*log2(v+1)
          entropy += t_tbl[v] - t_tbl[v + 1];
        }
        const int remain = n - pos - 1;
        right_ent[pos] = k_build * log2_tbl[remain] - entropy / float(remain);
      }
      right_ent[n - 1] = 0.f;

      const float base = left_ent[n - 1];
      for (int pos = 0; pos < n - 1; ++pos) {
        const float v1 = keys[order[pos]];
        const float v2 = keys[order[pos + 1]];
        if (v1 == v2) continue;

        const float left_w = (pos + 1) * (1.f / n) * left_ent[pos];
        const float right_w = (n - pos - 1) * (1.f / n) * right_ent[pos];
        const float gain = base - (left_w + right_w);

        if (gain > max_gain + tol) {
          max_gain = gain;
          max_dim = d;
          max_split = 0.5f * (v1 + v2);
        }
      }
    }

    return std::make_tuple(max_dim, max_split, max_gain);
  }

  std::vector<std::vector<std::vector<int>>> generate_random_directions(std::minstd_rand &r) {
    const int n_random_dim = density * dim;
    std::vector<std::vector<std::vector<int>>> dims_all(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      for (int tree_level = 0; tree_level < depth; ++tree_level) {
        std::vector<int> dims = sample_unique(dim, n_random_dim);
        dims_all[n_tree].push_back(dims);
      }
    }
    return dims_all;
  }

  std::pair<std::vector<int>, std::vector<float>> count_votes(
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

    std::vector<int> out_labels;
    std::vector<float> out_votes;
    out_labels.reserve(votes.size());
    out_votes.reserve(votes.size());

    int n_votes = 0;
    for (const auto &kv : votes) {
      const int cnt = kv.second;
      if (cnt >= b) {
        out_labels.push_back(static_cast<int>(kv.first));
        out_votes.push_back(static_cast<float>(cnt));
        n_votes += cnt;
      }
    }

    if (!out_votes.empty()) {
      const float inv = 1.0f / (static_cast<float>(n_votes) * static_cast<float>(n_trees));
      for (float &v : out_votes) v *= inv;
    }

    return {std::move(out_labels), std::move(out_votes)};
  }

  void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                    int tree_level, int i, int n_tree, std::vector<std::vector<int>> &labels_tree,
                    std::vector<std::vector<float>> &votes_tree,
                    const Eigen::Ref<const RowMatrix> &train,
                    const Eigen::Ref<const UIntRowMatrix> &knn,
                    const std::vector<std::vector<int>> &random_dims, std::minstd_rand &r,
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
