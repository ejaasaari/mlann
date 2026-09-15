#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

#include "detail/huge-buffer.h"
#include "detail/neighbor-query.h"
#include "mlann.h"
#include "utils.h"

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
  std::vector<float> left_ent;
  std::vector<uint32_t> sampled_labels;
  std::vector<uint32_t> touched_ids;

  void ensure_corpus(std::size_t n_corpus) {
    if (votes.size() != n_corpus) votes.assign(n_corpus, 0);
  }
  void ensure_n(int n) {
    if ((int)ids.size() < n) {
      ids.resize(n);
      order.resize(n);
      left_ent.resize(n);
    }
  }
};

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
    if (depth_ <= 0 || depth_ > std::log2(n_train) || depth_ > 29) {
      throw std::out_of_range(
          "The depth must belong to the set {1, ... , min(log2(n_train), 29)}.");
    }

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

    split_points = Eigen::MatrixXf(n_inner_nodes, n_trees);
    split_dimensions = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
        n_inner_nodes, n_trees);
    labels_all = std::vector<std::vector<std::vector<uint32_t>>>(n_trees);
    votes_all = std::vector<std::vector<std::vector<float>>>(n_trees);

    const auto random_dims_all = generate_random_directions();

    const int n = knn.rows();
    log2_tbl = std::vector<float>(n + 1);
    t_tbl = std::vector<float>(n + 1);

    log2_tbl[0] = 0.f;
    for (int i = 1; i <= n; ++i) log2_tbl[i] = std::log2(float(i));
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

        grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, labels_all[n_tree],
                     votes_all[n_tree], train, knn, random_dims_all[n_tree], n_subsample, scratch);
      }
    }
    mlann_detail::promote_existing_corpus_pages(corpus.data(),
                                                size_t(corpus.size()) * sizeof(float));
  }

  void query(const float *data, int k, float vote_threshold, int *out, Distance dist = L2,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const override {
    static thread_local mlann_detail::HugeBuffer<float> votes_total;
    static thread_local std::vector<uint32_t> elected;
    votes_total.resize(n_corpus);
    std::fill_n(votes_total.data(), n_corpus, 0.f);
    elected.clear();

    std::array<int, routing_batch_size> found_leaves;
    for (int first = 0; first < n_trees; first += routing_batch_size) {
      const int count = std::min(routing_batch_size, n_trees - first);
      route_batch(data, first, count, found_leaves.data());
      for (int t = 0; t < count; ++t) {
        const int leaf = found_leaves[t];
        mlann_detail::accumulate_neighbor_votes(labels_all[first + t][leaf],
                                                votes_all[first + t][leaf], votes_total.data(),
                                                vote_threshold, elected);
      }
    }

    if (out_n_elected) *out_n_elected = elected.size();
    exact_knn(Eigen::Map<const Eigen::RowVectorXf>(data, dim), k, elected, out, dist, out_distances,
              mlann_detail::compute_neighbor_scores, mlann_detail::compute_neighbor_topk);
  }

 private:
  static constexpr int routing_batch_size = 64;

  // Advance independent trees together while preserving their leaf-vote order.
  void route_batch(const float *query, int first, int count, int *leaves) const {
    std::array<int, routing_batch_size> nodes{}, active;
    std::iota(active.begin(), active.begin() + count, 0);
    int remaining = count;
    for (int level = 0; level < depth && remaining; ++level) {
      int next = 0;
      for (int i = 0; i < remaining; ++i) {
        const int t = active[i];
        const int node = nodes[t];
        const uint32_t dimension = split_dimensions(node, first + t);
        if (dimension == UINT32_MAX) {
          leaves[t] = (1 << (depth - level)) * (node + 1) - 1 - n_inner_nodes;
          continue;
        }
        nodes[t] = 2 * node + 1 + !(query[dimension] <= split_points(node, first + t));
        if (level + 1 == depth) {
          leaves[t] = nodes[t] - n_inner_nodes;
        } else {
          active[next++] = t;
        }
      }
      remaining = next;
    }
  }

  std::vector<float> log2_tbl;
  std::vector<float> t_tbl;

  std::tuple<int, float, float> split(const std::vector<int>::iterator &begin,
                                      const std::vector<int>::iterator &end,
                                      const std::vector<uint32_t> &random_dims,
                                      const Eigen::Ref<const RowMatrix> &train,
                                      const Eigen::Ref<const UIntRowMatrix> &knn, float tol,
                                      int n_corpus, int n_subsample, SplitScratch &scratch) {
    int n = int(end - begin);
    int max_dim = -1;
    float max_gain = 0.f, max_split = 0.f;
    if (n <= 1) return std::make_tuple(max_dim, max_split, max_gain);

    auto &local = scratch.local;
    if (n_subsample > 0 && n_subsample < n) {
      mlann_detail::sample_unique(n, n_subsample, local);
      n = n_subsample;
    } else {
      local.resize(n);
      std::iota(local.begin(), local.end(), 0);
    }

    scratch.ensure_corpus(n_corpus);
    scratch.ensure_n(n);

    auto &ids = scratch.ids;
    auto &order = scratch.order;
    auto &left_ent = scratch.left_ent;
    auto &votes = scratch.votes;

    for (int i = 0; i < n; ++i) {
      ids[i] = *(begin + local[i]);
      order[i].index = i;
    }

    const float *train_data = train.data();
    const int ncols = train.cols();
    const int k_build = knn.cols();
    const size_t n_sampled_labels = static_cast<size_t>(n) * static_cast<size_t>(k_build);

    scratch.sampled_labels.resize(n_sampled_labels);
    scratch.touched_ids.clear();
    scratch.touched_ids.reserve(std::min(n_sampled_labels, static_cast<size_t>(n_corpus)));

    auto &sampled_labels = scratch.sampled_labels;
    auto &touched_ids = scratch.touched_ids;

    int n_labels = 0;
    for (int i = 0; i < n; ++i) {
      const uint32_t *knn_ptr = knn.row(ids[i]).data();
      uint32_t *sampled_ptr = sampled_labels.data() + static_cast<size_t>(i) * k_build;
      for (int j = 0; j < k_build; ++j) {
        const uint32_t id = knn_ptr[j];
        int &dense_id = votes[id];
        if (dense_id == 0) {
          dense_id = ++n_labels;
          touched_ids.push_back(id);
        }
        sampled_ptr[j] = static_cast<uint32_t>(dense_id - 1);
      }
    }
    for (const uint32_t id : touched_ids) votes[id] = 0;

    const auto evaluate_dimensions = [&](auto &compact_votes) {
      for (uint32_t d : random_dims) {
        for (int i = 0; i < n; ++i) {
          SplitEntry &entry = order[i];
          const size_t offset = static_cast<size_t>(ids[entry.index]) * static_cast<size_t>(ncols) +
                                static_cast<size_t>(d);
          entry.key = train_data[offset];
        }

        miniselect::pdqsort_branchless(
            order.begin(), order.begin() + n,
            [](const SplitEntry &a, const SplitEntry &b) { return a.key < b.key; });

        float entropy = 0.f;
        for (int pos = 0; pos < n; ++pos) {
          const uint32_t *knn_ptr =
              sampled_labels.data() + static_cast<size_t>(order[pos].index) * k_build;
          for (int j = 0; j < k_build; ++j) {
            const int gid = int(knn_ptr[j]);
            const int v = ++compact_votes[gid];
            entropy += t_tbl[v];
          }
          left_ent[pos] = k_build * log2_tbl[pos + 1] - entropy / float(pos + 1);
        }

        const float base = left_ent[n - 1];
        for (int pos = 0; pos < n - 1; ++pos) {
          const uint32_t *knn_ptr =
              sampled_labels.data() + static_cast<size_t>(order[pos].index) * k_build;
          for (int j = 0; j < k_build; ++j) {
            const int gid = int(knn_ptr[j]);
            const int v = --compact_votes[gid];
            entropy -= t_tbl[v + 1];
          }
          const int remain = n - pos - 1;
          const float right_ent = k_build * log2_tbl[remain] - entropy / float(remain);
          const float v1 = order[pos].key;
          const float v2 = order[pos + 1].key;
          if (v1 == v2) continue;

          const float left_w = (pos + 1) * (1.f / n) * left_ent[pos];
          const float right_w = remain * (1.f / n) * right_ent;
          const float gain = base - (left_w + right_w);

          if (gain > max_gain + tol) {
            max_gain = gain;
            max_dim = d;
            max_split = 0.5f * (v1 + v2);
          }
        }

        // Restore the compact vote counters for the next dimension.
        const uint32_t *last_knn_ptr =
            sampled_labels.data() + static_cast<size_t>(order[n - 1].index) * k_build;
        for (int j = 0; j < k_build; ++j) --compact_votes[last_knn_ptr[j]];
      }
    };

    if (n_sampled_labels <= std::numeric_limits<uint16_t>::max()) {
      scratch.compact_votes_16.resize(n_labels);
      evaluate_dimensions(scratch.compact_votes_16);
    } else {
      scratch.compact_votes.resize(n_labels);
      evaluate_dimensions(scratch.compact_votes);
    }

    return std::make_tuple(max_dim, max_split, max_gain);
  }

  std::vector<std::vector<std::vector<uint32_t>>> generate_random_directions() {
    const int n_random_dim = density * dim;
    std::vector<std::vector<std::vector<uint32_t>>> dims_all(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      for (int tree_level = 0; tree_level < depth; ++tree_level) {
        std::vector<uint32_t> dims = mlann_detail::sample_unique(dim, n_random_dim);
        dims_all[n_tree].push_back(dims);
      }
    }
    return dims_all;
  }

  std::pair<std::vector<uint32_t>, std::vector<float>> count_votes(
      std::vector<int>::iterator leaf_begin, std::vector<int>::iterator leaf_end,
      const Eigen::Ref<const UIntRowMatrix> &knn, SplitScratch &scratch) {
    const int k_build = knn.cols();
    const size_t L = static_cast<size_t>(leaf_end - leaf_begin);
    const size_t M = L * static_cast<size_t>(k_build);

    scratch.touched_ids.clear();
    scratch.touched_ids.reserve(std::min(M, static_cast<size_t>(n_corpus)));

    auto &votes = scratch.votes;
    auto &touched_ids = scratch.touched_ids;

    for (auto it = leaf_begin; it != leaf_end; ++it) {
      const int col_idx = *it;
      const uint32_t *knn_ptr = knn.row(col_idx).data();
      for (int j = 0; j < k_build; ++j) {
        const uint32_t id = knn_ptr[j];
        // split() and the previous leaf leave every shared counter at zero.
        if (votes[id]++ == 0) touched_ids.push_back(id);
      }
    }

    std::vector<uint32_t> out_labels;
    std::vector<float> out_votes;
    out_labels.reserve(touched_ids.size());
    out_votes.reserve(touched_ids.size());

    int n_votes = 0;
    for (const uint32_t id : touched_ids) {
      const int cnt = votes[id];
      votes[id] = 0;
      if (cnt >= b) {
        out_labels.push_back(id);
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
                    int tree_level, int i, int n_tree,
                    std::vector<std::vector<uint32_t>> &labels_tree,
                    std::vector<std::vector<float>> &votes_tree,
                    const Eigen::Ref<const RowMatrix> &train,
                    const Eigen::Ref<const UIntRowMatrix> &knn,
                    const std::vector<std::vector<uint32_t>> &random_dims, int n_subsample,
                    SplitScratch &scratch) {
    if (tree_level == depth) {
      const int index_leaf = i - n_inner_nodes;
      auto ret = count_votes(begin, end, knn, scratch);
      labels_tree[index_leaf] = std::move(ret.first);
      votes_tree[index_leaf] = std::move(ret.second);
      return;
    }

    const auto s =
        split(begin, end, random_dims[tree_level], train, knn, tol, n_corpus, n_subsample, scratch);
    const int max_dim = std::get<0>(s);
    const float max_split = std::get<1>(s);

    if (max_dim == -1) {
      split_dimensions(i, n_tree) = UINT32_MAX;
      const int levels2leaf = depth - tree_level;
      const int index_leaf = (1 << levels2leaf) * (i + 1) - 1 - n_inner_nodes;
      auto ret = count_votes(begin, end, knn, scratch);
      labels_tree[index_leaf] = std::move(ret.first);
      votes_tree[index_leaf] = std::move(ret.second);
      return;
    }

    const float *data = train.data();
    const int cols = train.cols();
    auto mid = std::partition(begin, end, [data, cols, max_dim, max_split](const int em) {
      const size_t offset =
          static_cast<size_t>(em) * static_cast<size_t>(cols) + static_cast<size_t>(max_dim);
      return data[offset] <= max_split;
    });

    split_points(i, n_tree) = max_split;
    split_dimensions(i, n_tree) = static_cast<uint32_t>(max_dim);

    const int idx_left = 2 * i + 1;
    const int idx_right = idx_left + 1;
    grow_subtree(begin, mid, tree_level + 1, idx_left, n_tree, labels_tree, votes_tree, train, knn,
                 random_dims, n_subsample, scratch);
    grow_subtree(mid, end, tree_level + 1, idx_right, n_tree, labels_tree, votes_tree, train, knn,
                 random_dims, n_subsample, scratch);
  }

  int n_subsample = 200;
  float tol = 0.001;
};
