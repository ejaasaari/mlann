#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "../detail/huge-buffer.h"
#include "../detail/neighbor-query.h"
#include "../mlann.h"
#include "../utils.h"

class RF : public MLANN {
  public:
    RF(const float* corpus_, int n_corpus_, int dim_) : MLANN(corpus_, n_corpus_, dim_) {}

    void grow(
        int n_trees_,
        int depth_,
        const Eigen::Ref<const UIntRowMatrix>& knn_,
        const Eigen::Ref<const RowMatrix>& train_,
        float density_ = -1.0,
        int b_ = 1
    ) override {
        if (!empty()) {
            throw std::logic_error("The index has already been grown.");
        }

        if (n_trees_ <= 0) {
            throw std::out_of_range("The number of trees must be positive.");
        }

        const int n_train = train_.rows();
        if (depth_ <= 0 || depth_ > std::log2(n_train) || depth_ > 29) {
            throw std::out_of_range(
                "The depth must belong to the set {1, ... , min(log2(n_train), 29)}."
            );
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

        split_points = Eigen::MatrixXf::Zero(n_inner_nodes, n_trees);
        split_dimensions = UIntRowMatrix::Zero(n_inner_nodes, n_trees);
        labels_all = std::vector<std::vector<std::vector<uint32_t>>>(n_trees);
        votes_all = std::vector<std::vector<std::vector<float>>>(n_trees);
        prepare_tuning(knn, false, n_train);

        const auto random_dims_all = generate_random_directions();

        const int n = knn.rows();
        log2_tbl = std::vector<float>(n + 1);
        t_tbl = std::vector<float>(n + 1);

        log2_tbl[0] = 0.f;
        for (int i = 1; i <= n; ++i)
            log2_tbl[i] = std::log2(float(i));
        for (int i = 0; i <= n; ++i)
            t_tbl[i] = i * log2_tbl[i];
        for (int i = n; i > 0; --i)
            t_tbl[i] -= t_tbl[i - 1];

#pragma omp parallel
        {
            TreeScratch scratch;
            scratch.ensure_corpus(n_corpus);
            std::vector<int> indices(n_train);

#pragma omp for schedule(dynamic, 1) nowait
            for (int tree = 0; tree < n_trees; ++tree) {
                labels_all[tree] = std::vector<std::vector<uint32_t>>(n_leaves);
                votes_all[tree] = std::vector<std::vector<float>>(n_leaves);

                std::iota(indices.begin(), indices.end(), 0);

                grow_subtree(
                    indices.begin(),
                    indices.end(),
                    0,
                    0,
                    tree,
                    labels_all[tree],
                    votes_all[tree],
                    train,
                    knn,
                    random_dims_all[tree],
                    scratch
                );
                finish_tuning_tree(tree, indices);
            }
        }
        mlann_detail::promote_existing_corpus_pages(
            corpus.data(), size_t(corpus.size()) * sizeof(float)
        );
    }

    std::unique_ptr<MLANN> make_view(int trees, int d) const override {
        auto view = std::make_unique<RF>(corpus.data(), n_corpus, dim);
        initialize_view(*view, trees, d);
        view->unscaled_votes = true;
        return view;
    }

    size_t index_bytes() const override {
        return MLANN::index_bytes() + sizeof(RF) - sizeof(MLANN);
    }

    void query(
        const float* data,
        int k,
        float vote_threshold,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr,
        int* out_n_elected = nullptr
    ) const override {
        static thread_local mlann_detail::HugeBuffer<float> votes_total;
        static thread_local std::vector<uint32_t> elected;
        votes_total.resize(n_corpus);
        std::fill_n(votes_total.data(), n_corpus, 0.f);
        elected.clear();
        const auto& query_labels = labels_all;
        const auto& query_votes = votes_all;

        std::array<int, routing_batch_size> leaves;
        for (int first = 0; first < n_trees; first += routing_batch_size) {
            const int count = std::min(routing_batch_size, n_trees - first);
            route_batch(data, first, count, leaves.data());
            for (int t = 0; t < count; ++t) {
                const int leaf = leaves[t];
                if (accumulate_tuning_votes(
                        first + t, leaf, votes_total.data(), vote_threshold, elected
                    ))
                    continue;
                if (unscaled_votes) {
                    // Same float sums and final divisor as calibration. Threshold
                    // crossings track election without destroying cumulative scores.
                    const auto& labels = query_labels[first + t][leaf];
                    const auto& weights = query_votes[first + t][leaf];
                    for (size_t j = 0; j < labels.size(); ++j) {
                        float& total = votes_total[labels[j]];
                        const float previous = total;
                        total += weights[j];
                        if (total / float(n_trees) >= vote_threshold &&
                            (previous / float(n_trees) < vote_threshold || previous == 0))
                            elected.push_back(labels[j]);
                    }
                    continue;
                }
                mlann_detail::accumulate_neighbor_votes(
                    labels_all[first + t][leaf],
                    votes_all[first + t][leaf],
                    votes_total.data(),
                    vote_threshold,
                    elected
                );
            }
        }

        if (out_n_elected)
            *out_n_elected = elected.size();
        exact_knn(
            Eigen::Map<const Eigen::RowVectorXf>(data, dim),
            k,
            elected,
            out,
            dist,
            out_distances,
            mlann_detail::compute_neighbor_scores,
            mlann_detail::compute_neighbor_topk
        );
    }

  protected:
    void route_for_timing(const float* query, int* leaves) const override {
        for (int first = 0; first < n_trees; first += routing_batch_size)
            route_batch(
                query, first, std::min(routing_batch_size, n_trees - first), leaves + first
            );
    }

    bool probability_scores() const override { return true; }
    void tuning_path(const float* q, int tree, int* path) const override {
        path[0] = 0;
        bool terminal = false;
        for (int level = 0; level < depth; ++level) {
            const int node = path[level];
            terminal = terminal || split_dimensions(node, tree) == UINT32_MAX;
            path[level + 1] =
                terminal
                    ? node
                    : 2 * node +
                          (q[split_dimensions(node, tree)] <= split_points(node, tree) ? 1 : 2);
        }
    }

  private:
    bool unscaled_votes = false;
    using IndexIterator = std::vector<int>::iterator;
    static constexpr int routing_batch_size = 64;
    std::vector<float> log2_tbl;
    std::vector<float> t_tbl;
    static constexpr int split_scoring_row_cap = 400;
    float tol = 0.001;

    struct SplitEntry {
        float key;
        int index;
    };

    struct TreeScratch {
        std::vector<int> votes;
        std::vector<int> compact_votes;
        std::vector<uint16_t> compact_votes_16;
        std::vector<int> ids;
        std::vector<uint32_t> local;
        std::vector<SplitEntry> order;
        std::vector<float> left_ent;
        std::vector<uint32_t> sampled_labels;
        std::vector<size_t> sampled_offsets;
        std::vector<size_t> label_counts;
        std::vector<uint32_t> touched_ids;

        void ensure_corpus(std::size_t n_corpus) {
            if (votes.size() != n_corpus)
                votes.assign(n_corpus, 0);
        }

        void ensure_n(int n) {
            if ((int) ids.size() < n) {
                ids.resize(n);
                order.resize(n);
                left_ent.resize(n);
            }
        }
    };

    std::tuple<int, float, float> split(
        const IndexIterator& begin,
        const IndexIterator& end,
        const std::vector<uint32_t>& random_dims,
        const Eigen::Ref<const RowMatrix>& train,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        float tol,
        int n_corpus,
        TreeScratch& scratch
    ) {
        int n = int(end - begin);
        int max_dim = -1;
        float max_gain = 0.f, max_split = 0.f;
        if (n <= 1)
            return std::make_tuple(max_dim, max_split, max_gain);

        auto& local = scratch.local;
        if (n > split_scoring_row_cap) {
            mlann_detail::sample_unique(n, split_scoring_row_cap, local);
            n = split_scoring_row_cap;
        } else {
            local.resize(n);
            std::iota(local.begin(), local.end(), 0);
        }

        scratch.ensure_corpus(n_corpus);
        scratch.ensure_n(n);

        auto& ids = scratch.ids;
        auto& order = scratch.order;
        auto& left_ent = scratch.left_ent;
        auto& votes = scratch.votes;

        for (int i = 0; i < n; ++i) {
            ids[i] = *(begin + local[i]);
            order[i].index = i;
        }

        const float* train_data = train.data();
        const int ncols = train.cols();
        const int k_build = knn.cols();
        const size_t n_sampled_labels = static_cast<size_t>(n) * static_cast<size_t>(k_build);

        scratch.sampled_labels.resize(n_sampled_labels);
        scratch.touched_ids.clear();
        scratch.touched_ids.reserve(std::min(n_sampled_labels, static_cast<size_t>(n_corpus)));

        auto& sampled_labels = scratch.sampled_labels;
        auto& touched_ids = scratch.touched_ids;

        auto& label_counts = scratch.label_counts;
        label_counts.clear();
        label_counts.reserve(std::min(n_sampled_labels, static_cast<size_t>(n_corpus)));
        int n_labels = 0;
        for (int i = 0; i < n; ++i) {
            const uint32_t* knn_ptr = knn.row(ids[i]).data();
            uint32_t* sampled_ptr = sampled_labels.data() + static_cast<size_t>(i) * k_build;
            for (int j = 0; j < k_build; ++j) {
                const uint32_t id = knn_ptr[j];
                int& dense_id = votes[id];
                if (dense_id == 0) {
                    dense_id = ++n_labels;
                    touched_ids.push_back(id);
                    label_counts.push_back(0);
                }
                ++label_counts[dense_id - 1];
                sampled_ptr[j] = static_cast<uint32_t>(dense_id - 1);
            }
        }
        for (const uint32_t id : touched_ids)
            votes[id] = 0;

        // Singleton labels contribute t_tbl[1] == 0 for every candidate split.
        // Keep the repeated labels in their original order and retain k_build
        // for entropy normalization, including the omitted singleton labels.
        n_labels = 0;
        for (auto& count : label_counts)
            count = count > 1 ? static_cast<size_t>(n_labels++) : SIZE_MAX;
        scratch.sampled_offsets.resize(n + 1);
        size_t kept = 0;
        for (int row = 0; row < n; ++row) {
            scratch.sampled_offsets[row] = kept;
            for (int j = 0; j < k_build; ++j) {
                const auto mapped =
                    label_counts[sampled_labels[static_cast<size_t>(row) * k_build + j]];
                if (mapped != SIZE_MAX)
                    sampled_labels[kept++] = static_cast<uint32_t>(mapped);
            }
        }
        scratch.sampled_offsets[n] = kept;

        const auto evaluate_dimensions = [&](auto& compact_votes) {
            for (uint32_t d : random_dims) {
                for (int i = 0; i < n; ++i) {
                    SplitEntry& entry = order[i];
                    const size_t offset =
                        static_cast<size_t>(ids[entry.index]) * static_cast<size_t>(ncols) +
                        static_cast<size_t>(d);
                    entry.key = train_data[offset];
                }

                miniselect::pdqsort_branchless(
                    order.begin(), order.begin() + n, [](const SplitEntry& a, const SplitEntry& b) {
                        return a.key < b.key;
                    }
                );

                float entropy = 0.f;
                for (int pos = 0; pos < n; ++pos) {
                    const uint32_t* knn_ptr =
                        sampled_labels.data() + scratch.sampled_offsets[order[pos].index];
                    const size_t count = scratch.sampled_offsets[order[pos].index + 1] -
                                         scratch.sampled_offsets[order[pos].index];
                    for (size_t j = 0; j < count; ++j) {
                        const int gid = int(knn_ptr[j]);
                        const int v = ++compact_votes[gid];
                        entropy += t_tbl[v];
                    }
                    left_ent[pos] = k_build * log2_tbl[pos + 1] - entropy / float(pos + 1);
                }

                const float base = left_ent[n - 1];
                for (int pos = 0; pos < n - 1; ++pos) {
                    const uint32_t* knn_ptr =
                        sampled_labels.data() + scratch.sampled_offsets[order[pos].index];
                    const size_t count = scratch.sampled_offsets[order[pos].index + 1] -
                                         scratch.sampled_offsets[order[pos].index];
                    for (size_t j = 0; j < count; ++j) {
                        const int gid = int(knn_ptr[j]);
                        const int v = --compact_votes[gid];
                        entropy -= t_tbl[v + 1];
                    }
                    const int remain = n - pos - 1;
                    const float right_ent = k_build * log2_tbl[remain] - entropy / float(remain);
                    const float v1 = order[pos].key;
                    const float v2 = order[pos + 1].key;
                    if (v1 == v2)
                        continue;

                    const float left_w = (pos + 1) * (1.f / n) * left_ent[pos];
                    const float right_w = remain * (1.f / n) * right_ent;
                    const float gain = base - (left_w + right_w);

                    if (gain > max_gain + tol) {
                        max_gain = gain;
                        max_dim = d;
                        max_split = 0.5f * (v1 + v2);
                    }
                }

                const uint32_t* last_knn_ptr =
                    sampled_labels.data() + scratch.sampled_offsets[order[n - 1].index];
                const size_t last_count = scratch.sampled_offsets[order[n - 1].index + 1] -
                                          scratch.sampled_offsets[order[n - 1].index];
                for (size_t j = 0; j < last_count; ++j)
                    --compact_votes[last_knn_ptr[j]];
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
        for (int tree = 0; tree < n_trees; ++tree) {
            for (int tree_level = 0; tree_level < depth; ++tree_level) {
                std::vector<uint32_t> dims = mlann_detail::sample_unique(dim, n_random_dim);
                dims_all[tree].push_back(dims);
            }
        }
        return dims_all;
    }

    std::pair<std::vector<uint32_t>, std::vector<float>> count_votes(
        IndexIterator leaf_begin,
        IndexIterator leaf_end,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        TreeScratch& scratch
    ) {
        const int k_build = knn.cols();
        const size_t L = static_cast<size_t>(leaf_end - leaf_begin);
        const size_t M = L * static_cast<size_t>(k_build);

        scratch.touched_ids.clear();
        scratch.touched_ids.reserve(std::min(M, static_cast<size_t>(n_corpus)));

        auto& votes = scratch.votes;
        auto& touched_ids = scratch.touched_ids;

        for (auto it = leaf_begin; it != leaf_end; ++it) {
            const int col_idx = *it;
            const uint32_t* knn_ptr = knn.row(col_idx).data();
            for (int j = 0; j < k_build; ++j) {
                const uint32_t id = knn_ptr[j];
                // split() and the previous leaf leave every shared counter at zero.
                if (votes[id]++ == 0)
                    touched_ids.push_back(id);
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
            for (float& v : out_votes)
                v *= inv;
        }

        return {std::move(out_labels), std::move(out_votes)};
    }

    void grow_subtree(
        IndexIterator begin,
        IndexIterator end,
        int tree_level,
        int i,
        int tree,
        std::vector<std::vector<uint32_t>>& labels_tree,
        std::vector<std::vector<float>>& votes_tree,
        const Eigen::Ref<const RowMatrix>& train,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const std::vector<std::vector<uint32_t>>& random_dims,
        TreeScratch& scratch
    ) {
        if (tree_level == depth) {
            record_tuning_node(tree, i, begin, end);
            if (tuning_structure_only)
                return;
            const int index_leaf = i - n_inner_nodes;
            auto ret = count_votes(begin, end, knn, scratch);
            labels_tree[index_leaf] = std::move(ret.first);
            votes_tree[index_leaf] = std::move(ret.second);
            return;
        }

        record_tuning_node(tree, i, begin, end);

        const auto s =
            split(begin, end, random_dims[tree_level], train, knn, tol, n_corpus, scratch);
        const int max_dim = std::get<0>(s);
        const float max_split = std::get<1>(s);

        if (max_dim == -1) {
            split_dimensions(i, tree) = UINT32_MAX;
            if (tuning_structure_only)
                return;
            const int levels2leaf = depth - tree_level;
            const int index_leaf = (1 << levels2leaf) * (i + 1) - 1 - n_inner_nodes;
            auto ret = count_votes(begin, end, knn, scratch);
            labels_tree[index_leaf] = std::move(ret.first);
            votes_tree[index_leaf] = std::move(ret.second);
            return;
        }

        const float* data = train.data();
        const int cols = train.cols();
        auto mid = std::partition(begin, end, [data, cols, max_dim, max_split](const int em) {
            const size_t offset =
                static_cast<size_t>(em) * static_cast<size_t>(cols) + static_cast<size_t>(max_dim);
            return data[offset] <= max_split;
        });

        split_points(i, tree) = max_split;
        split_dimensions(i, tree) = static_cast<uint32_t>(max_dim);

        const int idx_left = 2 * i + 1;
        const int idx_right = idx_left + 1;
        grow_subtree(
            begin,
            mid,
            tree_level + 1,
            idx_left,
            tree,
            labels_tree,
            votes_tree,
            train,
            knn,
            random_dims,
            scratch
        );
        grow_subtree(
            mid,
            end,
            tree_level + 1,
            idx_right,
            tree,
            labels_tree,
            votes_tree,
            train,
            knn,
            random_dims,
            scratch
        );
    }

    void route_batch(const float* query, int first, int count, int* leaves) const {
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
};
