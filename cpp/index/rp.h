#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../detail/huge-buffer.h"
#include "../detail/neighbor-query.h"
#include "../mlann.h"

// Median-split forest with one Gaussian projection per tree level. Below
// density 1, each coordinate is included independently with that probability.
class RP : public MLANN {
  public:
    RP(const float* corpus_, int n_corpus_, int dim_) : MLANN(corpus_, n_corpus_, dim_) {}

    void grow(
        int n_trees_,
        int depth_,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        float density_ = -1.0,
        int b_ = 1
    ) override {
        grow_impl(n_trees_, depth_, knn, train, density_, b_, false);
    }

    // Partition the corpus itself; leaves hold corpus IDs with implicit unit votes.
    void grow_unsupervised(int n_trees_, int depth_, float density_ = -1.0) override {
        grow_impl(n_trees_, depth_, UIntRowMatrix(), corpus, density_, 1, true);
    }

  private:
    void grow_impl(
        int n_trees_,
        int depth_,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        float density_,
        int b_,
        bool unsupervised
    ) {
        if (!empty()) {
            throw std::logic_error("The index has already been grown.");
        }
        if (n_trees_ <= 0) {
            throw std::out_of_range("The number of trees must be positive.");
        }
        const int n_train = train.rows();
        if (depth_ <= 0 || depth_ > std::log2(n_train) || depth_ > 29) {
            throw std::out_of_range(
                "The depth must belong to the set {1, ... , min(log2(n_train), 29)}."
            );
        }
        if (dim <= 0 || n_corpus <= 0 || train.cols() != dim ||
            (unsupervised && !train.allFinite()) ||
            (!unsupervised &&
             (knn.rows() != n_train || knn.cols() < 1 || knn.maxCoeff() >= uint32_t(n_corpus)))) {
            throw std::invalid_argument("Invalid forest data or dimensions.");
        }
        const float requested_density = density_ < 0 ? float(1.0 / std::sqrt(dim)) : density_;
        if (!std::isfinite(requested_density)) {
            throw std::invalid_argument("Density must be finite.");
        }
        if (n_trees_ > std::numeric_limits<int>::max() / depth_) {
            throw std::length_error("Too many random projections.");
        }

        corpus_leaves = unsupervised;
        n_trees = n_trees_;
        depth = depth_;
        n_inner_nodes = (1 << depth) - 1;
        n_leaves = 1 << depth;
        n_array = 1 << (depth + 1);
        n_pool = n_trees * depth;
        density = requested_density;
        b = b_;

        initialize_projections();
        split_points.resize(n_inner_nodes, n_trees);
        labels_all.resize(n_trees);
        prepare_tuning(knn, unsupervised, n_train);
        // This bound includes duplicate IDs within a training row. Larger raw
        // counts retain float storage, avoiding truncation at the uint16 limit.
        const uint64_t max_leaf_rows = (uint64_t(n_train) + n_leaves - 1) / n_leaves;
        compact_leaf_votes =
            !corpus_leaves &&
            max_leaf_rows <= std::numeric_limits<uint16_t>::max() / uint64_t(knn.cols());
        if (compact_leaf_votes)
            votes16_all.resize(n_trees);
        else if (!corpus_leaves)
            votes_all.resize(n_trees);

#pragma omp parallel
        {
            TreeScratch scratch(corpus_leaves ? 0 : n_corpus, n_train, depth);
#pragma omp for schedule(dynamic, 1) nowait
            for (int tree = 0; tree < n_trees; ++tree) {
                labels_all[tree].resize(n_leaves);
                if (compact_leaf_votes)
                    votes16_all[tree].resize(n_leaves);
                else if (!corpus_leaves)
                    votes_all[tree].resize(n_leaves);
                std::iota(scratch.rows.begin(), scratch.rows.end(), 0);
                if (density < 1) {
                    // Keep each sparse-projection level contiguous for selection.
                    scratch.projections.noalias() =
                        sparse_random_matrix.middleRows(tree * depth, depth) * train.transpose();
                    grow_subtree(
                        scratch.rows.begin(),
                        scratch.rows.end(),
                        0,
                        0,
                        tree,
                        knn,
                        scratch.projections,
                        scratch
                    );
                } else {
                    Eigen::Map<Eigen::MatrixXf> projections(
                        scratch.projection_storage.data(), depth, n_train
                    );
                    projections.noalias() =
                        dense_random_matrix.middleRows(tree * depth, depth) * train.transpose();
                    grow_subtree(
                        scratch.rows.begin(),
                        scratch.rows.end(),
                        0,
                        0,
                        tree,
                        knn,
                        projections,
                        scratch
                    );
                }
                finish_tuning_tree(tree, scratch.rows);
            }
        }
        mlann_detail::promote_existing_corpus_pages(
            corpus.data(), size_t(corpus.size()) * sizeof(float)
        );
    }

  public:
    std::unique_ptr<MLANN> make_view(int trees, int d) const override {
        auto view = std::make_unique<RP>(corpus.data(), n_corpus, dim);
        initialize_view(*view, trees, d);
        view->corpus_leaves = tuning_unit_labels;
        view->compact_leaf_votes = view->compact_view_votes(tuning_unit_labels, view->votes16_all);
        if (density < 1) {
            view->sparse_random_matrix.resize(trees * d, dim);
            std::vector<Eigen::Triplet<float>> entries;
            for (int t = 0; t < trees; ++t)
                for (int l = 0; l < d; ++l)
                    for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(
                             sparse_random_matrix, t * depth + l
                         );
                         it;
                         ++it)
                        entries.emplace_back(t * d + l, it.col(), it.value());
            view->sparse_random_matrix.setFromTriplets(entries.begin(), entries.end());
            view->sparse_random_matrix.makeCompressed();
        } else {
            view->dense_random_matrix.resize(trees * d, dim);
            for (int t = 0; t < trees; ++t)
                view->dense_random_matrix.middleRows(t * d, d) =
                    dense_random_matrix.middleRows(t * depth, d);
        }
        return view;
    }

    size_t index_bytes() const override {
        return MLANN::index_bytes() + sizeof(RP) - sizeof(MLANN) + payload_bytes(votes16_all) +
               size_t(dense_random_matrix.size()) * sizeof(float) +
               size_t(sparse_random_matrix.nonZeros()) * (sizeof(float) + sizeof(int)) +
               (sparse_random_matrix.size()
                    ? size_t(sparse_random_matrix.outerSize() + 1) * sizeof(int)
                    : 0);
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
        static thread_local Eigen::VectorXf projected;
        static thread_local mlann_detail::HugeBuffer<float> votes_total;
        static thread_local std::vector<uint32_t> elected;
        projected.resize(n_pool);
        const Eigen::Map<const Eigen::VectorXf> query(data, dim);
        if (density < 1) {
            projected.noalias() = sparse_random_matrix * query;
        } else {
            projected.noalias() = dense_random_matrix * query;
        }
        votes_total.resize(n_corpus);
        std::fill_n(votes_total.data(), n_corpus, 0.f);
        elected.clear();

        std::array<int, routing_batch_size> leaves;
        for (int first = 0; first < n_trees; first += routing_batch_size) {
            const int count = std::min(routing_batch_size, n_trees - first);
            route_batch(projected.data(), first, count, leaves.data());
            for (int t = 0; t < count; ++t) {
                const int leaf = leaves[t];
                if (accumulate_tuning_votes(
                        first + t, leaf, votes_total.data(), vote_threshold, elected
                    ))
                    continue;
                if (corpus_leaves) {
                    mlann_detail::accumulate_unit_votes(
                        labels_all[first + t][leaf], votes_total.data(), vote_threshold, elected
                    );
                } else if (compact_leaf_votes) {
                    mlann_detail::accumulate_neighbor_votes(
                        labels_all[first + t][leaf],
                        votes16_all[first + t][leaf],
                        votes_total.data(),
                        vote_threshold,
                        elected
                    );
                } else {
                    mlann_detail::accumulate_neighbor_votes(
                        labels_all[first + t][leaf],
                        votes_all[first + t][leaf],
                        votes_total.data(),
                        vote_threshold,
                        elected
                    );
                }
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
    void tuning_path(const float* q, int tree, int* path) const override {
        Eigen::VectorXf projected;
        const Eigen::Map<const Eigen::VectorXf> query(q, dim);
        if (density < 1)
            projected = sparse_random_matrix.middleRows(tree * depth, depth) * query;
        else
            projected = dense_random_matrix.middleRows(tree * depth, depth) * query;
        path[0] = 0;
        for (int level = 0; level < depth; ++level) {
            const int node = path[level];
            path[level + 1] = 2 * node + (projected[level] <= split_points(node, tree) ? 1 : 2);
        }
    }
    bool compact_leaf_votes = false;
    std::vector<std::vector<std::vector<uint16_t>>> votes16_all;

  private:
    using IndexIterator = std::vector<int>::iterator;
    static constexpr int routing_batch_size = 64;
    static constexpr int cached_selection_min_size = 256;
    bool corpus_leaves = false;
    RowMatrix dense_random_matrix;
    Eigen::SparseMatrix<float, Eigen::RowMajor> sparse_random_matrix;

    struct ProjectionRow {
        float value;
        int row;
    };

    struct TreeScratch {
        std::vector<ProjectionRow> keys;
        std::vector<int> rows;
        std::vector<int> votes;
        std::vector<uint32_t> touched_ids;
        // Huge pages reduce translation overhead when gathering shuffled row IDs.
        mlann_detail::HugeBuffer<float> projection_storage;
        Eigen::Map<RowMatrix> projections;

        TreeScratch(int corpus_size, int train_size, int depth)
            : keys(train_size >= cached_selection_min_size ? train_size : 0), rows(train_size),
              votes(corpus_size, 0),
              projections(allocate_projections(depth, train_size), depth, train_size) {}

        TreeScratch(const TreeScratch&) = delete;
        TreeScratch& operator=(const TreeScratch&) = delete;

      private:
        float* allocate_projections(int depth, int train_size) {
            projection_storage.resize(size_t(depth) * train_size);
            return projection_storage.data();
        }
    };

    void initialize_projections() {
        std::random_device rd;
        std::minstd_rand generator(rd());
        std::normal_distribution<float> normal(0, 1);
        if (density < 1) {
            std::uniform_real_distribution<float> uniform(0, 1);
            sparse_random_matrix.resize(n_pool, dim);
            // Sorted coordinates allow direct CSR insertion.
            for (int row = 0; row < n_pool; ++row) {
                sparse_random_matrix.startVec(row);
                for (int column = 0; column < dim; ++column) {
                    if (uniform(generator) > density)
                        continue;
                    sparse_random_matrix.insertBack(row, column) = normal(generator);
                }
            }
            sparse_random_matrix.finalize();
            sparse_random_matrix.makeCompressed();
        } else {
            dense_random_matrix.resize(n_pool, dim);
            std::generate(
                dense_random_matrix.data(),
                dense_random_matrix.data() + dense_random_matrix.size(),
                [&] { return normal(generator); }
            );
        }
    }

    void make_leaf(
        IndexIterator begin,
        IndexIterator end,
        int tree,
        int leaf,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        TreeScratch& scratch
    ) {
        if (tuning_structure_only)
            return;
        if (corpus_leaves) {
            labels_all[tree][leaf].assign(begin, end);
            return;
        }
        scratch.touched_ids.clear();
        scratch.touched_ids.reserve(std::min<size_t>(n_corpus, size_t(end - begin) * knn.cols()));
        for (auto it = begin; it != end; ++it) {
            const uint32_t* labels = knn.row(*it).data();
            for (int j = 0; j < knn.cols(); ++j) {
                const auto label = labels[j];
                if (scratch.votes[label]++ == 0)
                    scratch.touched_ids.push_back(label);
            }
        }
        auto& labels = labels_all[tree][leaf];
        const auto store_votes = [&](auto& votes) {
            labels.reserve(scratch.touched_ids.size());
            votes.reserve(scratch.touched_ids.size());
            for (const auto label : scratch.touched_ids) {
                const int count = scratch.votes[label];
                scratch.votes[label] = 0;
                if (count >= b) {
                    labels.push_back(label);
                    votes.push_back(count);
                }
            }
        };
        if (compact_leaf_votes)
            store_votes(votes16_all[tree][leaf]);
        else
            store_votes(votes_all[tree][leaf]);
    }

    template <class ProjectionMatrix>
    void grow_subtree(
        IndexIterator begin,
        IndexIterator end,
        int level,
        int node,
        int tree,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const ProjectionMatrix& projections,
        TreeScratch& scratch
    ) {
        if (level == depth) {
            record_tuning_node(tree, node, begin, end);
            make_leaf(begin, end, tree, node - n_inner_nodes, knn, scratch);
            return;
        }
        record_tuning_node(tree, node, begin, end);
        const int count = end - begin;
        const auto mid = end - count / 2;
        if (count >= cached_selection_min_size) {
            // Contiguous keys avoid repeated shuffled gathers during selection.
            auto first = scratch.keys.begin();
            auto last = first + count;
            for (int i = 0; i < count; ++i)
                first[i] = {projections(level, begin[i]), begin[i]};
            const auto less = [](const ProjectionRow& left, const ProjectionRow& right) {
                return left.value < right.value;
            };
            miniselect::pdqselect_branchless(first, first + count / 2, last, less);
            const auto mid_key = last - count / 2;
            if (count % 2) {
                split_points(node, tree) = (mid_key - 1)->value;
            } else {
                const auto left = std::max_element(first, mid_key, less);
                split_points(node, tree) = (mid_key->value + left->value) / 2.0;
            }
            for (int i = 0; i < count; ++i)
                begin[i] = first[i].row;
        } else {
            const auto less = [&](int left, int right) {
                return projections(level, left) < projections(level, right);
            };
            miniselect::pdqselect_branchless(begin, begin + count / 2, end, less);
            if (count % 2) {
                split_points(node, tree) = projections(level, *(mid - 1));
            } else {
                const auto left = std::max_element(begin, mid, less);
                split_points(node, tree) =
                    (projections(level, *mid) + projections(level, *left)) / 2.0;
            }
        }
        grow_subtree(begin, mid, level + 1, 2 * node + 1, tree, knn, projections, scratch);
        grow_subtree(mid, end, level + 1, 2 * node + 2, tree, knn, projections, scratch);
    }

    void route_batch(const float* projected, int first, int count, int* leaves) const {
        std::array<int, routing_batch_size> nodes{};
        for (int level = 0; level < depth; ++level) {
            for (int t = 0; t < count; ++t) {
                const int tree = first + t;
                nodes[t] =
                    2 * nodes[t] +
                    (projected[tree * depth + level] <= split_points(nodes[t], tree) ? 1 : 2);
            }
        }
        for (int t = 0; t < count; ++t)
            leaves[t] = nodes[t] - n_inner_nodes;
    }
};
