#pragma once

#include <Eigen/Dense>
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

// Randomized k-d forest: choose uniformly among the top_variance_dims highest-variance
// coordinates at each node, then split at the median.
class KD : public MLANN {
  public:
    KD(const float* corpus_, int n_corpus_, int dim_, int top_variance_dims_ = 5)
        : MLANN(corpus_, n_corpus_, dim_) {
        configure(top_variance_dims_);
    }

    void configure(int top_variance_dims_) {
        if (!empty())
            throw std::logic_error("The index has already been grown.");
        if (top_variance_dims_ < 1)
            throw std::invalid_argument("top_variance_dims must be positive.");
        top_variance_dims = top_variance_dims_;
    }

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
        if (dim <= 0 || n_corpus <= 0 || train.cols() != dim || b_ < 1 || !train.allFinite() ||
            (!unsupervised &&
             (knn.rows() != n_train || knn.cols() < 1 || knn.maxCoeff() >= uint32_t(n_corpus)))) {
            throw std::invalid_argument("Invalid forest data or dimensions.");
        }
        // KD always ranks all input dimensions; density is accepted for API compatibility.
        (void) density_;

        corpus_leaves = unsupervised;
        n_trees = n_trees_;
        depth = depth_;
        n_inner_nodes = (1 << depth) - 1;
        n_leaves = 1 << depth;
        n_array = 1 << (depth + 1);
        density = 1.f;
        b = b_;

        split_points.resize(n_inner_nodes, n_trees);
        split_dimensions.resize(n_inner_nodes, n_trees);
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
            TreeScratch scratch(corpus_leaves ? 0 : n_corpus, n_train, dim);
#pragma omp for schedule(dynamic, 1) nowait
            for (int tree = 0; tree < n_trees; ++tree) {
                labels_all[tree].resize(n_leaves);
                if (compact_leaf_votes)
                    votes16_all[tree].resize(n_leaves);
                else if (!corpus_leaves)
                    votes_all[tree].resize(n_leaves);
                std::iota(scratch.rows.begin(), scratch.rows.end(), 0);
                std::random_device rd;
                std::minstd_rand generator(rd());
                grow_subtree(
                    scratch.rows.begin(),
                    scratch.rows.end(),
                    0,
                    0,
                    tree,
                    knn,
                    train,
                    generator,
                    scratch
                );
                finish_tuning_tree(tree, scratch.rows);
            }
        }
        mlann_detail::promote_existing_corpus_pages(
            corpus.data(), size_t(corpus.size()) * sizeof(float)
        );
    }

  public:
    size_t index_bytes() const override {
        return MLANN::index_bytes() + sizeof(KD) - sizeof(MLANN) + payload_bytes(votes16_all);
    }

    std::unique_ptr<MLANN> make_view(int trees, int d) const override {
        auto view = std::make_unique<KD>(corpus.data(), n_corpus, dim, top_variance_dims);
        initialize_view(*view, trees, d);
        view->corpus_leaves = tuning_unit_labels;
        view->compact_leaf_votes = view->compact_view_votes(tuning_unit_labels, view->votes16_all);
        return view;
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

        std::array<int, routing_batch_size> leaves;
        for (int first = 0; first < n_trees; first += routing_batch_size) {
            const int count = std::min(routing_batch_size, n_trees - first);
            route_batch(data, first, count, leaves.data());
            for (int t = 0; t < count; ++t) {
                const int leaf = leaves[t];
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
        path[0] = 0;
        for (int level = 0; level < depth; ++level) {
            const int node = path[level];
            path[level + 1] =
                2 * node + (q[split_dimensions(node, tree)] <= split_points(node, tree) ? 1 : 2);
        }
    }
    bool compact_leaf_votes = false;
    std::vector<std::vector<std::vector<uint16_t>>> votes16_all;

  private:
    using IndexIterator = std::vector<int>::iterator;
    static constexpr int routing_batch_size = 64;
    bool corpus_leaves = false;
    int top_variance_dims = 5;

    struct TreeScratch {
        std::vector<int> rows, dimensions, votes;
        std::vector<uint32_t> touched_ids;
        std::vector<float> row_scores;
        Eigen::RowVectorXd mean, variance;

        TreeScratch(int corpus_size, int train_size, int dim)
            : rows(train_size), dimensions(dim), votes(corpus_size, 0), row_scores(train_size),
              mean(dim), variance(dim) {}
    };

    int choose_dimension(
        IndexIterator begin,
        IndexIterator end,
        const Eigen::Ref<const RowMatrix>& train,
        std::minstd_rand& generator,
        TreeScratch& scratch
    ) const {
        // Two-pass double accumulation keeps small variances accurate at large offsets.
        scratch.mean.setZero();
        for (auto it = begin; it != end; ++it)
            scratch.mean += train.row(*it).cast<double>();
        scratch.mean /= double(end - begin);
        scratch.variance.setZero();
        for (auto it = begin; it != end; ++it) {
            scratch.variance.array() +=
                (train.row(*it).cast<double>() - scratch.mean).array().square();
        }
        // The common variance denominator cannot change the dimension ranking.
        std::iota(scratch.dimensions.begin(), scratch.dimensions.end(), 0);
        const int count = std::min(top_variance_dims, dim);
        if (count < dim) {
            miniselect::pdqpartial_sort_branchless(
                scratch.dimensions.begin(),
                scratch.dimensions.begin() + count,
                scratch.dimensions.end(),
                [&](int left, int right) {
                    if (scratch.variance[left] != scratch.variance[right]) {
                        return scratch.variance[left] > scratch.variance[right];
                    }
                    return left < right;
                }
            );
        }
        return scratch.dimensions[std::uniform_int_distribution<int>(0, count - 1)(generator)];
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

    void grow_subtree(
        IndexIterator begin,
        IndexIterator end,
        int level,
        int node,
        int tree,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        std::minstd_rand& generator,
        TreeScratch& scratch
    ) {
        if (level == depth) {
            record_tuning_node(tree, node, begin, end);
            make_leaf(begin, end, tree, node - n_inner_nodes, knn, scratch);
            return;
        }
        record_tuning_node(tree, node, begin, end);
        const int dimension = choose_dimension(begin, end, train, generator, scratch);
        split_dimensions(node, tree) = dimension;
        for (auto it = begin; it != end; ++it)
            scratch.row_scores[*it] = train(*it, dimension);
        const auto less = [&](int left, int right) {
            return scratch.row_scores[left] < scratch.row_scores[right];
        };
        const int count = end - begin;
        miniselect::pdqselect_branchless(begin, begin + count / 2, end, less);
        const auto mid = end - count / 2;
        if (count % 2) {
            split_points(node, tree) = scratch.row_scores[*(mid - 1)];
        } else {
            const auto left = std::max_element(begin, mid, less);
            const float lower = scratch.row_scores[*left], upper = scratch.row_scores[*mid];
            // Avoid overflow and keep adjacent floats on opposite sides of the split.
            float threshold = (double(lower) + upper) / 2.0;
            if (lower < upper && threshold >= upper)
                threshold = lower;
            split_points(node, tree) = threshold;
        }
        grow_subtree(begin, mid, level + 1, 2 * node + 1, tree, knn, train, generator, scratch);
        grow_subtree(mid, end, level + 1, 2 * node + 2, tree, knn, train, generator, scratch);
    }

    void route_batch(const float* query, int first, int count, int* leaves) const {
        std::array<int, routing_batch_size> nodes{};
        for (int level = 0; level < depth; ++level) {
            for (int t = 0; t < count; ++t) {
                const int tree = first + t;
                const int node = nodes[t];
                nodes[t] =
                    2 * node +
                    (query[split_dimensions(node, tree)] <= split_points(node, tree) ? 1 : 2);
            }
        }
        for (int t = 0; t < count; ++t)
            leaves[t] = nodes[t] - n_inner_nodes;
    }
};
