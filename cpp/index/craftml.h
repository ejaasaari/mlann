#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../detail/clustering.h"
#include "../detail/huge-buffer.h"
#include "../detail/neighbor-query.h"
#include "../mlann.h"
#include "../utils.h"

struct CraftMLOptions {
    int n_trees = 10;
    int max_depth = 20;
    int branching_factor = 16;
    int leaf_size = 32;
    int label_dim = 512;
    int feature_dim = 0; // Zero means A = I, including for raw L2 data.
    int iterations = 2;
    int node_sample_size = 1000;
    uint32_t seed = 42;
    Distance distance = L2;
};

// CRAFTML (Siblini et al., ICML 2018), adapted to corpus-ID multilabel targets.
// SxSy signed hashing, spherical label k-means++, and feature-centroid routing.
// All training queries participate in every tree; only split fitting is sampled.
class CraftML : public MLANN {
  public:
    struct LabelScore {
        uint32_t id;
        float score;
    };

    struct Node {
        RowMatrix centroids;
        std::vector<uint32_t> children;
        std::vector<LabelScore> labels;
        int training_size = 0;

        bool is_leaf() const { return children.empty(); }
    };

    struct Tree {
        uint64_t feature_seed = 0;
        uint64_t label_seed = 0;
        std::vector<Node> nodes;
    };

    struct QueryStats {
        size_t visited_labels = 0;
        size_t unique_labels = 0;
        int candidates = 0;
        bool fallback = false;
    };

    CraftML(const float* corpus_, int n_corpus_, int dim_) : MLANN(corpus_, n_corpus_, dim_) {
        if (!corpus_ || n_corpus_ <= 0 || dim_ <= 0) {
            throw std::invalid_argument("Corpus must be nonempty");
        }
    }

    const std::vector<Tree>& trees() const { return forest; }
    Distance distance() const { return options.distance; }

    void grow(
        int n_trees_,
        int depth_,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        float density_ = -1.0,
        int b_ = 1
    ) override {
        if (density_ != -1 || b_ != 1) {
            throw std::invalid_argument(
                "Use CraftML::build to configure feature hashing and leaves"
            );
        }

        CraftMLOptions build_options;
        build_options.n_trees = n_trees_;
        build_options.max_depth = depth_;
        build(knn, train, build_options);
    }

    void build(
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        const CraftMLOptions& options_ = {}
    ) {
        if (!empty()) {
            throw std::logic_error("The index has already been built");
        }

        validate(knn, train, options_);
        options = options_;
        std::vector<Tree> new_forest(options_.n_trees);
        std::exception_ptr error;
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            TreeScratch scratch(n_corpus, train.rows(), options_.max_depth);
            // Release each worker's scratch as soon as its last tree finishes.
            // The parallel-region barrier still waits for all trees.
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1) nowait
#endif
            for (int t = 0; t < options_.n_trees; ++t) {
                try {
                    Tree& tree = new_forest[t];
                    tree.feature_seed = mlann_detail::mix(uint64_t(options_.seed) + 2ULL * t);
                    tree.label_seed = mlann_detail::mix(uint64_t(options_.seed) + 2ULL * t + 1);
                    std::mt19937 generator(
                        static_cast<uint32_t>(mlann_detail::mix(tree.label_seed))
                    );
                    auto& projected = scratch.projected;
                    if (options_.feature_dim > 0) {
                        projected.resize(train.rows(), options_.feature_dim);
                        scratch.feature_hashes.resize(dim);
                        for (int j = 0; j < dim; ++j)
                            scratch.feature_hashes[j] =
                                mlann_detail::mix(tree.feature_seed ^ uint64_t(j));
                        for (int i = 0; i < train.rows(); ++i) {
                            float* out = projected.row(i).data();
                            std::fill_n(out, options_.feature_dim, 0.f);
                            for (int j = 0; j < dim; ++j) {
                                const uint64_t hash = scratch.feature_hashes[j];
                                out[hash % options_.feature_dim] +=
                                    (hash >> 63) ? train(i, j) : -train(i, j);
                            }
                        }
                    }
                    const Eigen::Map<const RowMatrix, 0, Eigen::OuterStride<>> features(
                        options_.feature_dim > 0 ? projected.data() : train.data(),
                        train.rows(),
                        options_.feature_dim > 0 ? options_.feature_dim : dim,
                        Eigen::OuterStride<>(
                            options_.feature_dim > 0 ? projected.outerStride() : train.outerStride()
                        )
                    );
                    auto& ids = scratch.ids;
                    std::iota(ids.begin(), ids.end(), 0);
                    grow_node(tree, ids, 0, ids.size(), 0, features, knn, generator, scratch);
                } catch (...) {
#ifdef _OPENMP
#pragma omp critical(craftml_build_error)
#endif
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }
        }
        if (error)
            std::rethrow_exception(error);
        forest = std::move(new_forest);
        n_trees = options_.n_trees;
        mlann_detail::promote_existing_corpus_pages(
            corpus.data(), size_t(corpus.size()) * sizeof(float)
        );
    }

    // Unpruned forest probabilities in original corpus-ID space, sorted by ID.
    std::vector<LabelScore> predict(const float* q) const {
        auto& scratch = query_scratch();
        accumulate(q, scratch, nullptr);
        std::vector<LabelScore> result;
        result.reserve(scratch.touched.size());
        for (uint32_t id : scratch.touched)
            result.push_back({id, scratch.votes[id] / n_trees});
        miniselect::pdqsort_branchless(
            result.begin(), result.end(), [](const auto& a, const auto& b) { return a.id < b.id; }
        );
        return result;
    }

    using MLANN::query;

    void query(
        const float* data,
        int k,
        float vote_threshold,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr,
        int* out_n_elected = nullptr
    ) const override {
        search(data, k, -1, vote_threshold, out, dist, out_distances, out_n_elected);
    }

    // budget == -1 selects strict score > threshold; otherwise budget must be >= k.
    // Too few supported candidates triggers exact full-corpus search, never short output.
    void search(
        const float* q,
        int k,
        int budget,
        float threshold,
        int* out,
        Distance dist = L2,
        float* distances = nullptr,
        int* elected = nullptr,
        QueryStats* stats = nullptr
    ) const {
        if (k <= 0 || k > n_corpus || !out) {
            throw std::invalid_argument("k must be in [1, corpus size]");
        }
        if (dist != options.distance) {
            throw std::invalid_argument("Search metric must match build metric");
        }
        if (budget != -1 && budget < k) {
            throw std::invalid_argument("candidate_budget must be >= k");
        }
        if (budget == -1 && (!std::isfinite(threshold) || threshold < 0 || threshold > 1)) {
            throw std::invalid_argument("Probability threshold must be in [0, 1]");
        }

        auto& scratch = query_scratch();
        if (stats)
            *stats = {};
        accumulate(q, scratch, stats);
        scratch.candidates.clear();
        if (budget == -1) {
            for (uint32_t id : scratch.touched)
                if (scratch.votes[id] / n_trees > threshold)
                    scratch.candidates.push_back(id);
        } else {
            scratch.ranked.clear();
            for (uint32_t id : scratch.touched)
                scratch.ranked.push_back({id, scratch.votes[id] / n_trees});
            if (scratch.ranked.size() > static_cast<size_t>(budget)) {
                miniselect::pdqselect_branchless(
                    scratch.ranked.begin(),
                    scratch.ranked.begin() + budget,
                    scratch.ranked.end(),
                    [](const auto& a, const auto& b) {
                        return a.score > b.score || (a.score == b.score && a.id < b.id);
                    }
                );
                scratch.ranked.resize(budget);
            }
            for (const auto& entry : scratch.ranked)
                scratch.candidates.push_back(entry.id);
        }
        const bool fallback = scratch.candidates.size() < static_cast<size_t>(k);
        const int count = fallback ? n_corpus : static_cast<int>(scratch.candidates.size());
        if (elected)
            *elected = count;
        if (stats) {
            stats->candidates = count;
            stats->fallback = fallback;
        }
        if (fallback) {
            scratch.candidates.resize(n_corpus);
            std::iota(scratch.candidates.begin(), scratch.candidates.end(), 0);
        }
        // Sequential corpus access also makes the k=1 tie break agree with exact search.
        miniselect::pdqsort_branchless(scratch.candidates.begin(), scratch.candidates.end());
        const Eigen::Map<const Eigen::RowVectorXf> query(q, dim);
        MLANN::exact_knn(
            query,
            k,
            scratch.candidates,
            out,
            dist,
            distances,
            mlann_detail::compute_neighbor_scores,
            mlann_detail::compute_neighbor_topk
        );
    }

  private:
    CraftMLOptions options;
    std::vector<Tree> forest;

    static constexpr int routing_batch_size = 64;

    // One workspace per build worker; fitting arrays are reused after partitioning a node.
    // Only child boundaries must survive recursive calls, so those are kept per depth.
    struct TreeScratch {
        std::vector<int> ids;
        std::vector<int> grouped;
        std::vector<int> routes;
        std::vector<int> assignment;
        std::vector<int> sizes;
        std::vector<uint32_t> counts;
        std::vector<uint32_t> touched;
        std::vector<uint64_t> feature_hashes;
        std::vector<std::vector<size_t>> boundaries;
        std::vector<size_t> offsets;
        RowMatrix labels;
        RowMatrix centers;
        RowMatrix classifier;
        RowMatrix projected;
        Eigen::VectorXf nearest;

        TreeScratch(int corpus_size, int train_size, int max_depth)
            : ids(train_size), grouped(train_size), routes(train_size), counts(corpus_size, 0),
              boundaries(max_depth + 1) {}
    };

    struct QueryScratch {
        mlann_detail::HugeBuffer<float> votes;
        std::vector<uint32_t> touched;
        std::vector<uint32_t> candidates;
        std::vector<LabelScore> ranked;
        RowMatrix projected;

        void clear(int corpus_size) {
            if (votes.size() != size_t(corpus_size)) {
                votes.resize(corpus_size);
                std::fill_n(votes.data(), corpus_size, 0.f);
            } else {
                for (uint32_t id : touched)
                    votes[id] = 0.f;
            }
            touched.clear();
        }

        // Leaf IDs are unique and probabilities positive. Zero marks unseen IDs.
        // Preserve the accumulation order across trees, including in SIMD batches.
        void add(const std::vector<LabelScore>& labels) {
            size_t i = 0;
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
            static_assert(sizeof(LabelScore) == 8 && offsetof(LabelScore, score) == 4);
            const __m512i offsets =
                _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            for (; i + 16 <= labels.size(); i += 16) {
                for (size_t j = i + 32; j < labels.size() && j < i + 48; ++j)
                    __builtin_prefetch(votes.data() + labels[j].id, 1, 1);
                const __m512i ids = _mm512_i32gather_epi32(offsets, &labels[i].id, 4);
                const __m512 weights = _mm512_i32gather_ps(offsets, &labels[i].score, 4);
                const __m512 previous = _mm512_i32gather_ps(ids, votes.data(), 4);
                unsigned fresh = _mm512_cmp_ps_mask(previous, _mm512_setzero_ps(), _CMP_EQ_OQ);
                _mm512_i32scatter_ps(votes.data(), ids, _mm512_add_ps(previous, weights), 4);
                while (fresh) {
                    touched.push_back(labels[i + __builtin_ctz(fresh)].id);
                    fresh &= fresh - 1;
                }
            }
#endif
            for (; i < labels.size(); ++i) {
#if defined(__GNUC__) || defined(__clang__)
                if (i + 32 < labels.size())
                    __builtin_prefetch(votes.data() + labels[i + 32].id, 1, 1);
#endif
                const auto& entry = labels[i];
                if (votes[entry.id] == 0.f)
                    touched.push_back(entry.id);
                votes[entry.id] += entry.score;
            }
        }
    };

    static QueryScratch& query_scratch() {
        static thread_local QueryScratch scratch;
        return scratch;
    }

    static void project(const float* x, int n, float* out, int p, uint64_t seed) {
        std::fill(out, out + p, 0);
        for (int j = 0; j < n; ++j) {
            const uint64_t hash = mlann_detail::mix(seed ^ uint64_t(j));
            out[hash % p] += (hash >> 63) ? x[j] : -x[j];
        }
    }

    void validate(
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        const CraftMLOptions& options_
    ) const {
        if (options_.n_trees <= 0 || options_.max_depth < 0 || options_.max_depth > 64 ||
            options_.branching_factor < 2 || options_.leaf_size <= 0 || options_.label_dim <= 0 ||
            options_.feature_dim < 0 || options_.iterations <= 0 ||
            options_.node_sample_size < options_.branching_factor ||
            (options_.distance != IP && options_.distance != L2)) {
            throw std::invalid_argument("Invalid CraftML build parameters");
        }
        if (!train.rows() || train.cols() != dim ||
            train.rows() > std::numeric_limits<int>::max() || knn.rows() != train.rows() ||
            knn.cols() <= 0 || knn.cols() > n_corpus) {
            throw std::invalid_argument("Incompatible training features or neighbor labels");
        }
        if (!train.allFinite() || !corpus.allFinite()) {
            throw std::invalid_argument("Corpus and training features must be finite");
        }

        std::vector<uint32_t> row(knn.cols());
        for (int i = 0; i < knn.rows(); ++i) {
            std::copy(knn.row(i).data(), knn.row(i).data() + knn.cols(), row.begin());
            miniselect::pdqsort_branchless(row.begin(), row.end());
            if (row.back() >= static_cast<uint32_t>(n_corpus) ||
                std::adjacent_find(row.begin(), row.end()) != row.end()) {
                throw std::invalid_argument(
                    "Neighbor labels must be distinct valid corpus IDs per query"
                );
            }
        }
    }

    void check_query(const float* q) const {
        if (empty()) {
            throw std::logic_error("Cannot query before building index");
        }
        if (!q || !Eigen::Map<const Eigen::RowVectorXf>(q, dim).allFinite()) {
            throw std::invalid_argument("Query features must be finite");
        }
    }

    template <class Derived>
    int route(const float* q, const Eigen::MatrixBase<Derived>& centroids) const {
        const Eigen::Map<const Eigen::RowVectorXf> x(q, centroids.cols());
        int best = 0;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < centroids.rows(); ++c) {
            const float score = options.distance == IP ? x.dot(centroids.row(c))
                                                       : -(x - centroids.row(c)).squaredNorm();
            if (score > best_score) {
                best_score = score;
                best = c;
            }
        }
        return best;
    }

    // Leave raw sums in the table; callers normalize while collecting results.
    void accumulate(const float* q, QueryScratch& scratch, QueryStats* stats) const {
        check_query(q);
        scratch.clear(n_corpus);
        const int feature_dim = options.feature_dim;
        if (feature_dim > 0)
            scratch.projected.resize(routing_batch_size, feature_dim);
        for (int first = 0; first < n_trees; first += routing_batch_size) {
            const int count = std::min(routing_batch_size, n_trees - first);
            std::array<uint32_t, routing_batch_size> nodes{};
            std::array<int, routing_batch_size> active;
            for (int t = 0; t < count; ++t) {
                active[t] = t;
                if (feature_dim > 0)
                    project(
                        q,
                        dim,
                        scratch.projected.row(t).data(),
                        feature_dim,
                        forest[first + t].feature_seed
                    );
            }
            int remaining = count;
            while (remaining) {
                int next = 0;
                for (int i = 0; i < remaining; ++i) {
                    const int t = active[i];
                    const auto& tree = forest[first + t];
                    const auto& node = tree.nodes[nodes[t]];
                    if (node.is_leaf())
                        continue;
                    const float* features = feature_dim > 0 ? scratch.projected.row(t).data() : q;
                    nodes[t] = node.children[route(features, node.centroids)];
                    active[next++] = t;
                }
                remaining = next;
            }
            for (int t = 0; t < count; ++t) {
                const auto& labels = forest[first + t].nodes[nodes[t]].labels;
                if (stats)
                    stats->visited_labels += labels.size();
                scratch.add(labels);
            }
        }
        if (stats)
            stats->unique_labels = scratch.touched.size();
    }

    void make_leaf(
        Node& node,
        const std::vector<int>& ids,
        size_t begin,
        size_t end,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        TreeScratch& scratch
    ) const {
        scratch.touched.clear();
        for (size_t i = begin; i < end; ++i) {
            const uint32_t* labels = knn.row(ids[i]).data();
            for (int j = 0; j < knn.cols(); ++j)
                if (scratch.counts[labels[j]]++ == 0)
                    scratch.touched.push_back(labels[j]);
        }
        miniselect::pdqsort_branchless(scratch.touched.begin(), scratch.touched.end());
        node.labels.resize(scratch.touched.size());
        for (size_t i = 0; i < scratch.touched.size(); ++i) {
            const uint32_t id = scratch.touched[i];
            node.labels[i] = {id, static_cast<float>(scratch.counts[id]) / (end - begin)};
            scratch.counts[id] = 0;
        }
    }

    // Sample query rows and hash their corpus-ID labels into normalized label vectors.
    void sample_labels(
        std::vector<int>& ids,
        size_t begin,
        size_t end,
        uint64_t label_seed,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        std::mt19937& generator,
        TreeScratch& scratch
    ) const {
        const int sample_size =
            static_cast<int>(std::min(end - begin, size_t(options.node_sample_size)));

        // A partial Fisher-Yates shuffle samples without replacement in O(sample size).
        for (int i = 0; i < sample_size; ++i) {
            std::uniform_int_distribution<size_t> pick(begin + i, end - 1);
            std::swap(ids[begin + i], ids[pick(generator)]);
        }

        auto& labels = scratch.labels;
        labels.resize(sample_size, options.label_dim);
        labels.setZero();
        for (int i = 0; i < sample_size; ++i) {
            for (int j = 0; j < knn.cols(); ++j) {
                const uint64_t hash =
                    mlann_detail::mix(label_seed ^ uint64_t(knn(ids[begin + i], j)));
                // Arithmetic avoids a branch on the random sign bit.
                const float sign = float(2 * int(hash >> 63) - 1);
                labels(i, hash % options.label_dim) += sign;
            }
            labels.row(i).stableNormalize(); // Zero sketches remain zero.
        }
    }

    // Choose label centers with k-means++ sampling; coincident labels may yield one center.
    int initialize_label_centers(std::mt19937& generator, TreeScratch& scratch) const {
        const auto& labels = scratch.labels;
        const int sample_size = labels.rows();
        const int branches = std::min(options.branching_factor, sample_size);

        auto& centers = scratch.centers;
        centers.resize(branches, options.label_dim);
        std::uniform_int_distribution<int> first(0, sample_size - 1);
        centers.row(0) = labels.row(first(generator));

        auto& nearest = scratch.nearest;
        nearest.resize(sample_size);
        nearest.setConstant(std::numeric_limits<float>::infinity());

        int count = 1;
        while (count < branches) {
            for (int i = 0; i < sample_size; ++i) {
                const float distance = (labels.row(i) - centers.row(count - 1)).squaredNorm();
                nearest(i) = std::min(nearest(i), distance);
            }

            const double total = nearest.cast<double>().sum();
            if (total <= 1e-12)
                break;

            std::uniform_real_distribution<double> pick(0, total);
            double target = pick(generator);
            int selected = sample_size - 1;
            for (int i = 0; i < sample_size; ++i) {
                target -= nearest(i);
                if (target < 0) {
                    selected = i;
                    break;
                }
            }
            centers.row(count++) = labels.row(selected);
        }

        scratch.centers.conservativeResize(count, Eigen::NoChange);
        return count;
    }

    void cluster_labels(TreeScratch& scratch) const {
        mlann_detail::KMeans::refine(
            scratch.labels,
            scratch.centers,
            scratch.assignment,
            scratch.sizes,
            true,
            options.iterations
        );
    }

    // Average query features within each label cluster to obtain routing centroids.
    int fit_feature_centroids(
        const std::vector<int>& ids,
        size_t begin,
        const Eigen::Ref<const RowMatrix>& features,
        TreeScratch& scratch
    ) const {
        const auto& assignment = scratch.assignment;
        const auto& sizes = scratch.sizes;
        const int sample_size = assignment.size();
        const int count = sizes.size();

        auto& classifier = scratch.classifier;
        classifier.resize(count, features.cols());
        classifier.setZero();
        for (int i = 0; i < sample_size; ++i) {
            classifier.row(assignment[i]) += features.row(ids[begin + i]);
        }

        int fitted = 0;
        for (int c = 0; c < count; ++c) {
            if (!sizes[c])
                continue;
            classifier.row(c) /= sizes[c];
            if (options.distance == IP)
                classifier.row(c).stableNormalize();
            classifier.row(fitted++) = classifier.row(c);
        }

        classifier.conservativeResize(fitted, Eigen::NoChange);
        return fitted;
    }

    bool partition_children(
        Node& node,
        std::vector<int>& ids,
        size_t begin,
        size_t end,
        int level,
        const Eigen::Ref<const RowMatrix>& features,
        TreeScratch& scratch
    ) const {
        const auto& classifier = scratch.classifier;
        const int fitted = classifier.rows();
        auto& sizes = scratch.sizes;

        // Stable counting partition preserves row order and random draws in each child.
        sizes.assign(fitted, 0);
        for (size_t i = begin; i < end; ++i) {
            const int child = route(features.row(ids[i]).data(), classifier);
            scratch.routes[i] = child;
            ++sizes[child];
        }

        int occupied = 0;
        for (int size : sizes)
            occupied += size != 0;
        if (occupied < 2)
            return false;

        // Store only nonempty children and their contiguous row ranges.
        node.centroids.resize(occupied, features.cols());
        node.children.resize(occupied);
        auto& boundaries = scratch.boundaries[level];
        boundaries.resize(occupied + 1);
        scratch.offsets.resize(fitted);

        size_t offset = begin;
        int child = 0;
        for (int c = 0; c < fitted; ++c) {
            scratch.offsets[c] = offset;
            if (!sizes[c])
                continue;
            node.centroids.row(child) = classifier.row(c);
            boundaries[child++] = offset;
            offset += sizes[c];
        }
        boundaries[occupied] = end;

        for (size_t i = begin; i < end; ++i) {
            const int child = scratch.routes[i];
            const size_t destination = scratch.offsets[child]++;
            scratch.grouped[destination] = ids[i];
        }

        std::copy(
            scratch.grouped.begin() + begin, scratch.grouped.begin() + end, ids.begin() + begin
        );
        return true;
    }

    uint32_t grow_node(
        Tree& tree,
        std::vector<int>& ids,
        size_t begin,
        size_t end,
        int level,
        const Eigen::Ref<const RowMatrix>& features,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        std::mt19937& generator,
        TreeScratch& scratch
    ) const {
        const uint32_t node_id = static_cast<uint32_t>(tree.nodes.size());
        tree.nodes.emplace_back();
        tree.nodes[node_id].training_size = static_cast<int>(end - begin);

        const auto make_node_leaf = [&]() {
            make_leaf(tree.nodes[node_id], ids, begin, end, knn, scratch);
            return node_id;
        };

        if (end - begin <= static_cast<size_t>(options.leaf_size) || level >= options.max_depth) {
            return make_node_leaf();
        }

        sample_labels(ids, begin, end, tree.label_seed, knn, generator, scratch);
        if (initialize_label_centers(generator, scratch) < 2) {
            return make_node_leaf();
        }

        cluster_labels(scratch);
        if (fit_feature_centroids(ids, begin, features, scratch) < 2) {
            return make_node_leaf();
        }

        if (!partition_children(tree.nodes[node_id], ids, begin, end, level, features, scratch)) {
            return make_node_leaf();
        }

        // Only per-depth boundaries survive recursion; fitting buffers are shared.
        const auto& boundaries = scratch.boundaries[level];
        const int n_children = tree.nodes[node_id].children.size();
        for (int child = 0; child < n_children; ++child) {
            const uint32_t child_id = grow_node(
                tree,
                ids,
                boundaries[child],
                boundaries[child + 1],
                level + 1,
                features,
                knn,
                generator,
                scratch
            );
            // Fetch the node again because recursive growth can reallocate tree.nodes.
            tree.nodes[node_id].children[child] = child_id;
        }

        return node_id;
    }
};
