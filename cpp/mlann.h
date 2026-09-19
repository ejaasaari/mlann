#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "detail/distance.h"
#include "detail/neighbor-query.h"
#include "miniselect/pdqselect.h"

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using UIntRowMatrix = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class MLANN {
  public:
    MLANN(const float* corpus_, int n_corpus_, int dim_)
        : corpus(Eigen::Map<const RowMatrix>(corpus_, n_corpus_, dim_)), n_corpus(n_corpus_),
          dim(dim_) {}

    virtual ~MLANN() = default;

    virtual void grow(
        int n_trees_,
        int depth_,
        const Eigen::Ref<const UIntRowMatrix>& knn_,
        const Eigen::Ref<const RowMatrix>& train_,
        float density_ = -1.0,
        int b_ = 1
    ) {}

    virtual void grow_unsupervised(int n_trees_, int depth_, float density_ = -1.0) {
        throw std::invalid_argument(
            "Unsupervised builds are supported only by KD, SparsePCA, PCA and RP."
        );
    }

    virtual void query(
        const float* data,
        int k,
        float vote_threshold,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr,
        int* out_n_elected = nullptr
    ) const {}

    void query(
        const Eigen::Ref<const Eigen::RowVectorXf>& q,
        int k,
        float vote_threshold,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr,
        int* out_n_elected = nullptr
    ) const {
        query(q.data(), k, vote_threshold, out, dist, out_distances, out_n_elected);
    }

    static void exact_knn(
        const float* q_data,
        const float* X_data,
        int n_corpus,
        int dim,
        int k,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr
    ) {
        static thread_local std::vector<uint32_t> indices;
        indices.resize(n_corpus);
        std::iota(indices.begin(), indices.end(), 0);

        exact_knn_impl(
            q_data,
            X_data,
            dim,
            k,
            indices,
            out,
            dist,
            out_distances,
            mlann_detail::compute_neighbor_scores,
            mlann_detail::compute_neighbor_topk
        );
    }

    static void exact_knn(
        const Eigen::Ref<const Eigen::RowVectorXf>& q,
        const Eigen::Ref<const RowMatrix>& corpus,
        int k,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr
    ) {
        MLANN::exact_knn(
            q.data(), corpus.data(), corpus.rows(), corpus.cols(), k, out, dist, out_distances
        );
    }

    void exact_knn(
        const float* q,
        int k,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr
    ) const {
        MLANN::exact_knn(q, corpus.data(), n_corpus, dim, k, out, dist, out_distances);
    }

    void exact_knn(
        const Eigen::Ref<const Eigen::RowVectorXf>& q,
        int k,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr
    ) const {
        MLANN::exact_knn(q.data(), corpus.data(), n_corpus, dim, k, out, dist, out_distances);
    }

    bool empty() const { return n_trees == 0; }

    // Enabled only for a tuning master. Ordinary builds retain no memberships.
    void enable_tuning(bool structure_only = false) {
        if (!empty())
            throw std::logic_error("Enable tuning before growing the master");
        retain_membership = true;
        tuning_structure_only = structure_only;
    }

    void enable_view_cache() {
        check_view(1, 1);
        std::lock_guard<std::mutex> lock(view_cache_mutex);
        cache_views = true;
    }

    void clear_view_cache() const {
        std::lock_guard<std::mutex> lock(view_cache_mutex);
        cached_payloads.clear();
        cached_depth = 0;
        // Weak entries retain no payloads and allow reuse from still-live subsets.
    }

    struct ViewCacheInfo {
        int depth, trees;
        size_t bytes, trees_built;
    };

    ViewCacheInfo view_cache_info() const {
        std::lock_guard<std::mutex> lock(view_cache_mutex);
        ViewCacheInfo info{cached_depth, 0, 0, payload_trees_built};
        for (const auto& payload : cached_payloads)
            if (payload) {
                ++info.trees;
                info.bytes += payload->bytes;
            }
        return info;
    }

    virtual std::unique_ptr<MLANN> make_view(int, int) const {
        throw std::invalid_argument("Autotuning supports KD, RP, SparsePCA, PCA, RF and PLS only");
    }

    virtual std::unique_ptr<MLANN> make_timing_view(int trees, int d) const {
        return make_view(trees, d);
    }

    struct Calibration {
        int trees, depth;
        float threshold;
        double recall;
    };

    std::vector<Calibration> calibrate(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& truth,
        int min_depth,
        double target
    ) const {
        check_view(n_trees, min_depth);
        if (queries.cols() != dim || queries.rows() == 0 || !queries.allFinite() ||
            truth.rows() != queries.rows() || truth.cols() == 0 ||
            truth.maxCoeff() >= uint32_t(n_corpus) || !(target > 0 && target <= 1))
            throw std::invalid_argument("Invalid calibration data or recall target");
        const size_t count = truth.size();
        const size_t h = size_t(std::ceil(target * count));
        std::vector<std::vector<float>> scores(
            depth - min_depth + 1, std::vector<float>(count, 0.f)
        );
        std::vector<Calibration> result;
        stream_tuning_scores(queries, truth, min_depth, [&](int tree, const float* weights) {
            const float divisor = probability_scores() ? float(tree + 1) : 1.f;
            for (int d = min_depth; d <= depth; ++d) {
                auto& score = scores[d - min_depth];
                for (size_t j = 0; j < count; ++j)
                    score[j] += weights[size_t(d - min_depth) * count + j];
                auto scratch = score; // Never partition the live accumulator.
                std::nth_element(
                    scratch.begin(), scratch.begin() + h - 1, scratch.end(), std::greater<float>()
                );
                const float threshold = scratch[h - 1] / divisor;
                if (threshold <= 0)
                    continue;
                size_t hits = 0;
                for (float s : score)
                    hits += s / divisor >= threshold;
                result.push_back({tree + 1, d, threshold, double(hits) / count});
            }
        });
        return result;
    }

    struct CostEstimate {
        double votes = 0, candidates = 0, seconds = 0;
        size_t bytes_upper_bound = 0;
    };

    struct FrontierConfiguration {
        Calibration configuration{0, 0, 0, 0};
        CostEstimate cost;
    };

    // Evaluate every empirical recall breakpoint, retaining at most m*k+1
    // records regardless of the number of trees, depths or vote thresholds.
    std::vector<FrontierConfiguration> calibrate_frontier(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& truth,
        int min_depth,
        const std::vector<uint32_t>& sample,
        int cost_queries,
        Distance dist,
        size_t memory_budget = 0,
        int query_k = 0
    ) const {
        check_view(n_trees, min_depth);
        // Calibration may use a uniform subset of the requested top-k neighbors.
        if (query_k == 0)
            query_k = int(truth.cols());
        if (queries.rows() == 0 || queries.cols() != dim || !queries.allFinite() ||
            truth.rows() != queries.rows() || truth.cols() == 0 || truth.cols() > n_corpus ||
            query_k < truth.cols() || query_k > n_corpus ||
            truth.maxCoeff() >= uint32_t(n_corpus) || sample.empty() || cost_queries < 1 ||
            cost_queries > queries.rows() || (dist != L2 && dist != IP))
            throw std::invalid_argument("Invalid frontier calibration data");
        for (auto id : sample)
            if (id >= uint32_t(n_corpus))
                throw std::invalid_argument("Invalid sampled corpus ID");

        std::vector<size_t> offsets(queries.rows() + 1, 0);
        for (int q = 0; q < queries.rows(); ++q)
            offsets[q + 1] = offsets[q] + truth.cols() + (q < cost_queries ? sample.size() : 0);
        UIntRowMatrix ids(1, offsets.back());
        for (int q = 0; q < queries.rows(); ++q) {
            std::copy_n(truth.row(q).data(), truth.cols(), ids.data() + offsets[q]);
            if (q < cost_queries)
                std::copy(sample.begin(), sample.end(), ids.data() + offsets[q] + truth.cols());
        }
        const auto kernels =
            benchmark_query_kernels(queries.topRows(cost_queries), query_k, dist);
        const size_t method_bytes = index_bytes() - MLANN::index_bytes();
        std::vector<FrontierConfiguration> best(truth.size() + 1);
        std::vector<float> scores(size_t(depth - min_depth + 1) * ids.size(), 0.f);
        std::vector<double> votes(depth - min_depth + 1, 0.);
        std::vector<float> neighbor_scores(truth.size()),
            candidate_scores(cost_queries * sample.size());
        stream_requested_scores(
            queries, ids, offsets, min_depth, [&](int tree, const float* weights) {
                const float divisor = probability_scores() ? float(tree + 1) : 1.f;
                for (int d = min_depth; d <= depth; ++d) {
                    const size_t base = size_t(d - min_depth) * ids.size();
                    for (size_t j = 0; j < size_t(ids.size()); ++j)
                        scores[base + j] += weights[base + j];
                    for (int q = 0; q < queries.rows(); ++q) {
                        for (int j = 0; j < truth.cols(); ++j)
                            neighbor_scores[size_t(q) * truth.cols() + j] =
                                scores[base + offsets[q] + j] / divisor;
                        if (q < cost_queries)
                            for (size_t j = 0; j < sample.size(); ++j) {
                                const size_t pos = base + offsets[q] + truth.cols() + j;
                                candidate_scores[size_t(q) * sample.size() + j] =
                                    scores[pos] / divisor;
                                votes[d - min_depth] += weights[pos] > 0;
                            }
                    }
                    const size_t bound = tuning_storage_bound({tree + 1, d, 0, 0}, method_bytes);
                    if (memory_budget && bound > memory_budget)
                        continue;
                    update_recall_frontier(
                        best,
                        neighbor_scores,
                        candidate_scores,
                        tree + 1,
                        d,
                        votes[d - min_depth],
                        bound,
                        kernels
                    );
                }
            }
        );
        return compact_recall_frontier(best);
    }

    // Uniform corpus-label sampling estimates work, not calibration recall.
    std::vector<CostEstimate> estimate_costs(
        const Eigen::Ref<const RowMatrix>& queries,
        const std::vector<Calibration>& configs,
        const std::vector<uint32_t>& sample,
        int k,
        Distance dist
    ) const {
        if (queries.rows() == 0 || queries.cols() != dim || !queries.allFinite() ||
            sample.empty() || k < 1 || k > n_corpus)
            throw std::invalid_argument("Invalid cost sample");
        int min_depth = depth;
        for (const auto& c : configs) {
            check_view(c.trees, c.depth);
            if (!(c.threshold > 0) || !std::isfinite(c.threshold))
                throw std::invalid_argument("Invalid cost threshold");
            min_depth = std::min(min_depth, c.depth);
        }
        UIntRowMatrix ids(queries.rows(), sample.size());
        for (size_t j = 0; j < sample.size(); ++j) {
            if (sample[j] >= uint32_t(n_corpus))
                throw std::invalid_argument("Invalid sampled corpus ID");
            ids.col(j).setConstant(sample[j]);
        }

        auto result = estimate_sampled_work(queries, ids, configs, min_depth);
        const auto kernels = benchmark_query_kernels(queries, k, dist);
        for (size_t i = 0; i < configs.size(); ++i)
            result[i].seconds = configs[i].trees * configs[i].depth * kernels.route +
                                result[i].votes * kernels.vote +
                                result[i].candidates * kernels.distance;
        return result;
    }

    std::vector<double> predict_recall(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& truth,
        int trees,
        int d,
        float threshold
    ) const {
        check_view(trees, d);
        if (queries.cols() != dim || queries.rows() == 0 || !queries.allFinite() ||
            truth.rows() != queries.rows() || truth.cols() == 0 ||
            truth.maxCoeff() >= uint32_t(n_corpus) || !std::isfinite(threshold) || threshold <= 0)
            throw std::invalid_argument("Invalid recall prediction inputs");
        std::vector<float> scores(truth.size(), 0.f);
        stream_tuning_scores(
            queries,
            truth,
            d,
            [&](int, const float* weights) {
                for (size_t j = 0; j < scores.size(); ++j)
                    scores[j] += weights[j];
            },
            trees,
            d
        );
        const float divisor = probability_scores() ? float(trees) : 1.f;
        std::vector<double> recalls(queries.rows(), 0.);
        for (int i = 0; i < queries.rows(); ++i) {
            int hits = 0;
            for (int j = 0; j < truth.cols(); ++j)
                hits += scores[size_t(i) * truth.cols() + j] / divisor >= threshold;
            recalls[i] = double(hits) / truth.cols();
        }
        return recalls;
    }

    // Deployed tuning-view storage; corpus, allocator overhead and query scratch excluded.
    virtual size_t index_bytes() const {
        size_t bytes = sizeof(MLANN) + size_t(split_points.size()) * sizeof(float) +
                       size_t(split_dimensions.size()) * sizeof(uint32_t);
        bytes += shared_payloads.capacity() * sizeof(std::shared_ptr<const TuningTreePayload>);
        for (const auto& payload : shared_payloads)
            bytes += payload->bytes;
        return bytes + payload_bytes(labels_all) + payload_bytes(votes_all);
    }

  protected:
    template <class Forest>
    static size_t payload_bytes(const Forest& forest) {
        size_t n = forest.capacity() * sizeof(typename Forest::value_type);
        for (const auto& tree : forest) {
            n += tree.capacity() * sizeof(typename std::decay_t<decltype(tree)>::value_type);
            for (const auto& leaf : tree)
                n += leaf.capacity() * sizeof(typename std::decay_t<decltype(leaf)>::value_type);
        }
        return n;
    }

    bool compact_view_votes(bool unit, std::vector<std::vector<std::vector<uint16_t>>>& compact) {
        if (!shared_payloads.empty())
            return false; // Shared leaves are already stored in their final representation.
        if (!unit) {
            for (const auto& tree : votes_all)
                for (const auto& leaf : tree)
                    for (float weight : leaf)
                        if (weight > std::numeric_limits<uint16_t>::max())
                            return false;
            compact.resize(n_trees);
            for (int t = 0; t < n_trees; ++t) {
                compact[t].resize(n_leaves);
                for (int leaf = 0; leaf < n_leaves; ++leaf)
                    compact[t][leaf].assign(votes_all[t][leaf].begin(), votes_all[t][leaf].end());
            }
        }
        decltype(votes_all)().swap(votes_all);
        return !unit;
    }

    bool retain_membership = false;

    struct TuningLeafPayload {
        std::vector<uint32_t> labels;
        std::vector<uint16_t> compact;
        std::vector<float> weights;
    };

    struct TuningTreePayload {
        std::vector<TuningLeafPayload> leaves;
        size_t bytes = 0;
        bool unit = false, probability = false;
    };

    std::vector<std::shared_ptr<const TuningTreePayload>> shared_payloads;
    mutable std::mutex view_cache_mutex;
    bool cache_views = false;
    mutable int cached_depth = 0;
    mutable size_t payload_trees_built = 0;
    mutable std::vector<std::shared_ptr<const TuningTreePayload>> cached_payloads;
    mutable std::unordered_map<int, std::vector<std::weak_ptr<const TuningTreePayload>>>
        weak_payloads;

    bool accumulate_tuning_votes(
        int tree,
        int leaf,
        float* totals,
        float threshold,
        std::vector<uint32_t>& elected
    ) const {
        if (shared_payloads.empty())
            return false;
        const auto& payload = *shared_payloads[tree];
        const auto& data = payload.leaves[leaf];
        if (payload.unit) {
            mlann_detail::accumulate_unit_votes(data.labels, totals, threshold, elected);
        } else if (payload.probability) {
            for (size_t j = 0; j < data.labels.size(); ++j) {
                float& total = totals[data.labels[j]];
                const float previous = total;
                total += data.weights[j];
                if (total / float(n_trees) >= threshold &&
                    (previous / float(n_trees) < threshold || previous == 0))
                    elected.push_back(data.labels[j]);
            }
        } else if (!data.compact.empty()) {
            mlann_detail::accumulate_neighbor_votes(
                data.labels, data.compact, totals, threshold, elected
            );
        } else {
            mlann_detail::accumulate_neighbor_votes(
                data.labels, data.weights, totals, threshold, elected
            );
        }
        return true;
    }
    bool tuning_structure_only = false;
    bool tuning_unit_labels = false;
    UIntRowMatrix tuning_labels;
    std::vector<std::vector<int>> tuning_permutations;
    std::vector<std::vector<std::pair<int, int>>> tuning_intervals;
    std::vector<std::vector<int>::iterator> tuning_begins;

    virtual bool probability_scores() const { return false; }
    virtual void tuning_path(const float*, int, int*) const {
        throw std::logic_error("Unsupported tuning forest");
    }
    void prepare_tuning(const Eigen::Ref<const UIntRowMatrix>& labels, bool unit, int rows) {
        if (!retain_membership)
            return;
        tuning_unit_labels = unit;
        tuning_labels = labels;
        tuning_permutations.resize(n_trees, std::vector<int>(rows));
        tuning_intervals.resize(
            n_trees, std::vector<std::pair<int, int>>(2 * n_leaves - 1, {-1, -1})
        );
        tuning_begins.resize(n_trees);
    }
    void record_tuning_node(
        int tree,
        int node,
        std::vector<int>::iterator begin,
        std::vector<int>::iterator end
    ) {
        if (!retain_membership)
            return;
        if (node == 0)
            tuning_begins[tree] = begin;
        tuning_intervals[tree][node] = {
            int(begin - tuning_begins[tree]), int(end - tuning_begins[tree])
        };
    }
    void finish_tuning_tree(int tree, const std::vector<int>& rows) {
        if (retain_membership)
            tuning_permutations[tree] = rows;
    }
    void check_view(int trees, int d) const {
        if (!retain_membership || empty() || tuning_intervals.empty())
            throw std::logic_error("A master with retained memberships is required");
        if (trees < 1 || trees > n_trees || d < 1 || d > depth)
            throw std::invalid_argument("Invalid forest prefix or depth");
    }

    struct QueryKernelCosts {
        double route, vote, distance;
    };

    static bool lower_cost(const CostEstimate& a, const CostEstimate& b) {
        return a.seconds < b.seconds ||
               (a.seconds == b.seconds && a.bytes_upper_bound < b.bytes_upper_bound);
    }

    void update_recall_frontier(
        std::vector<FrontierConfiguration>& best,
        std::vector<float>& neighbors,
        std::vector<float>& candidates,
        int trees,
        int d,
        double votes,
        size_t bound,
        const QueryKernelCosts& kernels
    ) const {
        std::sort(neighbors.begin(), neighbors.end(), std::greater<float>());
        std::sort(candidates.begin(), candidates.end(), std::greater<float>());
        const double scale = double(n_corpus) / candidates.size();
        CostEstimate cost;
        cost.votes = votes * scale;
        cost.bytes_upper_bound = bound;
        const double fixed_cost = trees * d * kernels.route + cost.votes * kernels.vote;
        size_t hits = 0, elected = 0;
        // Zero-support forests still contribute a usable best-effort entry.
        do {
            const float threshold =
                hits < neighbors.size() && neighbors[hits] > 0
                    ? neighbors[hits]
                    : (probability_scores() ? std::numeric_limits<float>::min() : 1.f);
            while (hits < neighbors.size() && neighbors[hits] >= threshold)
                ++hits;
            while (elected < candidates.size() && candidates[elected] >= threshold)
                ++elected;
            cost.candidates = elected * scale;
            cost.seconds = fixed_cost + cost.candidates * kernels.distance;
            auto& current = best[hits];
            if (!current.configuration.trees || lower_cost(cost, current.cost))
                current = {{trees, d, threshold, double(hits) / neighbors.size()}, cost};
            if (hits == neighbors.size() || neighbors[hits] <= 0)
                break;
        } while (true);
    }

    static std::vector<FrontierConfiguration> compact_recall_frontier(
        const std::vector<FrontierConfiguration>& best
    ) {
        std::vector<FrontierConfiguration> frontier;
        for (auto it = best.rbegin(); it != best.rend(); ++it)
            if (it->configuration.trees &&
                (frontier.empty() || lower_cost(it->cost, frontier.back().cost)))
                frontier.push_back(*it);
        std::reverse(frontier.begin(), frontier.end());
        return frontier;
    }

    static double elapsed_seconds(std::chrono::steady_clock::time_point started) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    // Conservative owned-storage bound; excludes allocator and query scratch.
    size_t tuning_storage_bound(const Calibration& config, size_t method_bytes) const {
        const size_t leaves = size_t(1) << config.depth;
        const size_t mass =
            size_t(tuning_permutations[0].size()) * (tuning_unit_labels ? 1 : tuning_labels.cols());
        const size_t entries = std::min(mass / size_t(b), leaves * size_t(n_corpus));
        return sizeof(MLANN) +
               config.trees *
                   (sizeof(TuningTreePayload) + sizeof(std::shared_ptr<const TuningTreePayload>) +
                    leaves * sizeof(TuningLeafPayload) + (leaves - 1) * 8 + entries * 8) +
               method_bytes;
    }

    std::vector<CostEstimate> estimate_sampled_work(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& ids,
        const std::vector<Calibration>& configs,
        int min_depth
    ) const {
        const size_t count = ids.size();
        const double scale = double(n_corpus) / count;
        std::vector<float> scores(size_t(depth - min_depth + 1) * count, 0.f);
        std::vector<double> votes(depth - min_depth + 1, 0.);
        std::vector<CostEstimate> result(configs.size());
        const size_t method_bytes = index_bytes() - MLANN::index_bytes();
        stream_tuning_scores(queries, ids, min_depth, [&](int tree, const float* weights) {
            for (int d = min_depth; d <= depth; ++d) {
                const size_t offset = size_t(d - min_depth) * count;
                for (size_t j = 0; j < count; ++j) {
                    scores[offset + j] += weights[offset + j];
                    votes[d - min_depth] += weights[offset + j] > 0;
                }
            }
            const float divisor = probability_scores() ? float(tree + 1) : 1.f;
            for (size_t i = 0; i < configs.size(); ++i) {
                const auto& c = configs[i];
                if (c.trees != tree + 1)
                    continue;
                const auto first = scores.begin() + size_t(c.depth - min_depth) * count;
                result[i].candidates = scale * std::count_if(first, first + count, [&](float s) {
                                           return s / divisor >= c.threshold;
                                       });
                result[i].votes = scale * votes[c.depth - min_depth];
                result[i].bytes_upper_bound = tuning_storage_bound(c, method_bytes);
            }
        });
        return result;
    }

    double benchmark_routing(const Eigen::Ref<const RowMatrix>& queries) const {
        std::vector<int> path(depth + 1);
        volatile int sink = 0;
        const auto before = std::chrono::steady_clock::now();
        for (int q = 0; q < queries.rows(); ++q)
            for (int t = 0; t < n_trees; ++t) {
                tuning_path(queries.row(q).data(), t, path.data());
                sink = path.back();
            }
        const double route_seconds =
            elapsed_seconds(before) / (queries.rows() * double(n_trees) * depth);
        (void) sink;
        return route_seconds;
    }

    void benchmark_vote_updates(
        const std::vector<uint32_t>& labels,
        const std::vector<float>& weights,
        std::vector<float>& totals,
        std::vector<uint32_t>& elected
    ) const {
        if (probability_scores()) {
            // Match RF's unscaled probability accumulation, not the faster
            // SIMD raw-vote kernel. Random ID order matches leaf payloads.
            const float threshold = 0.5f / float(n_trees);
            for (size_t j = 0; j < labels.size(); ++j) {
                float& total = totals[labels[j]];
                const float previous = total;
                total += weights[j];
                if (total / float(n_trees) >= threshold &&
                    (previous / float(n_trees) < threshold || previous == 0))
                    elected.push_back(labels[j]);
            }
        } else if (tuning_unit_labels) {
            mlann_detail::accumulate_unit_votes(labels, totals.data(), 1.f, elected);
        } else {
            mlann_detail::accumulate_neighbor_votes(labels, weights, totals.data(), 1.f, elected);
        }
    }

    QueryKernelCosts benchmark_query_kernels(
        const Eigen::Ref<const RowMatrix>& queries,
        int k,
        Distance dist
    ) const {
        const double route_seconds = benchmark_routing(queries);
        const size_t bench_size = std::min(size_t(n_corpus), size_t(65536));
        std::vector<uint32_t> labels(bench_size);
        for (size_t j = 0; j < bench_size; ++j)
            labels[j] = uint32_t(j * size_t(n_corpus) / bench_size);
        std::mt19937 benchmark_rng(1729);
        std::shuffle(labels.begin(), labels.end(), benchmark_rng);
        std::vector<float> weights(bench_size, 1.f), totals(n_corpus, 0.f);
        std::vector<uint32_t> elected;
        elected.reserve(bench_size);
        std::vector<int> found(k);
        double vote_seconds = std::numeric_limits<double>::infinity();
        double distance_seconds = vote_seconds;
        auto before = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < 4; ++repeat) {
            std::fill(totals.begin(), totals.end(), 0.f);
            elected.clear();
            before = std::chrono::steady_clock::now();
            benchmark_vote_updates(labels, weights, totals, elected);
            double elapsed = elapsed_seconds(before);
            if (repeat)
                vote_seconds = std::min(vote_seconds, elapsed / bench_size);
            before = std::chrono::steady_clock::now();
            exact_knn(
                Eigen::Map<const Eigen::RowVectorXf>(
                    queries.row(repeat % queries.rows()).data(), dim
                ),
                k,
                labels,
                found.data(),
                dist,
                nullptr,
                mlann_detail::compute_neighbor_scores,
                mlann_detail::compute_neighbor_topk
            );
            elapsed = elapsed_seconds(before);
            if (repeat)
                distance_seconds = std::min(distance_seconds, elapsed / bench_size);
        }

        return {route_seconds, vote_seconds, distance_seconds};
    }

    struct TuningPostings {
        // One posting-list key per requested (query, corpus ID) pair.
        std::vector<int> keys;
        std::vector<std::vector<int>> rows;
    };

    TuningPostings make_tuning_postings(const Eigen::Ref<const UIntRowMatrix>& ids) const {
        TuningPostings result;
        result.keys.resize(ids.size());
        std::vector<int> lookup(n_corpus, -1);
        for (Eigen::Index j = 0; j < ids.size(); ++j) {
            int& key = lookup[ids.data()[j]];
            if (key < 0) {
                key = int(result.rows.size());
                result.rows.emplace_back();
            }
            result.keys[j] = key;
        }
        if (tuning_unit_labels) {
            for (int id = 0; id < n_corpus; ++id)
                if (lookup[id] >= 0)
                    result.rows[lookup[id]].push_back(id);
        } else {
            for (int row = 0; row < tuning_labels.rows(); ++row)
                for (int j = 0; j < tuning_labels.cols(); ++j) {
                    const int key = lookup[tuning_labels(row, j)];
                    if (key >= 0)
                        result.rows[key].push_back(row);
                }
        }
        return result;
    }

    // Transform row occurrences into sorted positions in this tree's permutation.
    std::vector<std::vector<int>> tuning_posting_positions(
        int tree,
        const TuningPostings& postings
    ) const {
        const auto& permutation = tuning_permutations[tree];
        std::vector<int> inverse(permutation.size());
        for (size_t pos = 0; pos < inverse.size(); ++pos)
            inverse[permutation[pos]] = int(pos);
        auto positions = postings.rows;
        for (auto& list : positions) {
            for (int& row : list)
                row = inverse[row];
            std::sort(list.begin(), list.end());
        }
        return positions;
    }

    struct TuningMassCache {
        std::unordered_map<int, uint64_t> masses;
        std::vector<uint32_t> counts, touched;

        explicit TuningMassCache(size_t counter_count) : counts(counter_count, 0) {}
    };

    float tuning_node_scale(int tree, int node, TuningMassCache& cache) const {
        if (!probability_scores())
            return 1.f;
        const auto interval = tuning_intervals[tree][node];
        uint64_t mass = uint64_t(interval.second - interval.first) * tuning_labels.cols();
        if (b > 1) {
            auto found = cache.masses.find(node);
            if (found == cache.masses.end()) {
                count_node(tree, node, cache.counts, cache.touched);
                mass = 0;
                for (uint32_t id : cache.touched) {
                    if (cache.counts[id] >= uint32_t(b))
                        mass += cache.counts[id];
                    cache.counts[id] = 0;
                }
                cache.touched.clear();
                found = cache.masses.emplace(node, mass).first;
            }
            mass = found->second;
        }
        return mass ? 1.f / float(mass) : 1.f;
    }

    void score_tuning_tree(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& ids,
        const std::vector<size_t>& offsets,
        const TuningPostings& postings,
        int tree,
        int min_depth,
        int max_depth,
        float* scores
    ) const {
        const auto positions = tuning_posting_positions(tree, postings);
        TuningMassCache mass_cache(probability_scores() && b > 1 ? n_corpus : 0);
        std::vector<int> path(depth + 1);
        for (int q = 0; q < queries.rows(); ++q) {
            tuning_path(queries.row(q).data(), tree, path.data());
            for (int d = min_depth; d <= max_depth; ++d) {
                const auto interval = tuning_intervals[tree][path[d]];
                const float scale = tuning_node_scale(tree, path[d], mass_cache);
                const size_t depth_offset = size_t(d - min_depth) * ids.size();
                for (size_t offset = offsets[q]; offset < offsets[q + 1]; ++offset) {
                    const auto& list = positions[postings.keys[offset]];
                    const auto lo = std::lower_bound(list.begin(), list.end(), interval.first);
                    const auto hi = std::lower_bound(lo, list.end(), interval.second);
                    const auto count = hi - lo;
                    if (count >= b)
                        scores[depth_offset + offset] = float(count) * scale;
                }
            }
        }
    }

    static int tuning_score_batch_size(int tree_count, size_t scores_per_tree) {
        int workers = 1;
#ifdef _OPENMP
        workers = omp_get_max_threads();
#endif
        constexpr size_t budget = 128 * 1024 * 1024;
        return int(std::max(
            size_t(1),
            std::min(
                size_t(std::min(workers, tree_count)), budget / (scores_per_tree * sizeof(float))
            )
        ));
    }

    // Produce tree scores in bounded parallel batches; consume them in tree order
    // so prefix sums have exactly the deployed query's floating-point ordering.
    template <class Consumer>
    void stream_tuning_scores(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& ids,
        int min_depth,
        Consumer consume,
        int last_tree = -1,
        int last_depth = -1
    ) const {
        std::vector<size_t> offsets(queries.rows() + 1);
        for (size_t q = 0; q < offsets.size(); ++q)
            offsets[q] = q * ids.cols();
        stream_requested_scores(queries, ids, offsets, min_depth, consume, last_tree, last_depth);
    }

    // Ragged requests let calibration neighbors and sampled candidates share
    // one traversal without padding every query to the corpus-sample width.
    template <class Consumer>
    void stream_requested_scores(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& ids,
        const std::vector<size_t>& offsets,
        int min_depth,
        Consumer consume,
        int last_tree = -1,
        int last_depth = -1
    ) const {
        const int tree_count = last_tree < 0 ? n_trees : last_tree;
        const int max_depth = last_depth < 0 ? depth : last_depth;
        const auto postings = make_tuning_postings(ids);
        const size_t per_tree = size_t(max_depth - min_depth + 1) * ids.size();
        const int batch = tuning_score_batch_size(tree_count, per_tree);
        for (int first = 0; first < tree_count; first += batch) {
            const int size = std::min(batch, tree_count - first);
            std::vector<float> contributions(size * per_tree, 0.f);
            std::exception_ptr error;
#pragma omp parallel for schedule(static) num_threads(batch)
            for (int slot = 0; slot < size; ++slot) {
                try {
                    score_tuning_tree(
                        queries,
                        ids,
                        offsets,
                        postings,
                        first + slot,
                        min_depth,
                        max_depth,
                        contributions.data() + size_t(slot) * per_tree
                    );
                } catch (...) {
#pragma omp critical(tuning_scores_error)
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }
            if (error)
                std::rethrow_exception(error);
            for (int slot = 0; slot < size; ++slot)
                consume(first + slot, contributions.data() + size_t(slot) * per_tree);
        }
    }

    void count_node(
        int tree,
        int node,
        std::vector<uint32_t>& counts,
        std::vector<uint32_t>& touched
    ) const {
        const auto interval = tuning_intervals[tree][node];
        for (int pos = interval.first; pos < interval.second; ++pos) {
            const int row = tuning_permutations[tree][pos];
            if (tuning_unit_labels) {
                if (counts[row]++ == 0)
                    touched.push_back(uint32_t(row));
            } else {
                const auto* labels = tuning_labels.row(row).data();
                for (int j = 0; j < tuning_labels.cols(); ++j)
                    if (counts[labels[j]]++ == 0)
                        touched.push_back(labels[j]);
            }
        }
    }
    virtual std::pair<int, int> tuning_children(int tree, int node) const {
        const int left = 2 * node + 1;
        if (tuning_intervals[tree][left].first < 0)
            return {-1, -1};
        return {left, left + 1};
    }

    virtual size_t tuning_leaf_slots(int, int d) const { return size_t(1) << d; }

    virtual int tuning_leaf_slot(int node, int level, int d, int) const {
        return (1 << (d - level)) * (node + 1) - (1 << d);
    }

    void fill_tuning_leaf(
        TuningLeafPayload& leaf,
        int tree,
        int node,
        std::vector<uint32_t>& counts,
        std::vector<uint32_t>& touched
    ) const {
        if (tuning_unit_labels) {
            const auto interval = tuning_intervals[tree][node];
            const auto& rows = tuning_permutations[tree];
            leaf.labels.assign(rows.begin() + interval.first, rows.begin() + interval.second);
            return;
        }
        count_node(tree, node, counts, touched);
        uint64_t mass = 0;
        uint32_t maximum = 0;
        size_t retained = 0;
        for (auto id : touched)
            if (counts[id] >= uint32_t(b)) {
                mass += counts[id];
                maximum = std::max(maximum, counts[id]);
                ++retained;
            }
        const bool compact =
            !probability_scores() && maximum <= std::numeric_limits<uint16_t>::max();
        const float scale = probability_scores() && mass ? 1.f / float(mass) : 1.f;
        leaf.labels.reserve(retained);
        if (compact)
            leaf.compact.reserve(retained);
        else
            leaf.weights.reserve(retained);
        for (auto id : touched) {
            if (counts[id] >= uint32_t(b)) {
                leaf.labels.push_back(id);
                if (compact)
                    leaf.compact.push_back(uint16_t(counts[id]));
                else
                    leaf.weights.push_back(float(counts[id]) * scale);
            }
            counts[id] = 0;
        }
        touched.clear();
    }

    void fill_tuning_subtree(
        TuningTreePayload& payload,
        int tree,
        int node,
        int level,
        int d,
        int& cursor,
        std::vector<uint32_t>& counts,
        std::vector<uint32_t>& touched
    ) const {
        const auto children = level == d ? std::make_pair(-1, -1) : tuning_children(tree, node);
        if (children.first < 0) {
            fill_tuning_leaf(
                payload.leaves[tuning_leaf_slot(node, level, d, cursor++)],
                tree,
                node,
                counts,
                touched
            );
        } else {
            fill_tuning_subtree(
                payload, tree, children.first, level + 1, d, cursor, counts, touched
            );
            fill_tuning_subtree(
                payload, tree, children.second, level + 1, d, cursor, counts, touched
            );
        }
    }

    std::shared_ptr<const TuningTreePayload> make_tuning_payload(
        int tree,
        int d,
        std::vector<uint32_t>& counts,
        std::vector<uint32_t>& touched
    ) const {
        auto payload = std::make_shared<TuningTreePayload>();
        payload->unit = tuning_unit_labels;
        payload->probability = probability_scores();
        payload->leaves.resize(tuning_leaf_slots(tree, d));
        int cursor = 0;
        fill_tuning_subtree(*payload, tree, 0, 0, d, cursor, counts, touched);
        payload->bytes =
            sizeof(TuningTreePayload) + payload->leaves.capacity() * sizeof(TuningLeafPayload);
        for (const auto& leaf : payload->leaves)
            payload->bytes += leaf.labels.capacity() * sizeof(uint32_t) +
                              leaf.compact.capacity() * sizeof(uint16_t) +
                              leaf.weights.capacity() * sizeof(float);
        return payload;
    }

    void initialize_shared_payloads(MLANN& view, int trees, int d) const {
        // Serialize cache updates, not queries: published tree payloads are immutable.
        std::unique_lock<std::mutex> lock(view_cache_mutex, std::defer_lock);
        if (cache_views)
            lock.lock();
        view.shared_payloads.resize(trees);
        if (cache_views) {
            if (cached_depth != d) {
                cached_payloads.clear(); // Do not strongly cache every visited depth.
                cached_depth = d;
            }
            auto& known = weak_payloads[d];
            known.resize(std::max(known.size(), size_t(trees)));
            for (int t = 0; t < trees; ++t)
                view.shared_payloads[t] = known[t].lock();
        }
        std::vector<int> missing;
        for (int t = 0; t < trees; ++t)
            if (!view.shared_payloads[t])
                missing.push_back(t);
        std::exception_ptr error;
        if (!missing.empty()) {
            // MSVC OpenMP requires a signed counter; missing.size() is bounded by trees (int).
            const int missing_count = static_cast<int>(missing.size());
#pragma omp parallel
            {
                std::vector<uint32_t> counts(tuning_unit_labels ? 0 : n_corpus, 0), touched;
#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < missing_count; ++i) {
                    try {
                        const int t = missing[i];
                        view.shared_payloads[t] = make_tuning_payload(t, d, counts, touched);
                    } catch (...) {
#pragma omp critical(tuning_payload_error)
                        {
                            if (!error)
                                error = std::current_exception();
                        }
                    }
                }
            }
        }
        if (error)
            std::rethrow_exception(error);
        if (cache_views) {
            payload_trees_built += missing.size();
            cached_payloads.resize(std::max(cached_payloads.size(), size_t(trees)));
            for (int t = 0; t < trees; ++t) {
                cached_payloads[t] = view.shared_payloads[t];
                weak_payloads[d][t] = view.shared_payloads[t];
            }
        }
    }

    void initialize_view(MLANN& view, int trees, int d) const {
        check_view(trees, d);
        view.n_trees = trees;
        view.depth = d;
        view.n_leaves = 1 << d;
        view.n_inner_nodes = view.n_leaves - 1;
        view.n_array = 1 << (d + 1);
        view.n_pool = trees * d;
        view.b = b;
        view.density = density;
        view.split_points = split_points.topLeftCorner(view.n_inner_nodes, trees);
        if (split_dimensions.size())
            view.split_dimensions = split_dimensions.topLeftCorner(view.n_inner_nodes, trees);
        initialize_shared_payloads(view, trees, d);
    }
    using CandidateScoreKernel = void (*)(
        const float*,
        const float*,
        size_t,
        const uint32_t*,
        size_t,
        mlann_detail::OneToManyMetric,
        mlann_detail::StridedFloatOutput
    );

    using ScoredCandidate = mlann_detail::ScoredCandidate;
    using CandidateTopKKernel = void (*)(
        const float*,
        const float*,
        size_t,
        const uint32_t*,
        size_t,
        size_t,
        mlann_detail::OneToManyMetric,
        ScoredCandidate*
    );

    void exact_knn(
        const Eigen::Map<const Eigen::RowVectorXf>& q,
        int k,
        const std::vector<uint32_t>& indices,
        int* out,
        Distance dist = L2,
        float* out_distances = nullptr,
        CandidateScoreKernel score_kernel = nullptr,
        CandidateTopKKernel topk_kernel = nullptr
    ) const {
        exact_knn_impl(
            q.data(),
            corpus.data(),
            dim,
            k,
            indices,
            out,
            dist,
            out_distances,
            score_kernel,
            topk_kernel
        );
    }

    static void exact_knn_impl(
        const float* q_data,
        const float* corpus_data,
        int dim,
        int k,
        const std::vector<uint32_t>& indices,
        int* out,
        Distance dist,
        float* out_distances,
        CandidateScoreKernel score_kernel,
        CandidateTopKKernel topk_kernel
    ) {
        if (k <= 0)
            return;

        if (indices.empty()) {
            for (int i = 0; i < k; ++i)
                out[i] = -1;
            if (out_distances) {
                for (int i = 0; i < k; ++i)
                    out_distances[i] = -1;
            }
            return;
        }

        const int n_elected = static_cast<int>(indices.size());
        const auto metric =
            dist == L2 ? mlann_detail::OneToManyMetric::L2 : mlann_detail::OneToManyMetric::IP;

        if (k == 1) {
            static thread_local Eigen::VectorXf distances;
            distances.resize(n_elected);
            if (score_kernel) {
                const mlann_detail::StridedFloatOutput output{
                    reinterpret_cast<unsigned char*>(distances.data()), sizeof(float)
                };
                score_kernel(q_data, corpus_data, dim, indices.data(), n_elected, metric, output);
            } else {
                mlann_detail::compute_one_to_many(
                    q_data,
                    corpus_data,
                    static_cast<std::size_t>(dim),
                    indices.data(),
                    static_cast<std::size_t>(n_elected),
                    metric,
                    distances.data()
                );
            }
            Eigen::MatrixXf::Index index = 0;
            for (int i = 1; i < n_elected; ++i) {
                const bool better =
                    dist == L2 ? distances[i] < distances[index] : distances[i] > distances[index];
                if (better || (distances[i] == distances[index] && indices[i] < indices[index]))
                    index = i;
            }

            if (dist == L2) {
                out[0] = indices[index];
                if (out_distances)
                    out_distances[0] = std::sqrt(distances(index));
            } else {
                out[0] = indices[index];
                if (out_distances)
                    out_distances[0] = distances(index);
            }

            return;
        }

        int n_to_sort = n_elected > k ? k : n_elected;
        static thread_local std::vector<ScoredCandidate> scored;
        scored.resize(topk_kernel ? n_to_sort : n_elected);
        if (topk_kernel) {
            topk_kernel(
                q_data,
                corpus_data,
                dim,
                indices.data(),
                n_elected,
                n_to_sort,
                metric,
                scored.data()
            );
        } else {
            for (int i = 0; i < n_elected; ++i)
                scored[i].label = indices[i];
            const mlann_detail::StridedFloatOutput scores{
                reinterpret_cast<unsigned char*>(scored.data()), sizeof(ScoredCandidate)
            };
            if (score_kernel) {
                score_kernel(q_data, corpus_data, dim, indices.data(), n_elected, metric, scores);
            } else {
                mlann_detail::compute_one_to_many(
                    q_data,
                    corpus_data,
                    static_cast<std::size_t>(dim),
                    indices.data(),
                    static_cast<std::size_t>(n_elected),
                    metric,
                    scores
                );
            }

            if (dist == L2) {
                miniselect::pdqpartial_sort_branchless(
                    scored.data(),
                    scored.data() + n_to_sort,
                    scored.data() + n_elected,
                    [](const ScoredCandidate& left, const ScoredCandidate& right) {
                        return left.score < right.score ||
                               (left.score == right.score && left.label < right.label);
                    }
                );
            } else {
                miniselect::pdqpartial_sort_branchless(
                    scored.data(),
                    scored.data() + n_to_sort,
                    scored.data() + n_elected,
                    [](const ScoredCandidate& left, const ScoredCandidate& right) {
                        return left.score > right.score ||
                               (left.score == right.score && left.label < right.label);
                    }
                );
            }
        }

        for (int i = 0; i < k; ++i) {
            out[i] = i < n_elected ? static_cast<int>(scored[i].label) : -1;
        }

        if (out_distances) {
            if (dist == L2) {
                for (int i = 0; i < k; ++i) {
                    out_distances[i] = i < n_elected ? std::sqrt(scored[i].score) : -1;
                }
            } else {
                for (int i = 0; i < k; ++i) {
                    out_distances[i] = i < n_elected ? scored[i].score : -1;
                }
            }
        }
    }

    const Eigen::Map<const RowMatrix> corpus;
    Eigen::MatrixXf split_points;
    Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> split_dimensions;
    std::vector<std::vector<std::vector<uint32_t>>> labels_all;
    std::vector<std::vector<std::vector<float>>> votes_all;

    const int n_corpus;
    const int dim;
    int n_trees = 0;
    int depth = 0;
    float density = -1.0; // Expected fraction of nonzero components in a projection matrix.
    int n_pool = 0;       // Projection vectors across all trees.
    int n_array = 0;      // Nodes per tree in the flat representation.
    int b = 0;
    int n_inner_nodes = 0;
    int n_leaves = 0;
};
