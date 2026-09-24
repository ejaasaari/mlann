#pragma once

#include <Eigen/Dense>
#include <array>
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
#include <unordered_set>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "detail/distance.h"
#include "detail/huge-buffer.h"
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
        throw std::invalid_argument("Unsupervised builds are supported only by KD, PCA and RP.");
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
        throw std::invalid_argument("Autotuning supports KD, RP, PCA, RF and CRAFTML only");
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
        double target,
        float fixed_threshold = 0
    ) const {
        check_view(n_trees, min_depth);
        if (queries.cols() != dim || queries.rows() == 0 || !queries.allFinite() ||
            truth.rows() != queries.rows() || truth.cols() == 0 ||
            truth.maxCoeff() >= uint32_t(n_corpus) || !(target > 0 && target <= 1))
            throw std::invalid_argument("Invalid calibration data or recall target");
        if (!std::isfinite(fixed_threshold) || fixed_threshold < 0)
            throw std::invalid_argument("Invalid fixed vote threshold");
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
                float threshold = fixed_threshold;
                if (threshold == 0) {
                    auto scratch = score; // Never partition the live accumulator.
                    std::nth_element(
                        scratch.begin(),
                        scratch.begin() + h - 1,
                        scratch.end(),
                        std::greater<float>()
                    );
                    threshold = scratch[h - 1] / divisor;
                }
                if (threshold <= 0)
                    continue;
                size_t hits = 0;
                for (float s : score)
                    hits += s / divisor >= threshold;
                if (hits >= h)
                    result.push_back({tree + 1, d, threshold, double(hits) / count});
            }
        });
        return result;
    }

    struct CostEstimate {
        double votes = 0, candidates = 0, seconds = 0;
        size_t bytes_upper_bound = 0;
        double touched = 0, elected_votes = 0;
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
        float fixed_threshold = 0
    ) const {
        check_view(n_trees, min_depth);
        if (!std::isfinite(fixed_threshold) || fixed_threshold < 0)
            throw std::invalid_argument("Invalid fixed vote threshold");
        if (queries.rows() == 0 || queries.cols() != dim || !queries.allFinite() ||
            truth.rows() != queries.rows() || truth.cols() == 0 || truth.cols() > n_corpus ||
            truth.maxCoeff() >= uint32_t(n_corpus) || sample.empty() || cost_queries < 1 ||
            cost_queries > queries.rows() || (dist != L2 && dist != IP))
            throw std::invalid_argument("Invalid frontier calibration data");
        for (auto id : sample)
            if (id >= uint32_t(n_corpus))
                throw std::invalid_argument("Invalid sampled corpus ID");

        size_t population = n_corpus;
        const auto reachable_sample = sample_cost_ids(sample, dist, population);
        const auto& cost_sample = reachable_sample.empty() ? sample : reachable_sample;
        std::vector<size_t> offsets(queries.rows() + 1, 0);
        for (int q = 0; q < queries.rows(); ++q)
            offsets[q + 1] =
                offsets[q] + truth.cols() + (q < cost_queries ? cost_sample.size() : 0);
        UIntRowMatrix ids(1, offsets.back());
        for (int q = 0; q < queries.rows(); ++q) {
            std::copy_n(truth.row(q).data(), truth.cols(), ids.data() + offsets[q]);
            if (q < cost_queries)
                std::copy(
                    cost_sample.begin(), cost_sample.end(), ids.data() + offsets[q] + truth.cols()
                );
        }
        const auto kernels =
            benchmark_query_kernels(queries.topRows(cost_queries), int(truth.cols()), dist);
        const size_t method_bytes = index_bytes() - MLANN::index_bytes();
        std::vector<FrontierConfiguration> best(truth.size() + 1);
        std::vector<float> scores(size_t(depth - min_depth + 1) * ids.size(), 0.f);
        const bool track_support = kernels.weighted_mixed_votes;
        std::vector<uint32_t> support(track_support ? scores.size() : 0, 0);
        std::vector<uint32_t> candidate_support(
            track_support ? cost_queries * cost_sample.size() : 0
        );
        std::vector<double> votes(depth - min_depth + 1, 0.);
        std::vector<float> neighbor_scores(truth.size()),
            candidate_scores(cost_queries * cost_sample.size());
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
                            for (size_t j = 0; j < cost_sample.size(); ++j) {
                                const size_t pos = base + offsets[q] + truth.cols() + j;
                                candidate_scores[size_t(q) * cost_sample.size() + j] =
                                    scores[pos] / divisor;
                                votes[d - min_depth] += weights[pos] > 0;
                                if (track_support) {
                                    support[pos] += weights[pos] > 0;
                                    candidate_support[size_t(q) * cost_sample.size() + j] =
                                        support[pos];
                                }
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
                        kernels,
                        fixed_threshold,
                        track_support ? &candidate_support : nullptr,
                        population
                    );
                }
            }
        );
        return compact_recall_frontier(best);
    }

    // Uniform label sampling estimates work, not calibration recall. For a small
    // reachable IP universe, omit known-zero IDs and scale by that universe.
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
        for (auto id : sample)
            if (id >= uint32_t(n_corpus))
                throw std::invalid_argument("Invalid sampled corpus ID");
        size_t population = n_corpus;
        const auto reachable_sample = sample_cost_ids(sample, dist, population);
        const auto& cost_sample = reachable_sample.empty() ? sample : reachable_sample;
        UIntRowMatrix ids(queries.rows(), cost_sample.size());
        for (size_t j = 0; j < cost_sample.size(); ++j)
            ids.col(j).setConstant(cost_sample[j]);

        auto result = estimate_sampled_work(
            queries,
            ids,
            configs,
            min_depth,
            probability_scores() && !sparse_tuning_votes(),
            population
        );
        const auto kernels = benchmark_query_kernels(queries, k, dist);
        for (size_t i = 0; i < configs.size(); ++i)
            result[i].seconds = kernels.seconds(
                configs[i].trees,
                configs[i].depth,
                result[i].votes,
                result[i].candidates,
                result[i].touched,
                result[i].elected_votes
            );
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
        accumulate_tuning_leaf(data, payload.unit, payload.probability, totals, threshold, elected);
        return true;
    }

    void accumulate_tuning_leaf(
        const TuningLeafPayload& data,
        bool unit,
        bool probability,
        float* totals,
        float threshold,
        std::vector<uint32_t>& elected
    ) const {
        if (unit) {
            mlann_detail::accumulate_unit_votes(data.labels, totals, threshold, elected);
        } else if (probability) {
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
    }
    bool tuning_structure_only = false;
    bool tuning_unit_labels = false;
    UIntRowMatrix tuning_labels;
    std::vector<std::vector<int>> tuning_permutations;
    std::vector<std::vector<std::pair<int, int>>> tuning_intervals;
    std::vector<std::vector<int>::iterator> tuning_begins;

    virtual bool probability_scores() const { return false; }
    virtual float tuning_node_score(uint32_t count, float scale, int) const {
        return float(count) * scale;
    }
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
        double setup = 0, election = 0;
        double touch = 0, sort = 0;
        std::vector<double> routing_by_depth;
        double mixed_votes = 0;
        bool weighted_mixed_votes = false;
        struct LeafCosts {
            int trees;
            double vote, election, touch, mixed;
            double seconds(double votes, double candidates, double touched, double fraction) const {
                return votes * vote + candidates * election + touched * touch +
                       mixed * votes * fraction * (1. - fraction);
            }
        };
        std::vector<LeafCosts> leaf_by_trees;
        std::vector<std::pair<double, double>> distance_by_candidates;
        double rerank_seconds(double candidates) const {
            if (distance_by_candidates.empty())
                return candidates * distance;
            auto hi = std::lower_bound(
                distance_by_candidates.begin(),
                distance_by_candidates.end(),
                candidates,
                [](const auto& point, double n) { return point.first < n; }
            );
            if (hi == distance_by_candidates.begin())
                return candidates * hi->second;
            if (hi == distance_by_candidates.end())
                return candidates * distance_by_candidates.back().second;
            const auto lo = hi - 1;
            const double weight =
                std::log(candidates / lo->first) / std::log(hi->first / lo->first);
            return candidates * (lo->second + weight * (hi->second - lo->second));
        }

        double leaf_seconds(
            int trees,
            double votes,
            double candidates,
            double touched,
            double elected_votes
        ) const {
            const double fraction = weighted_mixed_votes ? elected_votes / std::max(1., votes)
                                                         : candidates / std::max(1., touched);
            double leaf = LeafCosts{trees, vote, election, touch, mixed_votes}.seconds(
                votes, candidates, touched, fraction
            );
            if (!leaf_by_trees.empty()) {
                auto hi = std::lower_bound(
                    leaf_by_trees.begin(),
                    leaf_by_trees.end(),
                    trees,
                    [](const auto& point, int n) { return point.trees < n; }
                );
                if (hi == leaf_by_trees.end())
                    leaf = leaf_by_trees.back().seconds(votes, candidates, touched, fraction);
                else if (hi == leaf_by_trees.begin())
                    leaf = hi->seconds(votes, candidates, touched, fraction);
                else {
                    const auto lo = hi - 1;
                    const double weight = std::log(double(trees) / lo->trees) /
                                          std::log(double(hi->trees) / lo->trees);
                    leaf = (1. - weight) * lo->seconds(votes, candidates, touched, fraction) +
                           weight * hi->seconds(votes, candidates, touched, fraction);
                }
            }
            return leaf;
        }

        double seconds(
            int trees,
            int depth,
            double votes,
            double candidates,
            double touched = 0,
            double elected_votes = 0
        ) const {
            const double leaf = leaf_seconds(trees, votes, candidates, touched, elected_votes);
            const double routing =
                routing_by_depth.empty() ? depth * route : routing_by_depth[depth];
            return setup + trees * routing + leaf + rerank_seconds(candidates) +
                   sort * candidates * std::log2(std::max(1., candidates));
        }
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
        const QueryKernelCosts& kernels,
        float fixed_threshold,
        const std::vector<uint32_t>* support = nullptr,
        size_t population = 0
    ) const {
        std::sort(neighbors.begin(), neighbors.end(), std::greater<float>());
        std::vector<std::pair<float, uint32_t>> ranked;
        if (support) {
            ranked.reserve(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i)
                ranked.emplace_back(candidates[i], (*support)[i]);
            std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            for (size_t i = 0; i < candidates.size(); ++i)
                candidates[i] = ranked[i].first;
        } else {
            std::sort(candidates.begin(), candidates.end(), std::greater<float>());
        }
        const double scale = double(population ? population : n_corpus) / candidates.size();
        CostEstimate cost;
        cost.votes = votes * scale;
        cost.touched = scale * std::count_if(candidates.begin(), candidates.end(), [](float s) {
                           return s > 0;
                       });
        cost.bytes_upper_bound = bound;
        size_t hits = 0, elected = 0;
        // Zero-support forests still contribute a usable best-effort entry.
        do {
            const float threshold =
                fixed_threshold > 0 ? fixed_threshold
                : hits < neighbors.size() && neighbors[hits] > 0
                    ? neighbors[hits]
                    : (probability_scores() ? std::numeric_limits<float>::min() : 1.f);
            while (hits < neighbors.size() && neighbors[hits] >= threshold)
                ++hits;
            while (elected < candidates.size() && candidates[elected] >= threshold) {
                if (support)
                    cost.elected_votes += ranked[elected].second * scale;
                ++elected;
            }
            cost.candidates = elected * scale;
            cost.seconds = kernels.seconds(
                trees, d, cost.votes, cost.candidates, cost.touched, cost.elected_votes
            );
            auto& current = best[hits];
            if (!current.configuration.trees || lower_cost(cost, current.cost))
                current = {{trees, d, threshold, double(hits) / neighbors.size()}, cost};
            if (fixed_threshold > 0 || hits == neighbors.size() || neighbors[hits] <= 0)
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
    virtual size_t tuning_storage_bound(const Calibration& config, size_t method_bytes) const {
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
        int min_depth,
        bool track_support = false,
        size_t population = 0
    ) const {
        const size_t count = ids.size();
        const double scale = double(population ? population : n_corpus) / count;
        std::vector<float> scores(size_t(depth - min_depth + 1) * count, 0.f);
        std::vector<uint32_t> support(track_support ? scores.size() : 0, 0);
        std::vector<double> votes(depth - min_depth + 1, 0.);
        std::vector<CostEstimate> result(configs.size());
        const size_t method_bytes = index_bytes() - MLANN::index_bytes();
        stream_tuning_scores(queries, ids, min_depth, [&](int tree, const float* weights) {
            for (int d = min_depth; d <= depth; ++d) {
                const size_t offset = size_t(d - min_depth) * count;
                for (size_t j = 0; j < count; ++j) {
                    scores[offset + j] += weights[offset + j];
                    if (track_support)
                        support[offset + j] += weights[offset + j] > 0;
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
                result[i].touched =
                    scale * std::count_if(first, first + count, [](float s) { return s > 0; });
                if (track_support) {
                    double elected_votes = 0;
                    const size_t offset = size_t(c.depth - min_depth) * count;
                    for (size_t j = 0; j < count; ++j)
                        if (scores[offset + j] / divisor >= c.threshold)
                            elected_votes += support[offset + j];
                    result[i].elected_votes = elected_votes * scale;
                }
                result[i].votes = scale * votes[c.depth - min_depth];
                result[i].bytes_upper_bound = tuning_storage_bound(c, method_bytes);
            }
        });
        return result;
    }

    virtual bool tuning_compact_votes() const { return false; }
    virtual bool sparse_tuning_votes() const { return false; }

    struct SampledTuningLeaf {
        int tree, node;
        TuningLeafPayload payload;
    };
    struct SampledLeafWork {
        int query, depth;
        std::vector<size_t> leaves;
    };
    struct LeafWorkSample {
        std::vector<SampledTuningLeaf> leaves;
        std::vector<SampledLeafWork> work;
        size_t label_visits = 0;
    };
    static constexpr size_t leaf_work_budget = 16 << 20;
    static constexpr size_t candidate_pool_scan_budget = 4 << 20;

    LeafWorkSample sample_leaf_work(const Eigen::Ref<const RowMatrix>& queries) const {
        // Reconstruct only reached nodes, never complete candidate forests. Bound
        // logical input-label visits as well as construction, including cache hits.
        LeafWorkSample sample;
        std::unordered_map<uint64_t, size_t> cached;
        std::vector<uint32_t> counts(tuning_unit_labels ? 0 : n_corpus, 0), touched;
        const int query_count = int(std::min<Eigen::Index>(4, queries.rows()));
        const int tree_count = std::min(32, n_trees);
        const size_t query_budget = leaf_work_budget / query_count;
        const size_t labels_per_row = tuning_unit_labels ? 1 : tuning_labels.cols();
        for (int qi = 0; qi < query_count; ++qi) {
            const int q = int(qi * queries.rows() / query_count);
            std::vector<std::vector<int>> paths(tree_count, std::vector<int>(depth + 1));
            for (int t = 0; t < tree_count; ++t)
                tuning_path(queries.row(q).data(), t * n_trees / tree_count, paths[t].data());
            // Paths repeat terminal nodes. Sample depths within the reached
            // path so the probes include both terminal and internal workloads.
            int probe_depth = 1;
            for (const auto& path : paths) {
                int reached = 0;
                while (reached < depth && path[reached + 1] != path[reached])
                    ++reached;
                probe_depth = std::max(probe_depth, reached);
            }
            const std::array<int, 3> levels{
                probe_depth, std::max(1, 3 * probe_depth / 4), std::max(1, probe_depth / 2)
            };
            size_t used = 0;
            int previous_depth = -1;
            for (int d : levels) {
                if (d == previous_depth)
                    continue;
                previous_depth = d;
                SampledLeafWork work{q, d, {}};
                for (int t = 0; t < tree_count; ++t) {
                    const int tree = t * n_trees / tree_count, node = paths[t][d];
                    const auto interval = tuning_intervals[tree][node];
                    const size_t visits = size_t(interval.second - interval.first) * labels_per_row;
                    if (visits > query_budget - used)
                        continue;
                    const uint64_t key = (uint64_t(tree) << 32) | uint32_t(node);
                    auto found = cached.find(key);
                    if (found == cached.end()) {
                        const size_t index = sample.leaves.size();
                        sample.leaves.push_back({tree, node, {}});
                        fill_tuning_leaf(sample.leaves.back().payload, tree, node, counts, touched);
                        found = cached.emplace(key, index).first;
                    }
                    work.leaves.push_back(found->second);
                    used += visits;
                }
                if (!work.leaves.empty())
                    sample.work.push_back(std::move(work));
            }
            sample.label_visits += used;
        }
        return sample;
    }

    struct LeafCostObservation {
        double votes, touched, candidates, seconds;
        double elected_votes = 0, setup_seconds = 0;
    };

    virtual double benchmark_leaf_probe(
        const LeafWorkSample& sample,
        const SampledLeafWork& work,
        float threshold,
        float* totals,
        std::vector<uint32_t>& elected,
        size_t& candidates,
        double& setup_seconds
    ) const {
        const auto setup_started = std::chrono::steady_clock::now();
        std::fill_n(totals, n_corpus, 0.f);
        setup_seconds = elapsed_seconds(setup_started);
        elected.clear();
        const auto before = std::chrono::steady_clock::now();
        for (size_t index : work.leaves)
            accumulate_tuning_leaf(
                sample.leaves[index].payload,
                tuning_unit_labels,
                probability_scores(),
                totals,
                threshold,
                elected
            );
        const double elapsed = elapsed_seconds(before);
        candidates = elected.size();
        return elapsed;
    }

    static Eigen::Vector3d solve_nonnegative_leaf_costs(
        const Eigen::Matrix3d& normal,
        const Eigen::Vector3d& rhs,
        int dimensions
    ) {
        Eigen::Vector3d best = Eigen::Vector3d::Zero();
        double error = std::numeric_limits<double>::infinity();
        // At most three coefficients: enumerate nonnegative least-squares faces.
        for (int mask = 0; mask < (1 << dimensions); ++mask) {
            Eigen::Matrix3d a = normal;
            Eigen::Vector3d b = rhs;
            for (int j = 0; j < 3; ++j)
                if (!(mask & (1 << j))) {
                    a.row(j).setZero();
                    a.col(j).setZero();
                    a(j, j) = 1.;
                    b[j] = 0.;
                }
            const Eigen::Vector3d fit = a.ldlt().solve(b);
            if (!fit.allFinite() || fit.minCoeff() < 0)
                continue;
            const double loss = fit.dot(normal * fit) - 2 * fit.dot(rhs);
            if (loss < error) {
                error = loss;
                best = fit;
            }
        }
        return best;
    }

    void fit_leaf_costs(
        QueryKernelCosts& costs,
        const std::vector<LeafCostObservation>& data,
        bool anchor_zero_election = false
    ) const {
        if (data.empty())
            return; // Pruned/oversized leaves retain the bounded kernel fallback.
        const bool sparse = sparse_tuning_votes();
        const int dimensions = probability_scores() ? 3 : 2;
        Eigen::Vector3d prior(
            costs.vote * 1e9, costs.election * 1e9, (sparse ? costs.touch : costs.mixed_votes) * 1e9
        );
        double zero_election_vote = 0;
        size_t zero_election_samples = 0;
        if (anchor_zero_election && probability_scores() && !sparse) {
            std::vector<double> rates;
            for (const auto& row : data)
                if (row.candidates == 0)
                    rates.push_back(row.seconds * 1e9 / row.votes);
            zero_election_samples = rates.size();
            if (zero_election_samples) {
                // Use the median of probe rates, as with dense scratch clearing.
                // One cold or tiny reached leaf must not price every no-election
                // case at its unusually high per-update cost.
                std::sort(rates.begin(), rates.end());
                zero_election_vote = .5 * (rates[(rates.size() - 1) / 2] + rates[rates.size() / 2]);
                prior[0] = 0;
            }
        }
        // Relative work weighting prevents the largest leaf from dominating.
        // A weak prior stabilizes correlated update/touch counts and tiny samples.
        Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
        Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
        for (const auto& row : data) {
            const double fraction = costs.weighted_mixed_votes ? row.elected_votes / row.votes
                                                               : row.candidates / row.touched;
            // RF's conditional threshold crossings are most expensive when
            // outcomes are mixed. Keep this separate from the zero-election cost.
            Eigen::Vector3d x(
                zero_election_samples ? 0. : 1.,
                row.candidates / row.votes,
                sparse            ? row.touched / row.votes
                : dimensions == 3 ? fraction * (1. - fraction)
                                  : 0.
            );
            normal.noalias() += x * x.transpose();
            rhs.noalias() += x * (row.seconds * 1e9 / row.votes - zero_election_vote);
        }
        for (int j = 0; j < 3; ++j) {
            const double ridge = .1 * std::max(normal(j, j), 1e-6 * data.size());
            normal(j, j) += ridge;
            rhs[j] += ridge * prior[j];
        }
        const auto best = solve_nonnegative_leaf_costs(normal, rhs, dimensions);
        // RF's mixed threshold outcomes must not inflate the directly measured
        // no-election cost, which dominates high-threshold configurations.
        costs.vote = (zero_election_samples ? zero_election_vote : best[0]) * 1e-9;
        costs.election = best[1] * 1e-9;
        if (sparse)
            costs.touch = best[2] * 1e-9;
        else if (dimensions == 3)
            costs.mixed_votes = best[2] * 1e-9;
    }

    struct LeafCostProbe {
        SampledLeafWork work;
        size_t votes, touched;
        std::array<float, 4> thresholds;
        std::array<double, 4> elected_votes;
    };

    struct LeafProbeScratch {
        mlann_detail::HugeBuffer<float> totals;
        std::vector<uint32_t> support, touched, elected;
        std::vector<float> scores;

        LeafProbeScratch(int corpus_size, bool track_support)
            : support(track_support ? corpus_size : 0, 0) {
            totals.resize(corpus_size);
        }
    };

    std::vector<LeafCostProbe> prepare_leaf_probes(
        const LeafWorkSample& sample,
        int prefix,
        LeafProbeScratch& scratch
    ) const {
        auto& totals = scratch.totals;
        auto& support = scratch.support;
        auto& touched = scratch.touched;
        auto& elected = scratch.elected;
        auto& scores = scratch.scores;
        // Dense timing probes leave votes behind. Each prefix needs an empty
        // accumulator so its touched-ID list includes every reached label.
        std::fill_n(totals.data(), n_corpus, 0.f);
        std::fill(support.begin(), support.end(), 0);
        std::vector<LeafCostProbe> probes;
        const float divisor = probability_scores() ? float(n_trees) : 1.f;
        for (auto work : sample.work) {
            work.leaves.resize(std::min(size_t(prefix), work.leaves.size()));
            size_t votes = 0;
            // Exact support and real score quantiles determine threshold probes.
            for (size_t index : work.leaves) {
                const auto& leaf = sample.leaves[index].payload;
                votes += leaf.labels.size();
                for (size_t j = 0; j < leaf.labels.size(); ++j) {
                    const uint32_t id = leaf.labels[j];
                    if (!support.empty())
                        ++support[id];
                    if (totals[id] == 0.f)
                        touched.push_back(id);
                    totals[id] += tuning_unit_labels      ? 1.f
                                  : !leaf.compact.empty() ? leaf.compact[j]
                                                          : leaf.weights[j];
                }
            }
            scores.clear();
            for (uint32_t id : touched)
                scores.push_back(totals[id] / divisor);
            auto reset = [&] {
                for (uint32_t id : touched) {
                    totals[id] = 0.f;
                    if (!support.empty())
                        support[id] = 0;
                }
                touched.clear();
            };
            if (votes < 128) {
                reset();
                continue; // Timer overhead would dominate; retain the prior.
            }
            std::sort(scores.begin(), scores.end());
            const std::array<float, 4> thresholds{
                scores.front(),
                scores[scores.size() / 2],
                scores[3 * scores.size() / 4],
                std::nextafter(scores.back(), std::numeric_limits<float>::infinity())
            };
            elected.reserve(scores.size());
            touched.reserve(scores.size());
            std::array<double, 4> elected_votes{};
            if (!support.empty())
                for (uint32_t id : touched)
                    for (int j = 0; j < 4; ++j)
                        if (totals[id] / divisor >= thresholds[j])
                            elected_votes[j] += support[id];
            reset();
            probes.push_back({std::move(work), votes, scores.size(), thresholds, elected_votes});
        }
        return probes;
    }

    std::vector<LeafCostObservation> measure_leaf_probes(
        const LeafWorkSample& sample,
        const std::vector<LeafCostProbe>& probes,
        LeafProbeScratch& scratch
    ) const {
        auto& totals = scratch.totals;
        auto& elected = scratch.elected;
        std::vector<LeafCostObservation> observations(probes.size() * 4);
        for (auto& observation : observations) {
            observation.seconds = std::numeric_limits<double>::infinity();
            observation.setup_seconds = std::numeric_limits<double>::infinity();
        }
        // Cycle through reached leaves between repeats, within one prefix
        // size. Large prefixes must not evict every small-prefix payload.
        for (int repeat = 0; repeat < 4; ++repeat) {
            for (int threshold_index = 0; threshold_index < 4; ++threshold_index) {
                for (size_t i = 0; i < probes.size(); ++i) {
                    const auto& probe = probes[i];
                    const float threshold = probe.thresholds[threshold_index];
                    size_t candidates = 0;
                    double setup_seconds = 0;
                    const double seconds = benchmark_leaf_probe(
                        sample,
                        probe.work,
                        threshold,
                        totals.data(),
                        elected,
                        candidates,
                        setup_seconds
                    );
                    auto& observation = observations[threshold_index * probes.size() + i];
                    if (repeat)
                        observation = {
                            double(probe.votes),
                            double(probe.touched),
                            double(candidates),
                            std::min(observation.seconds, seconds),
                            probe.elected_votes[threshold_index],
                            std::min(observation.setup_seconds, setup_seconds)
                        };
                }
            }
        }
        return observations;
    }

    bool use_prefix_leaf_costs(Distance dist) const {
        // Working sets change with forest size. Concentrated sparse IP support
        // uses a pooled fit because tiny prefixes keep its payloads cache-resident.
        if (!probability_scores())
            return false;
        if (dist == L2)
            return true;
        return sparse_tuning_votes() ? bounded_candidate_pool().empty()
                                     : !small_ip_candidate_pool(dist).empty();
    }

    void calibrate_leaf_costs(
        const Eigen::Ref<const RowMatrix>& queries,
        QueryKernelCosts& costs,
        Distance dist
    ) const {
        costs.weighted_mixed_votes = probability_scores() && !sparse_tuning_votes();
        const auto sample = sample_leaf_work(queries);
        LeafProbeScratch scratch(n_corpus, costs.weighted_mixed_votes);
        std::vector<double> setup_samples;
        const QueryKernelCosts prior = costs;
        std::vector<int> prefixes{std::min(32, n_trees)};
        const bool fit_prefixes = use_prefix_leaf_costs(dist);
        if (fit_prefixes) {
            prefixes = {1, std::min(8, n_trees), std::min(32, n_trees)};
            prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
        }
        for (int prefix : prefixes) {
            const auto probes = prepare_leaf_probes(sample, prefix, scratch);
            const auto observations = measure_leaf_probes(sample, probes, scratch);
            QueryKernelCosts fitted = prior;
            fit_leaf_costs(fitted, observations, true);
            if (!sparse_tuning_votes())
                for (const auto& observation : observations)
                    setup_samples.push_back(observation.setup_seconds);
            if (fit_prefixes && !observations.empty())
                costs.leaf_by_trees.push_back(
                    {prefix, fitted.vote, fitted.election, fitted.touch, fitted.mixed_votes}
                );
            costs.vote = fitted.vote;
            costs.election = fitted.election;
            costs.touch = fitted.touch;
            costs.mixed_votes = fitted.mixed_votes;
        }
        if (!setup_samples.empty()) {
            // Clearing after real voting work is more representative than
            // clearing immediately after a large synthetic rerank. Use the
            // median of per-probe minima to resist occasional timer noise.
            auto middle = setup_samples.begin() + setup_samples.size() / 2;
            std::nth_element(setup_samples.begin(), middle, setup_samples.end());
            costs.setup = *middle;
        }
    }

    std::vector<uint32_t> small_ip_candidate_pool(Distance dist) const {
        // Sparse CraftML keeps corpus sampling and sorted rerank calibration.
        if (dist != IP || sparse_tuning_votes())
            return {};
        return bounded_candidate_pool();
    }

    std::vector<uint32_t> bounded_candidate_pool() const {
        // A supervised forest can only elect IDs present in its training labels.
        // Raw inner-product data can concentrate these labels in a small, reusable
        // working set. Uniform corpus probes then time mostly unreachable vectors.
        // Only replace the probe when the complete reachable set fits its existing
        // 65,536-ID limit. Bound the discovery scan and retain the fallback otherwise.
        if (tuning_unit_labels || !tuning_labels.size() ||
            size_t(tuning_labels.size()) > candidate_pool_scan_budget)
            return {};
        const size_t limit = std::min(size_t(n_corpus), size_t(65536));
        std::unordered_set<uint32_t> seen;
        seen.reserve(std::min(limit, size_t(tuning_labels.size())));
        for (Eigen::Index j = 0; j < tuning_labels.size(); ++j) {
            seen.insert(tuning_labels.data()[j]);
            if (seen.size() >= limit)
                return {};
        }
        std::vector<uint32_t> labels(seen.begin(), seen.end());
        std::sort(labels.begin(), labels.end());
        return labels;
    }

    std::vector<uint32_t> sample_cost_ids(
        const std::vector<uint32_t>& sample,
        Distance dist,
        size_t& population
    ) const {
        auto reachable = small_ip_candidate_pool(dist);
        population = reachable.empty() ? size_t(n_corpus) : reachable.size();
        if (reachable.size() > sample.size()) {
            // Preserve caller-controlled randomization without increasing sample
            // width. Sampling is uniform without replacement in the reachable set.
            std::seed_seq seed(sample.begin(), sample.end());
            std::mt19937 generator(seed);
            std::shuffle(reachable.begin(), reachable.end(), generator);
            reachable.resize(sample.size());
        }
        return reachable;
    }

    void calibrate_rerank_costs(
        const Eigen::Ref<const RowMatrix>& queries,
        int k,
        Distance dist,
        bool sorted,
        QueryKernelCosts& costs,
        const std::vector<uint32_t>& pool = {}
    ) const {
        // Small candidate batches have different cache and top-k costs. Cycle
        // through query-specific samples before repeating a batch, rather than
        // timing one tiny, permanently hot candidate list.
        const int query_count = int(std::min<Eigen::Index>(64, queries.rows()));
        const size_t population = pool.empty() ? size_t(n_corpus) : pool.size();
        const size_t anchor = std::min(population, size_t(65536));
        std::mt19937 generator(937);
        std::vector<int> found(k);
        for (size_t count : {size_t(16), size_t(64), size_t(256), size_t(1024)}) {
            if (count >= anchor)
                break;
            std::vector<std::vector<uint32_t>> batches(query_count, std::vector<uint32_t>(count));
            for (auto& labels : batches) {
                const size_t shift = generator() % population;
                for (size_t j = 0; j < count; ++j) {
                    const auto id = (j * population / count + shift) % population;
                    labels[j] = pool.empty() ? uint32_t(id) : pool[id];
                }
                if (sorted)
                    std::sort(labels.begin(), labels.end());
                else
                    std::shuffle(labels.begin(), labels.end(), generator);
            }
            double best = std::numeric_limits<double>::infinity();
            for (int repeat = 0; repeat < 4; ++repeat) {
                const auto before = std::chrono::steady_clock::now();
                for (int q = 0; q < query_count; ++q)
                    exact_knn(
                        Eigen::Map<const Eigen::RowVectorXf>(queries.row(q).data(), dim),
                        k,
                        batches[q],
                        found.data(),
                        dist,
                        nullptr,
                        mlann_detail::compute_neighbor_scores,
                        mlann_detail::compute_neighbor_topk
                    );
                const double per_candidate = elapsed_seconds(before) / (query_count * count);
                if (repeat)
                    best = std::min(best, per_candidate);
            }
            costs.distance_by_candidates.emplace_back(double(count), best);
        }
        costs.distance_by_candidates.emplace_back(double(anchor), costs.distance);
    }

    // Forests override this with the same batched traversal used by query().
    virtual void route_for_timing(const float* query, int* leaves) const {
        std::vector<int> path(depth + 1);
        for (int t = 0; t < n_trees; ++t) {
            tuning_path(query, t, path.data());
            leaves[t] = path.back();
        }
    }

    double benchmark_routing(const Eigen::Ref<const RowMatrix>& queries) const {
        std::vector<int> leaves(n_trees);
        volatile int sink = 0;
        const auto before = std::chrono::steady_clock::now();
        for (int q = 0; q < queries.rows(); ++q) {
            route_for_timing(queries.row(q).data(), leaves.data());
            for (int leaf : leaves)
                sink = leaf;
        }
        const double route_seconds =
            elapsed_seconds(before) / (queries.rows() * double(n_trees) * depth);
        (void) sink;
        return route_seconds;
    }

    void benchmark_vote_updates(
        const std::vector<uint32_t>& labels,
        const std::vector<float>& weights,
        const std::vector<uint16_t>& compact,
        float* totals,
        std::vector<uint32_t>& elected,
        float threshold
    ) const {
        if (probability_scores()) {
            // Match RF's unscaled probability accumulation, not the faster
            // SIMD raw-vote kernel. Random ID order matches leaf payloads.
            threshold /= float(n_trees);
            for (size_t j = 0; j < labels.size(); ++j) {
                float& total = totals[labels[j]];
                const float previous = total;
                total += weights[j];
                if (total / float(n_trees) >= threshold &&
                    (previous / float(n_trees) < threshold || previous == 0))
                    elected.push_back(labels[j]);
            }
        } else if (tuning_unit_labels) {
            mlann_detail::accumulate_unit_votes(labels, totals, threshold, elected);
        } else if (tuning_compact_votes()) {
            mlann_detail::accumulate_neighbor_votes(labels, compact, totals, threshold, elected);
        } else {
            mlann_detail::accumulate_neighbor_votes(labels, weights, totals, threshold, elected);
        }
    }

    virtual QueryKernelCosts benchmark_query_kernels(
        const Eigen::Ref<const RowMatrix>& queries,
        int k,
        Distance dist
    ) const {
        // Include dense scratch clearing and separate vote updates from election.
        // Microbenchmarks use the deployed storage representation and routing path.
        const size_t bench_size = std::min(size_t(n_corpus), size_t(65536));
        std::vector<uint32_t> labels(bench_size);
        std::mt19937 benchmark_rng(1729);
        std::vector<float> weights(bench_size, 1.f);
        std::vector<uint16_t> compact(bench_size, 1);
        mlann_detail::HugeBuffer<float> totals;
        totals.resize(n_corpus);
        std::vector<uint32_t> elected;
        elected.reserve(bench_size);
        std::vector<int> found(k);
        auto reachable = small_ip_candidate_pool(dist);
        QueryKernelCosts costs;
        costs.route = costs.vote = costs.distance = costs.setup = costs.election =
            std::numeric_limits<double>::infinity();
        for (int repeat = 0; repeat < 5; ++repeat) {
            // Rotate the sample instead of reranking an unusually hot fixed subset.
            const size_t shift = benchmark_rng() % size_t(n_corpus);
            for (size_t j = 0; j < bench_size; ++j)
                labels[j] = uint32_t((j * size_t(n_corpus) / bench_size + shift) % n_corpus);
            std::shuffle(labels.begin(), labels.end(), benchmark_rng);
            if (!reachable.empty())
                std::shuffle(reachable.begin(), reachable.end(), benchmark_rng);
            std::fill_n(totals.data(), n_corpus, 0.f);
            elected.clear();
            auto before = std::chrono::steady_clock::now();
            benchmark_vote_updates(labels, weights, compact, totals.data(), elected, 2.f);
            const double updated = elapsed_seconds(before) / bench_size;
            // Measure clearing after voting has touched the buffer. Timing the
            // first clear after a large synthetic rerank overstates short queries.
            before = std::chrono::steady_clock::now();
            std::fill_n(totals.data(), n_corpus, 0.f);
            const double cleared = elapsed_seconds(before);
            before = std::chrono::steady_clock::now();
            benchmark_vote_updates(labels, weights, compact, totals.data(), elected, 1.f);
            const double inserted = elapsed_seconds(before) / bench_size;
            before = std::chrono::steady_clock::now();
            exact_knn(
                Eigen::Map<const Eigen::RowVectorXf>(
                    queries.row(repeat % queries.rows()).data(), dim
                ),
                k,
                reachable.empty() ? labels : reachable,
                found.data(),
                dist,
                nullptr,
                mlann_detail::compute_neighbor_scores,
                mlann_detail::compute_neighbor_topk
            );
            const double reranked =
                elapsed_seconds(before) / (reachable.empty() ? bench_size : reachable.size());
            const double routed = benchmark_routing(queries);
            if (repeat) {
                costs.setup = std::min(costs.setup, cleared);
                costs.vote = std::min(costs.vote, updated);
                costs.election = std::min(costs.election, std::max(0., inserted - updated));
                costs.distance = std::min(costs.distance, reranked);
                costs.route = std::min(costs.route, routed);
            }
        }

        calibrate_leaf_costs(queries, costs, dist);
        if (dist == L2 || !reachable.empty())
            calibrate_rerank_costs(queries, k, dist, false, costs, reachable);
        return costs;
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
                        scores[depth_offset + offset] = tuning_node_score(
                            uint32_t(count), scale, interval.second - interval.first
                        );
                }
            }
        }
    }

    static int tuning_score_batch_size(int tree_count, size_t scores_per_tree) {
        int workers = 1;
#ifdef _OPENMP
        workers = omp_get_max_threads();
#endif
        constexpr size_t budget = size_t(128) * 1024 * 1024;
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

    virtual void fill_tuning_leaf(
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

    virtual void fill_tuning_subtree(
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
