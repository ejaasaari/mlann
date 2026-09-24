#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

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
    void enable_tuning(bool structure_only = false);

    void enable_view_cache();

    void clear_view_cache() const;

    struct ViewCacheInfo;

    ViewCacheInfo view_cache_info() const;

    virtual std::unique_ptr<MLANN> make_view(int, int) const;

    struct Calibration;

    std::vector<Calibration> calibrate(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& truth,
        int min_depth,
        double target,
        float fixed_threshold = 0
    ) const;

    struct CostEstimate;

    struct FrontierConfiguration;

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
    ) const;

    // Uniform label sampling estimates work, not calibration recall. For a small
    // reachable IP universe, omit known-zero IDs and scale by that universe.
    std::vector<CostEstimate> estimate_costs(
        const Eigen::Ref<const RowMatrix>& queries,
        const std::vector<Calibration>& configs,
        const std::vector<uint32_t>& sample,
        int k,
        Distance dist
    ) const;

    std::vector<double> predict_recall(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& truth,
        int trees,
        int d,
        float threshold
    ) const;

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

    virtual bool probability_scores() const;
    virtual float tuning_node_score(uint32_t count, float scale, int) const;
    virtual void tuning_path(const float*, int, int*) const;
    void prepare_tuning(const Eigen::Ref<const UIntRowMatrix>& labels, bool unit, int rows);
    void record_tuning_node(
        int tree,
        int node,
        std::vector<int>::iterator begin,
        std::vector<int>::iterator end
    );
    void finish_tuning_tree(int tree, const std::vector<int>& rows);
    void check_view(int trees, int d) const;

    struct QueryKernelCosts;

    static bool lower_cost(const CostEstimate& a, const CostEstimate& b);

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
    ) const;

    static std::vector<FrontierConfiguration> compact_recall_frontier(
        const std::vector<FrontierConfiguration>& best
    );

    static double elapsed_seconds(std::chrono::steady_clock::time_point started);

    // Conservative owned-storage bound; excludes allocator and query scratch.
    virtual size_t tuning_storage_bound(const Calibration& config, size_t method_bytes) const;

    std::vector<CostEstimate> estimate_sampled_work(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& ids,
        const std::vector<Calibration>& configs,
        int min_depth,
        bool track_support = false,
        size_t population = 0
    ) const;

    virtual bool tuning_compact_votes() const;
    virtual bool sparse_tuning_votes() const;

    struct SampledTuningLeaf;
    struct SampledLeafWork;
    struct LeafWorkSample;
    static constexpr size_t leaf_work_budget = 16 << 20;
    static constexpr size_t candidate_pool_scan_budget = 4 << 20;

    LeafWorkSample sample_leaf_work(const Eigen::Ref<const RowMatrix>& queries) const;

    struct LeafCostObservation;

    virtual double benchmark_leaf_probe(
        const LeafWorkSample& sample,
        const SampledLeafWork& work,
        float threshold,
        float* totals,
        std::vector<uint32_t>& elected,
        size_t& candidates,
        double& setup_seconds
    ) const;

    static Eigen::Vector3d solve_nonnegative_leaf_costs(
        const Eigen::Matrix3d& normal,
        const Eigen::Vector3d& rhs,
        int dimensions
    );

    void fit_leaf_costs(
        QueryKernelCosts& costs,
        const std::vector<LeafCostObservation>& data,
        bool anchor_zero_election = false
    ) const;

    struct LeafCostProbe;

    struct LeafProbeScratch;

    std::vector<LeafCostProbe> prepare_leaf_probes(
        const LeafWorkSample& sample,
        int prefix,
        LeafProbeScratch& scratch
    ) const;

    std::vector<LeafCostObservation> measure_leaf_probes(
        const LeafWorkSample& sample,
        const std::vector<LeafCostProbe>& probes,
        LeafProbeScratch& scratch
    ) const;

    bool use_prefix_leaf_costs(Distance dist) const;

    void calibrate_leaf_costs(
        const Eigen::Ref<const RowMatrix>& queries,
        QueryKernelCosts& costs,
        Distance dist
    ) const;

    std::vector<uint32_t> small_ip_candidate_pool(Distance dist) const;

    std::vector<uint32_t> bounded_candidate_pool() const;

    std::vector<uint32_t> sample_cost_ids(
        const std::vector<uint32_t>& sample,
        Distance dist,
        size_t& population
    ) const;

    void calibrate_rerank_costs(
        const Eigen::Ref<const RowMatrix>& queries,
        int k,
        Distance dist,
        bool sorted,
        QueryKernelCosts& costs,
        const std::vector<uint32_t>& pool = {}
    ) const;

    // Forests override this with the same batched traversal used by query().
    virtual void route_for_timing(const float* query, int* leaves) const;

    double benchmark_routing(const Eigen::Ref<const RowMatrix>& queries) const;

    void benchmark_vote_updates(
        const std::vector<uint32_t>& labels,
        const std::vector<float>& weights,
        const std::vector<uint16_t>& compact,
        float* totals,
        std::vector<uint32_t>& elected,
        float threshold
    ) const;

    virtual QueryKernelCosts benchmark_query_kernels(
        const Eigen::Ref<const RowMatrix>& queries,
        int k,
        Distance dist
    ) const;

    struct TuningPostings;

    TuningPostings make_tuning_postings(const Eigen::Ref<const UIntRowMatrix>& ids) const;

    // Transform row occurrences into sorted positions in this tree's permutation.
    std::vector<std::vector<int>> tuning_posting_positions(
        int tree,
        const TuningPostings& postings
    ) const;

    struct TuningMassCache;

    float tuning_node_scale(int tree, int node, TuningMassCache& cache) const;

    void score_tuning_tree(
        const Eigen::Ref<const RowMatrix>& queries,
        const Eigen::Ref<const UIntRowMatrix>& ids,
        const std::vector<size_t>& offsets,
        const TuningPostings& postings,
        int tree,
        int min_depth,
        int max_depth,
        float* scores
    ) const;

    static int tuning_score_batch_size(int tree_count, size_t scores_per_tree);

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
    ) const;

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
    ) const;

    void count_node(
        int tree,
        int node,
        std::vector<uint32_t>& counts,
        std::vector<uint32_t>& touched
    ) const;
    virtual std::pair<int, int> tuning_children(int tree, int node) const;

    virtual size_t tuning_leaf_slots(int, int d) const;

    virtual int tuning_leaf_slot(int node, int level, int d, int) const;

    virtual void fill_tuning_leaf(
        TuningLeafPayload& leaf,
        int tree,
        int node,
        std::vector<uint32_t>& counts,
        std::vector<uint32_t>& touched
    ) const;

    virtual void fill_tuning_subtree(
        TuningTreePayload& payload,
        int tree,
        int node,
        int level,
        int d,
        int& cursor,
        std::vector<uint32_t>& counts,
        std::vector<uint32_t>& touched
    ) const;

    std::shared_ptr<const TuningTreePayload> make_tuning_payload(
        int tree,
        int d,
        std::vector<uint32_t>& counts,
        std::vector<uint32_t>& touched
    ) const;

    void initialize_shared_payloads(MLANN& view, int trees, int d) const;

    void initialize_view(MLANN& view, int trees, int d) const;
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

#include "autotune.h"
