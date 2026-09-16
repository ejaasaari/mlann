#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
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

  protected:
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
            Eigen::MatrixXf::Index index;

            if (dist == L2) {
                distances.minCoeff(&index);
                out[0] = indices[index];
                if (out_distances)
                    out_distances[0] = std::sqrt(distances(index));
            } else {
                distances.maxCoeff(&index);
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
                        return left.score < right.score;
                    }
                );
            } else {
                miniselect::pdqpartial_sort_branchless(
                    scored.data(),
                    scored.data() + n_to_sort,
                    scored.data() + n_elected,
                    [](const ScoredCandidate& left, const ScoredCandidate& right) {
                        return left.score > right.score;
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
