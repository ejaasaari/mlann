#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../detail/huge-buffer.h"
#include "../detail/neighbor-query.h"
#include "../mlann.h"

namespace pca_detail {

struct FitStats {
    int iterations = 0;
    bool fallback = false;
};

// Sparse PCA solves in the smaller feature or sample space. Double precision
// keeps the direction accurate when the leading eigenvalues are close together.
inline Eigen::VectorXf direct_direction(const Eigen::MatrixXd& points) {
    const bool dual = points.cols() < points.rows();
    Eigen::MatrixXd covariance;
    if (dual) {
        covariance.noalias() = points.transpose() * points;
    } else {
        covariance.noalias() = points * points.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("PCA eigensolver failed");
    }

    Eigen::VectorXd direction = solver.eigenvectors().rightCols(1);
    if (dual)
        direction = (points * direction).eval();
    const double norm = direction.stableNorm();
    if (!direction.allFinite() || !(norm > 0)) {
        throw std::runtime_error("PCA eigensolver returned an invalid direction");
    }
    return (direction / norm).cast<float>();
}

// PCA trades exact convergence for bounded build cost. Apply the covariance
// through the samples, with no dense covariance matrix or eigensolver fallback.
inline Eigen::VectorXf power_direction(
    const Eigen::MatrixXf& centered,
    Eigen::VectorXf direction,
    FitStats* stats
) {
    constexpr int max_iterations = 20;
    constexpr float tolerance = 1e-3f;
    Eigen::VectorXf projected(centered.cols());
    Eigen::VectorXf product(centered.rows());
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        projected.noalias() = centered.transpose() * direction;
        product.noalias() = centered * projected;
        if (stats)
            ++stats->iterations;

        const float norm = product.norm();
        if (!(norm > 0)) {
            // An initial direction in the nullspace has no covariance product. A
            // nonzero sample gives iteration a direction within the data's span.
            Eigen::Index column;
            centered.colwise().squaredNorm().maxCoeff(&column);
            direction = centered.col(column).normalized();
            if (stats)
                stats->fallback = true;
            continue;
        }

        const float eigenvalue = direction.dot(product);
        const float residual = (product - eigenvalue * direction).norm();
        direction = product / norm;
        if (residual <= tolerance * eigenvalue)
            break;
    }
    return direction;
}

// Points are columns. SparsePCA fits in double precision; PCA scales centered
// samples before float power iteration.
inline Eigen::VectorXf principal_direction(
    const Eigen::Ref<const Eigen::MatrixXf>& points,
    Eigen::VectorXf initial,
    bool approximate,
    FitStats* stats = nullptr,
    bool points_validated = false
) {
    if (stats)
        *stats = {};
    if (points.rows() == 0 || points.cols() < 2 || (!points_validated && !points.allFinite())) {
        throw std::invalid_argument("PCA requires finite points and at least two samples");
    }
    if (initial.size() != points.rows() || !initial.allFinite() || initial.stableNorm() == 0) {
        initial = Eigen::VectorXf::Ones(points.rows());
    }
    initial /= initial.cwiseAbs().maxCoeff();
    initial /= initial.norm();

    if (!approximate) {
        Eigen::MatrixXd centered = points.cast<double>();
        const Eigen::VectorXd mean = centered.rowwise().mean();
        centered.colwise() -= mean;
        // Finite float inputs and an int-sized row count cannot overflow or
        // underflow double covariance products, so no intermediate scaling is needed.
        if (centered.isZero(0))
            return initial;
        return direct_direction(centered);
    }

    const Eigen::VectorXd mean = points.cast<double>().rowwise().mean();
    double scale = 0;
    for (Eigen::Index i = 0; i < points.cols(); ++i) {
        scale = std::max(scale, (points.col(i).cast<double>() - mean).cwiseAbs().maxCoeff());
    }
    if (scale == 0)
        return initial; // Every direction is valid for constant data.

    Eigen::MatrixXf centered(points.rows(), points.cols());
    for (Eigen::Index i = 0; i < points.cols(); ++i) {
        centered.col(i) = ((points.col(i).cast<double>() - mean) / scale).cast<float>();
    }
    centered /= centered.norm();

    return power_direction(centered, initial, stats);
}

} // namespace pca_detail

// Median-split PCA forest. SparsePCA samples coordinates with replacement;
// PCA uses every coordinate and caps fitting rows with n_subsample.
class SparsePCA : public MLANN {
  public:
    SparsePCA(const float* corpus_, int n_corpus_, int dim_)
        : SparsePCA(corpus_, n_corpus_, dim_, false) {}

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
            !corpus.allFinite() ||
            (!unsupervised &&
             (knn.rows() != n_train || knn.cols() < 1 || knn.maxCoeff() >= uint32_t(n_corpus)))) {
            throw std::invalid_argument("Invalid forest data or dimensions.");
        }
        const float requested_density =
            full_dimensions ? 1.f : (density_ < 0 ? float(1.0 / std::sqrt(dim)) : density_);
        if (!std::isfinite(requested_density) || requested_density < 0.f ||
            requested_density > 1.f) {
            throw std::invalid_argument("Density must belong to [0, 1].");
        }

        corpus_leaves = unsupervised;
        n_trees = n_trees_;
        depth = depth_;
        n_inner_nodes = (1 << depth) - 1;
        n_leaves = 1 << depth;
        n_array = 1 << (depth + 1);
        b = b_;
        density = requested_density;
        support = static_cast<int>(density * dim);
        const Eigen::Index node_count = Eigen::Index(n_inner_nodes) * n_trees;

        split_points.resize(n_inner_nodes, n_trees);
        projections.resize(node_count, support);
        // Full-support nodes share implicit coordinates 0..dim-1.
        if (!full_dimensions)
            projection_dims.resize(node_count, support);
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
        std::exception_ptr error;

#pragma omp parallel
        {
            TreeScratch scratch(corpus_leaves ? 0 : n_corpus, n_train, n_subsample);
#pragma omp for schedule(dynamic, 1) nowait
            for (int tree = 0; tree < n_trees; ++tree) {
                try {
                    labels_all[tree].resize(n_leaves);
                    if (compact_leaf_votes)
                        votes16_all[tree].resize(n_leaves);
                    else if (!corpus_leaves)
                        votes_all[tree].resize(n_leaves);
                    std::iota(scratch.rows.begin(), scratch.rows.end(), 0);

                    std::random_device rd;
                    std::minstd_rand generator(rd());
                    initialize_projections(tree, generator);
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
                } catch (...) {
#pragma omp critical(pca_build_error)
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }
        }
        if (error) {
            n_trees = 0;
            labels_all.clear();
            votes_all.clear();
            votes16_all.clear();
            projections.resize(0, 0);
            projection_dims.resize(0, 0);
            split_points.resize(0, 0);
            std::rethrow_exception(error);
        }
        mlann_detail::promote_existing_corpus_pages(
            corpus.data(), size_t(corpus.size()) * sizeof(float)
        );
    }

  public:
    std::unique_ptr<MLANN> make_view(int trees, int d) const override {
        auto view = std::unique_ptr<SparsePCA>(
            new SparsePCA(corpus.data(), n_corpus, dim, full_dimensions)
        );
        initialize_view(*view, trees, d);
        view->corpus_leaves = tuning_unit_labels;
        view->compact_leaf_votes = view->compact_view_votes(tuning_unit_labels, view->votes16_all);
        view->support = support;
        view->n_subsample = n_subsample;
        view->projections.resize(Eigen::Index(trees) * view->n_inner_nodes, support);
        if (!full_dimensions)
            view->projection_dims.resize(view->projections.rows(), support);
        for (int t = 0; t < trees; ++t) {
            view->projections.middleRows(view->projection_row(t, 0), view->n_inner_nodes) =
                projections.middleRows(projection_row(t, 0), view->n_inner_nodes);
            if (!full_dimensions)
                view->projection_dims.middleRows(view->projection_row(t, 0), view->n_inner_nodes) =
                    projection_dims.middleRows(projection_row(t, 0), view->n_inner_nodes);
        }
        return view;
    }

    size_t index_bytes() const override {
        return MLANN::index_bytes() + sizeof(SparsePCA) - sizeof(MLANN) +
               payload_bytes(votes16_all) + size_t(projections.size()) * sizeof(float) +
               size_t(projection_dims.size()) * sizeof(uint32_t);
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
            const float score = project(q, projection_row(tree, node));
            path[level + 1] = 2 * node + (score <= split_points(node, tree) ? 1 : 2);
        }
    }
    int n_subsample = 300;
    bool compact_leaf_votes = false;
    std::vector<std::vector<std::vector<uint16_t>>> votes16_all;

    SparsePCA(const float* corpus_, int n_corpus_, int dim_, bool full_dimensions_)
        : MLANN(corpus_, n_corpus_, dim_), full_dimensions(full_dimensions_) {}

  private:
    using IndexIterator = std::vector<int>::iterator;
    static constexpr int routing_batch_size = 64;
    bool corpus_leaves = false;
    const bool full_dimensions;
    int support = 0;
    RowMatrix projections;
    UIntRowMatrix projection_dims;

    struct TreeScratch {
        std::vector<int> rows;
        std::vector<int> votes;
        std::vector<uint32_t> touched_ids;
        std::vector<int> sampled_rows;
        std::vector<float> row_scores;
        Eigen::MatrixXf fit;

        TreeScratch(int corpus_size, int train_size, int n_subsample)
            : rows(train_size), votes(corpus_size, 0), row_scores(train_size) {
            sampled_rows.reserve(std::min(n_subsample, train_size));
        }
    };

    Eigen::Index projection_row(int tree, int node) const {
        return Eigen::Index(tree) * n_inner_nodes + node;
    }

    void initialize_projections(int tree, std::minstd_rand& generator) {
        std::uniform_int_distribution<int> coordinate(0, dim - 1);
        std::normal_distribution<float> normal(0, 1);
        for (int node = 0; node < n_inner_nodes; ++node) {
            const auto row = projection_row(tree, node);
            if (!full_dimensions) {
                for (int j = 0; j < support; ++j)
                    projection_dims(row, j) = coordinate(generator);
            }
            for (int j = 0; j < support; ++j)
                projections(row, j) = normal(generator);
        }
    }

    void gather_points(
        IndexIterator begin,
        int count,
        Eigen::Index row,
        const Eigen::Ref<const RowMatrix>& train,
        Eigen::MatrixXf& output
    ) const {
        if (output.rows() != support || output.cols() < count)
            output.resize(support, count);
        for (int i = 0; i < count; ++i) {
            const float* point = train.row(begin[i]).data();
            float* column = output.col(i).data();
            if (full_dimensions) {
                std::copy_n(point, support, column);
            } else {
                for (int j = 0; j < support; ++j)
                    column[j] = point[projection_dims(row, j)];
            }
        }
    }

    void fit_projection(
        IndexIterator begin,
        IndexIterator end,
        Eigen::Index row,
        const Eigen::Ref<const RowMatrix>& train,
        std::minstd_rand& generator,
        TreeScratch& scratch
    ) {
        const int count = end - begin;
        if (support == 0) {
            for (auto it = begin; it != end; ++it)
                scratch.row_scores[*it] = 0.f;
            return;
        }
        int fit_count = count;
        if (full_dimensions && n_subsample > 0 && count > n_subsample) {
            scratch.sampled_rows.clear();
            std::sample(
                begin, end, std::back_inserter(scratch.sampled_rows), n_subsample, generator
            );
            fit_count = n_subsample;
            gather_points(scratch.sampled_rows.begin(), fit_count, row, train, scratch.fit);
        } else {
            gather_points(begin, fit_count, row, train, scratch.fit);
        }

        const auto points = scratch.fit.leftCols(fit_count);
        // grow_impl validates all training values before fitting any node.
        const Eigen::VectorXf direction = pca_detail::principal_direction(
            points, projections.row(row).transpose(), full_dimensions, nullptr, full_dimensions
        );
        projections.row(row) = direction.transpose();

        if (!full_dimensions) {
            // Sparse fits contain every row, so their gathered coordinates can be reused.
            for (int i = 0; i < count; ++i) {
                const float* point = scratch.fit.col(i).data();
                float score = 0.f;
                for (int j = 0; j < support; ++j)
                    score += point[j] * projections(row, j);
                scratch.row_scores[begin[i]] = score;
            }
        } else {
            // Interleave four rows to reuse weights and overlap independent sums,
            // preserving each row's feature accumulation order.
            int i = 0;
            for (; i + 4 <= count; i += 4) {
                const float* p0 = train.row(begin[i + 0]).data();
                float s0 = 0.f;
                const float* p1 = train.row(begin[i + 1]).data();
                float s1 = 0.f;
                const float* p2 = train.row(begin[i + 2]).data();
                float s2 = 0.f;
                const float* p3 = train.row(begin[i + 3]).data();
                float s3 = 0.f;
                for (int j = 0; j < support; ++j) {
                    const float weight = projections(row, j);
                    s0 += p0[j] * weight;
                    s1 += p1[j] * weight;
                    s2 += p2[j] * weight;
                    s3 += p3[j] * weight;
                }
                scratch.row_scores[begin[i + 0]] = s0;
                scratch.row_scores[begin[i + 1]] = s1;
                scratch.row_scores[begin[i + 2]] = s2;
                scratch.row_scores[begin[i + 3]] = s3;
            }
            for (; i < count; ++i)
                scratch.row_scores[begin[i]] = project(train.row(begin[i]).data(), row);
        }
    }

    float project(const float* point, Eigen::Index row) const {
        float score = 0.f;
        if (full_dimensions) {
            for (int j = 0; j < support; ++j)
                score += point[j] * projections(row, j);
        } else {
            for (int j = 0; j < support; ++j) {
                score += point[projection_dims(row, j)] * projections(row, j);
            }
        }
        return score;
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
        const int count = end - begin;
        const auto mid = end - count / 2;
        fit_projection(begin, end, projection_row(tree, node), train, generator, scratch);
        const auto less = [&](int left, int right) {
            return scratch.row_scores[left] < scratch.row_scores[right];
        };
        miniselect::pdqselect_branchless(begin, begin + count / 2, end, less);
        if (count % 2) {
            split_points(node, tree) = scratch.row_scores[*(mid - 1)];
        } else {
            const auto left = std::max_element(begin, mid, less);
            split_points(node, tree) = (scratch.row_scores[*mid] + scratch.row_scores[*left]) / 2.0;
        }
        grow_subtree(begin, mid, level + 1, 2 * node + 1, tree, knn, train, generator, scratch);
        grow_subtree(mid, end, level + 1, 2 * node + 2, tree, knn, train, generator, scratch);
    }

    void route_batch(const float* query, int first, int count, int* leaves) const {
        std::array<int, routing_batch_size> nodes{};
        for (int level = 0; level < depth; ++level) {
            for (int t = 0; t < count; ++t) {
                const auto row = projection_row(first + t, nodes[t]);
                const float score = project(query, row);
                nodes[t] = 2 * nodes[t] + (score <= split_points(nodes[t], first + t) ? 1 : 2);
            }
        }
        for (int t = 0; t < count; ++t)
            leaves[t] = nodes[t] - n_inner_nodes;
    }
};

class PCA : public SparsePCA {
  public:
    PCA(const float* corpus_, int n_corpus_, int dim_, int n_subsample_ = 300)
        : SparsePCA(corpus_, n_corpus_, dim_, true) {
        configure(n_subsample_);
    }

    void configure(int n_subsample_) {
        if (!empty())
            throw std::logic_error("The index has already been grown.");
        if (n_subsample_ < 0 || n_subsample_ == 1)
            throw std::invalid_argument(
                "PCA n_subsample must be 0 or at least 2; 0 uses all node rows."
            );
        n_subsample = n_subsample_;
    }
};
