#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../detail/huge-buffer.h"
#include "../detail/neighbor-query.h"
#include "../mlann.h"
#include "../utils.h"

namespace pls_detail {

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Sample {
    Matrix x;                // Centered queries, all input coordinates.
    std::vector<int> labels; // Compact corpus IDs, row major N x K.
    std::vector<int> counts; // Occurrences of each compact label.
    int k = 0;

    int n() const { return x.rows(); }
};

// Single-precision fitting, including centering, products and spectral solves.
inline Eigen::VectorXf leading_dense(const Eigen::MatrixXf& matrix) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(matrix);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Spectral solve failed");
    }
    if (!(solver.eigenvalues().tail(1)[0] > 0)) {
        return Eigen::VectorXf::Zero(matrix.rows());
    }
    return solver.eigenvectors().col(matrix.rows() - 1);
}

struct LeadingEigenStats {
    int iterations = 0;
    bool fallback = false;
};

// Bound the largest eigenvalue of a normalized symmetric tridiagonal matrix.
inline float largest_eigenvalue_bound(
    const Eigen::VectorXf& diagonal,
    const Eigen::VectorXf& off_diagonal
) {
    const int n = diagonal.size();

    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    constexpr float pivot_min = 16 * std::numeric_limits<float>::min();

    float lower = diagonal.maxCoeff();
    float upper = lower;
    for (int i = 0; i < n; ++i) {
        upper = std::max(
            upper, diagonal[i] + (i ? off_diagonal[i - 1] : 0) + (i + 1 < n ? off_diagonal[i] : 0)
        );
    }
    upper += 8 * epsilon;

    // Count eigenvalues below the trial shift using the signs of LDL' pivots.
    for (int step = 0; step < 64 && upper - lower > 8 * epsilon; ++step) {
        const float shift = lower + (upper - lower) / 2;
        float pivot = diagonal[0] - shift;
        if (std::abs(pivot) < pivot_min) {
            pivot = -pivot_min;
        }
        int below = pivot < 0;
        for (int i = 1; i < n; ++i) {
            pivot = diagonal[i] - shift - off_diagonal[i - 1] * (off_diagonal[i - 1] / pivot);
            if (std::abs(pivot) < pivot_min) {
                pivot = -pivot_min;
            }
            below += pivot < 0;
        }

        if (below < n) {
            lower = shift;
        } else {
            upper = shift;
        }
    }
    return upper;
}

// Compute only the largest eigenpair: Householder tridiagonalization, Sturm
// bisection, then inverse iteration. Never form the dense Householder Q or a
// complete eigenvector basis. Small problems and failed checks use Eigen's
// original solver. No random probes or fixed-budget approximate directions.
inline Eigen::VectorXf leading_bisect(
    const Eigen::MatrixXf& matrix,
    LeadingEigenStats* stats = nullptr
) {
    if (stats) {
        *stats = {};
    }
    const int n = matrix.rows();
    if (n <= 8) {
        return leading_dense(matrix);
    }

    const float scale = matrix.cwiseAbs().maxCoeff();
    if (!(scale > 0)) {
        return Eigen::VectorXf::Zero(n);
    }

    const Eigen::MatrixXf scaled = matrix / scale;
    Eigen::Tridiagonalization<Eigen::MatrixXf> reduction(scaled);
    Eigen::VectorXf diagonal = reduction.diagonal();
    Eigen::VectorXf off_diagonal = reduction.subDiagonal();

    // A diagonal sign change makes every off-diagonal nonnegative. Starting
    // inverse iteration with positive entries then cannot be orthogonal to the
    // leading eigenspace, including reducible / repeated-eigenvalue problems.
    Eigen::VectorXf signs = Eigen::VectorXf::Ones(n);
    for (int i = 1; i < n; ++i) {
        signs[i] = off_diagonal[i - 1] < 0 ? -signs[i - 1] : signs[i - 1];
    }

    off_diagonal = off_diagonal.cwiseAbs();
    float norm = 0;
    for (int i = 0; i < n; ++i) {
        norm = std::max(
            norm,
            std::abs(diagonal[i]) + (i ? off_diagonal[i - 1] : 0) +
                (i + 1 < n ? off_diagonal[i] : 0)
        );
    }
    if (!(norm > 0)) {
        return Eigen::VectorXf::Zero(n);
    }

    diagonal /= norm;
    off_diagonal /= norm;

    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    constexpr float tolerance = 2e-6f;
    const float upper = largest_eigenvalue_bound(diagonal, off_diagonal);

    auto fallback = [&]() -> Eigen::VectorXf {
        if (stats) {
            stats->fallback = true;
        }
        return leading_dense(matrix);
    };

    if (upper <= 0) {
        return Eigen::VectorXf::Zero(n);
    }

    // Keep the shifted matrix positive definite, even after bisection rounding.
    const float shift = upper + 8 * epsilon;
    Eigen::VectorXf pivots(n);
    Eigen::VectorXf factors(n - 1);
    pivots[0] = shift - diagonal[0];
    for (int i = 1; i < n; ++i) {
        if (!(pivots[i - 1] > 0)) {
            return fallback();
        }
        factors[i - 1] = -off_diagonal[i - 1] / pivots[i - 1];
        pivots[i] = shift - diagonal[i] + factors[i - 1] * off_diagonal[i - 1];
    }
    if (!(pivots[n - 1] > 0)) {
        return fallback();
    }

    Eigen::VectorXf direction = Eigen::VectorXf::Ones(n);
    Eigen::VectorXf product(n);
    for (int iteration = 0; iteration < 8; ++iteration) {
        if (stats) {
            ++stats->iterations;
        }

        for (int i = 1; i < n; ++i) {
            direction[i] -= factors[i - 1] * direction[i - 1];
        }
        direction.array() /= pivots.array();
        for (int i = n - 2; i >= 0; --i) {
            direction[i] -= factors[i] * direction[i + 1];
        }
        if (!direction.allFinite() || !(direction.norm() > 0)) {
            return fallback();
        }

        direction.normalize();
        product = diagonal.array() * direction.array();
        for (int i = 0; i + 1 < n; ++i) {
            product[i] += off_diagonal[i] * direction[i + 1];
            product[i + 1] += off_diagonal[i] * direction[i];
        }

        const float value = direction.dot(product);
        if (std::abs(upper - value) <= tolerance &&
            (product - value * direction).norm() <= tolerance) {
            if (!(value > 0)) {
                return fallback();
            }

            Eigen::VectorXf result =
                reduction.matrixQ() * (signs.array() * direction.array()).matrix();
            result.normalize();
            const Eigen::VectorXf residual =
                scaled.selfadjointView<Eigen::Lower>() * result - (value * norm) * result;
            if (residual.norm() <= 4 * tolerance * norm) {
                return result;
            }
            return fallback();
        }
    }
    return fallback();
}

// Lanczos with two-pass full reorthogonalization, checked against the original
// matrix. Small systems or unconverged iterations use the selected direct solve.
inline Eigen::VectorXf leading(const Eigen::MatrixXf& matrix, LeadingEigenStats* stats = nullptr) {
    const int n = matrix.rows();
    if (n < 48) {
        return leading_bisect(matrix, stats);
    }
    if (stats) {
        *stats = {};
    }

    const float scale = matrix.cwiseAbs().maxCoeff();
    if (!(scale > 0)) {
        return Eigen::VectorXf::Zero(n);
    }

    const Eigen::MatrixXf scaled_matrix = (matrix / scale).selfadjointView<Eigen::Lower>();
    const int limit = std::min(n, 48);
    Eigen::MatrixXf basis(n, limit);
    Eigen::VectorXf diagonal(limit);
    Eigen::VectorXf off_diagonal(limit);
    Eigen::VectorXf direction(n);

    std::minstd_rand generator(193U + n);
    for (int i = 0; i < n; ++i) {
        direction[i] = std::uniform_real_distribution<float>(-1, 1)(generator);
    }
    direction.normalize();

    const float tolerance = 1e-6f * scaled_matrix.norm();
    for (int j = 0; j < limit; ++j) {
        basis.col(j) = direction;
        Eigen::VectorXf residual = scaled_matrix * direction;
        diagonal[j] = direction.dot(residual);
        residual -= diagonal[j] * direction;
        if (j) {
            residual -= off_diagonal[j - 1] * basis.col(j - 1);
        }

        for (int pass = 0; pass < 2; ++pass) {
            const Eigen::VectorXf coefficients = basis.leftCols(j + 1).transpose() * residual;
            residual.noalias() -= basis.leftCols(j + 1) * coefficients;
        }

        off_diagonal[j] = residual.norm();
        const bool breakdown = off_diagonal[j] <= 32 * std::numeric_limits<float>::epsilon();
        if ((j + 1) % 8 == 0 || breakdown || j + 1 == limit) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> ritz;
            ritz.computeFromTridiagonal(diagonal.head(j + 1), off_diagonal.head(j));
            if (ritz.info() != Eigen::Success) {
                break;
            }

            const float value = ritz.eigenvalues()[j];
            Eigen::VectorXf result = basis.leftCols(j + 1) * ritz.eigenvectors().col(j);
            result.normalize();
            if (value > 0 && (scaled_matrix * result - value * result).norm() <= tolerance) {
                if (stats) {
                    stats->iterations = j + 1;
                }
                return result;
            }
            if (breakdown) {
                break;
            }
        }

        direction = residual / off_diagonal[j];
    }

    Eigen::VectorXf result = leading_bisect(matrix, stats);
    if (stats) {
        stats->fallback = true;
    }
    return result;
}

// An orthonormal change of basis, with no truncation or feature selection.
// When N < dim, every nonzero spectral direction lies in this row space.
struct QueryBasis {
    Eigen::HouseholderQR<Eigen::MatrixXf> qr;
    Eigen::MatrixXf basis;
    Matrix coordinates;
    bool reduced = false;

    explicit QueryBasis(const Matrix& queries, bool implicit = true) {
        if (queries.rows() < queries.cols()) {
            reduced = true;
            if (implicit) {
                // Keep the reflectors and apply them only to the fitted direction.
                qr.compute(queries.transpose());
                coordinates = qr.matrixQR()
                                  .topRows(queries.rows())
                                  .template triangularView<Eigen::Upper>()
                                  .transpose();
            } else {
                Eigen::HouseholderQR<Eigen::MatrixXf> explicit_qr(queries.transpose());
                basis = explicit_qr.householderQ() *
                        Eigen::MatrixXf::Identity(queries.cols(), queries.rows());
                coordinates = explicit_qr.matrixQR()
                                  .topRows(queries.rows())
                                  .template triangularView<Eigen::Upper>()
                                  .transpose();
            }
        }
    }

    const Matrix& get(const Matrix& queries) const { return reduced ? coordinates : queries; }

    Eigen::VectorXf expand(const Eigen::VectorXf& direction) const {
        Eigen::VectorXf result;
        if (!reduced) {
            result = direction;
        } else if (basis.size()) {
            result = basis * direction;
        } else {
            Eigen::VectorXf padded = Eigen::VectorXf::Zero(qr.rows());
            padded.head(direction.size()) = direction;
            result = qr.householderQ() * padded;
        }
        if (result.squaredNorm() > 0) {
            result.normalize();
        }
        return result;
    }
};

inline RowMatrix neighbor_means(
    const Eigen::Ref<const RowMatrix>& corpus,
    const Eigen::Ref<const UIntRowMatrix>& labels
) {
    RowMatrix means(labels.rows(), corpus.cols());
#pragma omp parallel
    {
        Eigen::RowVectorXf sum(corpus.cols());
        // Release worker-local storage before the parallel-region barrier.
#pragma omp for schedule(static) nowait
        for (int i = 0; i < labels.rows(); ++i) {
            sum.setZero();
            for (int j = 0; j < labels.cols(); ++j) {
                sum += corpus.row(labels(i, j));
            }
            means.row(i) = (sum / labels.cols());
        }
    }
    return means;
}

// Fit the query direction from its cross-covariance with neighbor means.
inline Eigen::VectorXf pls(const Sample& sample, const Matrix& targets) {
    QueryBasis query_basis(sample.x);
    Eigen::MatrixXf cross;
    if (query_basis.reduced) {
        const int size = sample.n();
        cross.resize(size, targets.cols());
        // Columns before each diagonal block are zero in the transposed QR factor.
        for (int first = 0; first < size; first += 32) {
            const int rows = std::min(32, size - first);
            cross.middleRows(first, rows).noalias() =
                query_basis.coordinates.transpose().block(first, first, rows, size - first) *
                targets.bottomRows(size - first);
        }
        cross /= size;
    } else {
        cross = (sample.x.transpose() * targets) / sample.n();
    }
    Eigen::MatrixXf gram = Eigen::MatrixXf::Zero(cross.rows(), cross.rows());
    gram.selfadjointView<Eigen::Lower>().rankUpdate(cross);
    gram.triangularView<Eigen::StrictlyUpper>() = gram.transpose();
    return query_basis.expand(leading(gram));
}

inline float clogc(int count) {
    return count ? count * std::log(float(count)) : 0;
}

struct PALTables {
    std::vector<float> label;
    std::vector<float> mass;
    std::vector<float> delta;

    PALTables(int n, int k) : label(n + 1), mass(n + 1), delta(n + 1) {
        for (int i = 1; i <= n; ++i) {
            label[i] = clogc(i);
            mass[i] = clogc(k * i);
            delta[i] = label[i] - label[i - 1];
        }
    }
};

struct Split {
    float threshold = 0;
    float loss = INFINITY;
    bool valid = false;
    float gain = 0;
};

struct ThresholdScratch {
    std::vector<int> order;
    std::vector<int> left;
    std::vector<int> right;
};

inline Split threshold(
    const Sample& sample,
    const Eigen::VectorXf& projection,
    const PALTables& tables,
    ThresholdScratch& scratch
) {
    auto& order = scratch.order;
    auto& left = scratch.left;
    auto& right = scratch.right;
    order.resize(sample.n());
    left.assign(sample.counts.size(), 0);
    right = sample.counts;
    std::iota(order.begin(), order.end(), 0);
    miniselect::pdqsort_branchless(order.begin(), order.end(), [&](int i, int j) {
        return projection[i] < projection[j] || (projection[i] == projection[j] && i < j);
    });

    float left_entropy = 0;
    float right_entropy = 0;
    for (int c : sample.counts) {
        right_entropy += tables.label[c];
    }

    const float parent = tables.mass[sample.n()] - right_entropy;
    Split best;
    for (int split_index = 0; split_index + 1 < sample.n(); ++split_index) {
        for (int j = 0; j < sample.k; ++j) {
            const int label = sample.labels[order[split_index] * sample.k + j];
            left_entropy += tables.delta[++left[label]];
            right_entropy -= tables.delta[right[label]--];
        }

        const float lower = projection[order[split_index]];
        const float upper = projection[order[split_index + 1]];
        if (!(lower < upper)) {
            continue;
        }

        const float loss = tables.mass[split_index + 1] +
                           tables.mass[sample.n() - split_index - 1] - left_entropy - right_entropy;
        if (loss < best.loss) {
            float split_point = lower + (upper - lower) / 2;
            if (!(split_point < upper)) {
                split_point = lower;
            }
            best = {split_point, loss, true, parent - loss};
        }
    }
    return best;
}

inline Split threshold(const Sample& sample, const Eigen::VectorXf& projection) {
    const PALTables tables(sample.n(), sample.k);
    ThresholdScratch scratch;
    return threshold(sample, projection, tables, scratch);
}

} // namespace pls_detail

class PLS : public MLANN {
  public:
    PLS(const float* corpus_, int n_corpus_, int dim_, int n_subsample_ = 300)
        : MLANN(corpus_, n_corpus_, dim_) {
        configure(n_subsample_);
    }

    void configure(int n_subsample_) {
        if (!empty())
            throw std::logic_error("The index has already been grown.");
        if (n_subsample_ < 2)
            throw std::invalid_argument("n_subsample must be >= 2");
        n_subsample = n_subsample_;
    }

    using MLANN::query;

    void grow(
        int n_trees_,
        int depth_,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        float density_ = -1.0,
        int b_ = 1
    ) override {
        validate_training(n_trees_, depth_, knn, train, b_);

        // Accepted for compatibility with MLANN::grow; PLS uses every feature.
        (void) density_;
        n_trees = n_trees_;
        depth = depth_;
        b = b_;

        forests.resize(n_trees_);
        leaves_all.resize(n_trees_);
        std::vector<std::vector<float>> tree_projections(n_trees_);
        const pls_detail::PALTables tables(std::min<int>(n_subsample, train.rows()), knn.cols());

        // Compute neighbor means once and release them after fitting the forest.
        RowMatrix targets = pls_detail::neighbor_means(corpus, knn);
#pragma omp parallel
        {
            TreeScratch scratch(n_corpus);
            // Release each worker's scratch as soon as its last tree finishes.
            // The parallel-region barrier still waits for all trees.
#pragma omp for schedule(dynamic, 1) nowait
            for (int t = 0; t < n_trees_; ++t) {
                std::vector<int> rows(train.rows());
                std::iota(rows.begin(), rows.end(), 0);
                forests[t].reserve(
                    std::min<size_t>((size_t(1) << (depth + 1)) - 1, 2 * train.rows())
                );
                scratch.projections.clear();
                grow_subtree(rows.begin(), rows.end(), 0, t, train, knn, targets, tables, scratch);
                forests[t].shrink_to_fit();
                leaves_all[t].shrink_to_fit();
                tree_projections[t] = std::move(scratch.projections);
            }
        }

        targets.resize(0, 0);
        pack_projections(tree_projections);
        mlann_detail::promote_existing_corpus_pages(
            corpus.data(), size_t(corpus.size()) * sizeof(float)
        );
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
                const auto& leaf = leaves_all[first + t][leaves[t]];
                mlann_detail::accumulate_neighbor_votes(
                    leaf.labels, leaf.votes, votes_total.data(), vote_threshold, elected
                );
            }
        }

        if (out_n_elected) {
            *out_n_elected = elected.size();
        }
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
    struct Node {
        uint32_t projection = 0;
        float threshold = 0;
        int left = -1;
        int right = -1;
        int leaf = -1;
    };

    struct Leaf {
        std::vector<uint32_t> labels;
        std::vector<float> votes;
    };

    std::vector<std::vector<Node>> forests;
    std::vector<std::vector<Leaf>> leaves_all;
    RowMatrix projections;
    static constexpr int routing_batch_size = 64;

    void route_batch(const float* query, int first, int count, int* leaves) const {
        std::array<int, routing_batch_size> nodes{};
        std::array<int, routing_batch_size> active;
        std::array<uint32_t, routing_batch_size> rows;
        std::array<float, routing_batch_size> scores;
        int remaining = 0;
        for (int t = 0; t < count; ++t) {
            if (forests[first + t][0].leaf < 0) {
                active[remaining++] = t;
            }
        }

        while (remaining) {
            for (int i = 0; i < remaining; ++i) {
                const int t = active[i];
                rows[i] = forests[first + t][nodes[t]].projection;
            }

            mlann_detail::compute_neighbor_one_to_many(
                query,
                projections.data(),
                dim,
                rows.data(),
                remaining,
                mlann_detail::OneToManyMetric::IP,
                scores.data()
            );

            int next = 0;
            for (int i = 0; i < remaining; ++i) {
                const int t = active[i];
                const auto& node = forests[first + t][nodes[t]];
                nodes[t] = scores[i] <= node.threshold ? node.left : node.right;
                if (forests[first + t][nodes[t]].leaf < 0) {
                    active[next++] = t;
                }
            }
            remaining = next;
        }

        for (int t = 0; t < count; ++t) {
            leaves[t] = forests[first + t][nodes[t]].leaf;
        }
    }

  private:
    using IndexIterator = std::vector<int>::iterator;
    int n_subsample = 300;

    struct TreeScratch {
        std::vector<int> label_map;
        std::vector<uint32_t> touched_ids;

        std::minstd_rand generator;
        std::vector<float> projections;
        pls_detail::ThresholdScratch threshold_scratch;

        explicit TreeScratch(int n_corpus)
            : label_map(n_corpus, 0), generator(std::random_device{}()) {}

        void reset() {
            for (auto id : touched_ids) {
                label_map[id] = 0;
            }
            touched_ids.clear();
        }
    };

    void validate_training(
        int n_trees_,
        int depth_,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        int b_
    ) const {
        if (!empty()) {
            throw std::logic_error("Index already grown");
        }
        if (dim < 1 || n_corpus < 1 || n_trees_ <= 0 || train.rows() < 2 || depth_ < 1 ||
            depth_ > 29 || depth_ > std::log2(train.rows()) || b_ < 1 || knn.cols() < 1 ||
            knn.rows() != train.rows() || train.cols() != dim || !train.allFinite() ||
            !corpus.allFinite() || knn.maxCoeff() >= uint32_t(n_corpus)) {
            throw std::invalid_argument("Invalid forest data or dimensions");
        }

        // Reuse one sorting buffer per worker instead of allocating for every row.
        int duplicates = 0;
#pragma omp parallel reduction(| : duplicates)
        {
            std::vector<uint32_t> ids(knn.cols());
            // Release worker-local storage before the parallel-region barrier.
#pragma omp for schedule(static) nowait
            for (int i = 0; i < knn.rows(); ++i) {
                std::copy_n(knn.row(i).data(), knn.cols(), ids.begin());
                miniselect::pdqsort_branchless(ids.begin(), ids.end());
                duplicates |= std::adjacent_find(ids.begin(), ids.end()) != ids.end();
            }
        }

        if (duplicates) {
            throw std::invalid_argument("Neighbor IDs must be distinct within each row");
        }
    }

    void pack_projections(std::vector<std::vector<float>>& trees) {
        size_t rows = 0;
        for (const auto& values : trees) {
            rows += values.size() / dim;
        }
        if (rows > std::numeric_limits<uint32_t>::max()) {
            throw std::length_error("Too many oblique projections");
        }

        projections.resize(rows, dim);
        size_t offset = 0;
        for (int t = 0; t < n_trees; ++t) {
            auto& values = trees[t];
            if (!values.empty()) {
                std::copy(values.begin(), values.end(), projections.data() + offset * dim);
                for (auto& node : forests[t]) {
                    if (node.leaf < 0) {
                        node.projection += uint32_t(offset);
                    }
                }
                offset += values.size() / dim;
            }
            std::vector<float>().swap(values);
        }
    }

    // Fitting, full-node partitioning and querying share the same float SIMD
    // reduction, including its feature tails.
    static float project(const Eigen::VectorXf& normal, const float* row) {
        const uint32_t zero = 0;
        float score;
        mlann_detail::compute_one_to_many(
            row, normal.data(), normal.size(), &zero, 1, mlann_detail::OneToManyMetric::IP, &score
        );
        return score;
    }

    void make_leaf(
        Node& node,
        int tree,
        IndexIterator begin,
        IndexIterator end,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        TreeScratch& scratch
    ) {
        node.leaf = leaves_all[tree].size();
        leaves_all[tree].emplace_back();
        auto& output = leaves_all[tree].back();

        scratch.touched_ids.reserve(std::min<size_t>(n_corpus, size_t(end - begin) * knn.cols()));
        for (auto it = begin; it != end; ++it) {
            for (int j = 0; j < knn.cols(); ++j) {
                const uint32_t id = knn(*it, j);
                if (scratch.label_map[id]++ == 0) {
                    scratch.touched_ids.push_back(id);
                }
            }
        }

        output.labels.reserve(scratch.touched_ids.size());
        output.votes.reserve(scratch.touched_ids.size());
        uint64_t total = 0;
        for (auto id : scratch.touched_ids) {
            if (scratch.label_map[id] >= b) {
                output.labels.push_back(id);
                output.votes.push_back(scratch.label_map[id]);
                total += scratch.label_map[id];
            }
        }

        if (total) {
            const float normalization = 1.f / (float(total) * float(n_trees));
            for (auto& v : output.votes) {
                v *= normalization;
            }
        }

        scratch.reset();
    }

    pls_detail::Sample sample_queries(
        IndexIterator begin,
        const std::vector<int>& sampled_rows,
        const Eigen::Ref<const RowMatrix>& train,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        TreeScratch& scratch
    ) const {
        const int n_sampled = sampled_rows.size();
        pls_detail::Sample sampled_queries;
        sampled_queries.k = knn.cols();
        sampled_queries.x.resize(n_sampled, dim);
        sampled_queries.labels.resize(size_t(n_sampled) * sampled_queries.k);
        sampled_queries.counts.reserve(std::min<size_t>(n_corpus, sampled_queries.labels.size()));

        for (int i = 0; i < n_sampled; ++i) {
            const int row = begin[sampled_rows[i]];
            sampled_queries.x.row(i) = train.row(row);
            for (int j = 0; j < sampled_queries.k; ++j) {
#if defined(__GNUC__) || defined(__clang__)
                if (j + 16 < sampled_queries.k) {
                    __builtin_prefetch(&scratch.label_map[knn(row, j + 16)], 1, 1);
                }
#endif
                const auto label = knn(row, j);
                if (!scratch.label_map[label]) {
                    scratch.touched_ids.push_back(label);
                    sampled_queries.counts.push_back(0);
                    scratch.label_map[label] = sampled_queries.counts.size();
                }
                const int compact_label = scratch.label_map[label] - 1;
                sampled_queries.labels[i * sampled_queries.k + j] = compact_label;
                ++sampled_queries.counts[compact_label];
            }
        }

        scratch.reset();
        sampled_queries.x.rowwise() -= sampled_queries.x.colwise().mean().eval();
        return sampled_queries;
    }

    struct FittedSplit {
        Eigen::VectorXf normal;
        pls_detail::Split split;
    };

    FittedSplit fit_split(
        IndexIterator begin,
        IndexIterator end,
        const Eigen::Ref<const RowMatrix>& train,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const RowMatrix& targets,
        const pls_detail::PALTables& tables,
        TreeScratch& scratch
    ) const {
        const int count = end - begin;
        const int n_sampled = std::min(n_subsample, count);
        const auto sampled_rows = mlann_detail::sample(count, n_sampled, scratch.generator);
        const pls_detail::Sample sampled_queries =
            sample_queries(begin, sampled_rows, train, knn, scratch);

        pls_detail::Matrix neighbor_targets(n_sampled, dim);
        for (int i = 0; i < n_sampled; ++i) {
            neighbor_targets.row(i) = targets.row(begin[sampled_rows[i]]);
        }
        neighbor_targets.rowwise() -= neighbor_targets.colwise().mean().eval();

        Eigen::VectorXf normal = pls_detail::pls(sampled_queries, neighbor_targets);
        Eigen::VectorXf scores(n_sampled);
        for (int i = 0; i < n_sampled; ++i) {
            scores[i] = project(normal, train.row(begin[sampled_rows[i]]).data());
        }

        const pls_detail::Split split =
            pls_detail::threshold(sampled_queries, scores, tables, scratch.threshold_scratch);
        return {std::move(normal), split};
    }

    int grow_subtree(
        IndexIterator begin,
        IndexIterator end,
        int level,
        int tree,
        const Eigen::Ref<const RowMatrix>& train,
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const RowMatrix& targets,
        const pls_detail::PALTables& tables,
        TreeScratch& scratch
    ) {
        const int index = forests[tree].size();
        forests[tree].emplace_back();
        Node node;

        const auto finish_leaf = [&]() {
            make_leaf(node, tree, begin, end, knn, scratch);
            forests[tree][index] = node;
            return index;
        };

        if (level == depth || end - begin < 2) {
            return finish_leaf();
        }

        const FittedSplit fit = fit_split(begin, end, train, knn, targets, tables, scratch);
        if (!fit.split.valid || !(fit.split.gain > 1e-9f)) {
            return finish_leaf();
        }

        // Fit on the sample, then route every training row through the chosen split.
        node.threshold = fit.split.threshold;
        const auto split_position = std::partition(begin, end, [&](int row) {
            return project(fit.normal, train.row(row).data()) <= node.threshold;
        });
        if (split_position == begin || split_position == end) {
            return finish_leaf();
        }

        node.projection = scratch.projections.size() / dim;
        scratch.projections.insert(
            scratch.projections.end(), fit.normal.data(), fit.normal.data() + dim
        );
        node.left = grow_subtree(
            begin, split_position, level + 1, tree, train, knn, targets, tables, scratch
        );
        node.right = grow_subtree(
            split_position, end, level + 1, tree, train, knn, targets, tables, scratch
        );

        // Recursive growth can reallocate the forest, so retain the index, not a reference.
        forests[tree][index] = node;
        return index;
    }
};
