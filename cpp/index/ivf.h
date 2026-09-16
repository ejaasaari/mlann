#pragma once

#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../detail/clustering.h"
#include "../mlann.h"

class IVF : public MLANN {
  public:
    struct LabelScore {
        uint32_t id;
        float score;
    };

    IVF(const float* corpus_, int n_corpus_, int dim_) : MLANN(corpus_, n_corpus_, dim_) {}

    Distance distance() const { return metric; }

    void build(
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        int n_trees_,
        int n_clusters,
        int subspace_dim,
        Distance dist = L2
    ) {
        validate(knn, train, n_trees_, n_clusters, subspace_dim, dist);

        std::random_device rd;
        std::mt19937 generator(rd());
        RowMatrix new_rotation = make_rotation(generator);

        std::vector<int> seeds(n_corpus);
        std::iota(seeds.begin(), seeds.end(), 0);
        std::shuffle(seeds.begin(), seeds.end(), generator);
        seeds.resize(n_clusters);
        const RowMatrix projected_corpus = project(corpus, new_rotation);
        const RowMatrix projected_train = project(train, new_rotation);
        std::vector<Partition> new_partitions(n_trees_);
        std::exception_ptr error;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int tree = 0; tree < n_trees_; ++tree) {
            try {
                auto& partition = new_partitions[tree];
                const Eigen::Index offset = Eigen::Index(tree) * subspace_dim;
                partition.centroids = mlann_detail::KMeans::fit(
                    projected_corpus.middleCols(offset, subspace_dim), seeds, partition.iterations
                );
                partition.norms = partition.centroids.rowwise().squaredNorm();
                const auto assignments = mlann_detail::KMeans::assign(
                    projected_train.middleCols(offset, subspace_dim), partition.centroids
                );
                make_cells(partition, assignments, knn);
            } catch (...) {
#ifdef _OPENMP
#pragma omp critical(ivf_build_error)
#endif
                {
                    if (!error) {
                        error = std::current_exception();
                    }
                }
            }
        }
        if (error) {
            std::rethrow_exception(error);
        }

        rotation = std::move(new_rotation);
        partitions = std::move(new_partitions);
        metric = dist;
        n_trees = n_trees_;
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
        if (k <= 0 || k > n_corpus || !out) {
            throw std::invalid_argument("k must be in [1, corpus size]");
        }
        if (!std::isfinite(vote_threshold) || vote_threshold < 0 || vote_threshold > 1) {
            throw std::invalid_argument("votes_required must be in [0, 1]");
        }
        if (dist != metric) {
            throw std::invalid_argument("Search metric must match build metric");
        }

        auto& scratch = query_scratch();
        accumulate(data, scratch);
        auto& elected = scratch.elected;
        elected.clear();

        for (uint32_t id : scratch.touched) {
            if (scratch.votes[id] >= vote_threshold) {
                elected.push_back(id);
            }
        }

        if (out_n_elected) {
            *out_n_elected = static_cast<int>(elected.size());
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

    void grow(
        int,
        int,
        const Eigen::Ref<const UIntRowMatrix>&,
        const Eigen::Ref<const RowMatrix>&,
        float = -1.0,
        int = 1
    ) override {
        throw std::invalid_argument("Use IVF::build with n_clusters and subspace_dim");
    }

  protected:
    struct Cell {
        std::vector<LabelScore> labels;
        int training_size = 0;
    };

    struct Partition {
        RowMatrix centroids;
        Eigen::VectorXf norms;
        std::vector<Cell> cells;
        int iterations = 0;
    };

    RowMatrix rotation;
    std::vector<Partition> partitions;

  private:
    Distance metric = L2;

    struct QueryScratch {
        Eigen::VectorXf projected;
        Eigen::VectorXf distances;
        std::vector<float> votes;
        std::vector<uint32_t> touched;
        std::vector<uint32_t> elected;
    };

    static QueryScratch& query_scratch() {
        static thread_local QueryScratch scratch;
        return scratch;
    }

    void validate(
        const Eigen::Ref<const UIntRowMatrix>& knn,
        const Eigen::Ref<const RowMatrix>& train,
        int n_trees_,
        int n_clusters,
        int subspace_dim,
        Distance dist
    ) const {
        if (!empty()) {
            throw std::logic_error("The index has already been built");
        }
        if (n_trees_ <= 0 || subspace_dim <= 0 || int64_t(n_trees_) * subspace_dim != dim) {
            throw std::invalid_argument("IVF requires n_trees * subspace_dim == dimension");
        }
        if (n_clusters <= 0 || n_clusters > n_corpus) {
            throw std::invalid_argument("n_clusters must be in [1, corpus size]");
        }
        if (dist != L2 && dist != IP) {
            throw std::invalid_argument("dist must be L2 or IP");
        }
        if (train.rows() <= 0 || train.rows() > std::numeric_limits<int>::max() ||
            train.cols() != dim || knn.rows() != train.rows() || knn.cols() <= 0 ||
            knn.cols() > n_corpus) {
            throw std::invalid_argument("Invalid training features or neighbor matrix shape");
        }

        if (!corpus.allFinite() || !train.allFinite()) {
            throw std::invalid_argument("Corpus and training features must be finite");
        }
        std::vector<uint32_t> labels(knn.cols());
        for (Eigen::Index row = 0; row < knn.rows(); ++row) {
            std::copy_n(knn.row(row).data(), knn.cols(), labels.begin());
            miniselect::pdqsort_branchless(labels.begin(), labels.end());
            if (labels.back() >= uint32_t(n_corpus) ||
                std::adjacent_find(labels.begin(), labels.end()) != labels.end()) {
                throw std::invalid_argument("knn rows must contain distinct valid corpus IDs");
            }
        }
    }

    RowMatrix make_rotation(std::mt19937& generator) const {
        std::normal_distribution<float> normal;
        RowMatrix gaussian(dim, dim);
        for (Eigen::Index i = 0; i < gaussian.size(); ++i) {
            gaussian.data()[i] = normal(generator);
        }

        Eigen::HouseholderQR<RowMatrix> qr(gaussian);
        RowMatrix new_rotation = qr.householderQ() * RowMatrix::Identity(dim, dim);
        // Correct QR's diagonal signs to obtain a Haar-distributed orthogonal matrix.
        for (int col = 0; col < dim; ++col) {
            if (qr.matrixQR()(col, col) < 0) {
                new_rotation.col(col) *= -1.0f;
            }
        }

        return new_rotation;
    }

    static void make_cells(
        Partition& partition,
        const std::vector<int>& assignments,
        const Eigen::Ref<const UIntRowMatrix>& knn
    ) {
        const int n_clusters = static_cast<int>(partition.centroids.rows());
        partition.cells.resize(n_clusters);
        std::vector<std::vector<int>> rows(n_clusters);
        for (Eigen::Index row = 0; row < knn.rows(); ++row) {
            rows[assignments[row]].push_back(static_cast<int>(row));
        }

        for (int cluster = 0; cluster < n_clusters; ++cluster) {
            auto& cell = partition.cells[cluster];
            cell.training_size = static_cast<int>(rows[cluster].size());
            std::unordered_map<uint32_t, uint32_t> counts;
            for (int row : rows[cluster]) {
                for (Eigen::Index col = 0; col < knn.cols(); ++col) {
                    ++counts[knn(row, col)];
                }
            }

            cell.labels.reserve(counts.size());
            for (const auto& entry : counts) {
                const float probability = float(entry.second) / float(cell.training_size);
                cell.labels.push_back({entry.first, probability});
            }

            miniselect::pdqsort_branchless(
                cell.labels.begin(), cell.labels.end(), [](const auto& a, const auto& b) {
                    return a.id < b.id;
                }
            );
        }
    }

    static RowMatrix project(const Eigen::Ref<const RowMatrix>& data, const RowMatrix& rotation) {
        RowMatrix result(data.rows(), data.cols());
        std::exception_ptr error;
#ifdef _OPENMP
#pragma omp parallel for if (data.rows() > 1024)
#endif
        for (Eigen::Index begin = 0; begin < data.rows(); begin += 1024) {
            try {
                const Eigen::Index rows = std::min<Eigen::Index>(1024, data.rows() - begin);
                result.middleRows(begin, rows).noalias() =
                    data.middleRows(begin, rows) * rotation.transpose();
            } catch (...) {
#ifdef _OPENMP
#pragma omp critical(ivf_projection_error)
#endif
                {
                    if (!error) {
                        error = std::current_exception();
                    }
                }
            }
        }
        if (error) {
            std::rethrow_exception(error);
        }
        if (!result.allFinite()) {
            throw std::invalid_argument("IVF rotation overflowed");
        }
        return result;
    }

    void accumulate(const float* q, QueryScratch& scratch) const {
        if (empty()) {
            throw std::logic_error("Cannot query before building index");
        }
        if (!q || !Eigen::Map<const Eigen::VectorXf>(q, dim).allFinite()) {
            throw std::invalid_argument("Query must be finite");
        }
        for (uint32_t id : scratch.touched) {
            scratch.votes[id] = 0;
        }

        scratch.touched.clear();
        scratch.votes.resize(n_corpus, 0);
        scratch.projected.noalias() = rotation * Eigen::Map<const Eigen::VectorXf>(q, dim);

        const int subspace_dim = dim / n_trees;
        for (int tree = 0; tree < n_trees; ++tree) {
            const auto& partition = partitions[tree];
            const Eigen::Index offset = Eigen::Index(tree) * subspace_dim;
            scratch.distances.noalias() =
                partition.centroids * scratch.projected.segment(offset, subspace_dim);
            scratch.distances = partition.norms - 2.0f * scratch.distances;
            if (!scratch.distances.allFinite()) {
                throw std::invalid_argument("IVF query distances overflowed");
            }

            Eigen::Index cluster;
            scratch.distances.minCoeff(&cluster);
            for (const auto& label : partition.cells[cluster].labels) {
                if (scratch.votes[label.id] == 0) {
                    scratch.touched.push_back(label.id);
                }
                scratch.votes[label.id] += label.score;
            }
        }
        for (uint32_t id : scratch.touched) {
            scratch.votes[id] /= float(n_trees);
        }
    }
};
