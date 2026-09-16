#pragma once

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../mlann.h"

namespace mlann_detail {

class KMeans {
  public:
    static constexpr int default_max_iterations = 10;

    static std::vector<int> assign(
        const Eigen::Ref<const RowMatrix>& data,
        const Eigen::Ref<const RowMatrix>& centroids,
        bool spherical = false,
        bool parallel = true
    ) {
        if (centroids.rows() == 0 || centroids.cols() != data.cols())
            throw std::invalid_argument("Invalid clustering centroids");
        const int n_clusters = static_cast<int>(centroids.rows());
        Eigen::RowVectorXf norms;
        if (!spherical)
            norms = centroids.rowwise().squaredNorm().transpose();
        // Bound each worker's similarity matrix.
        const int block_rows =
            std::clamp(
                static_cast<int>((1 << 20) / (sizeof(float) * size_t(n_clusters))), 128, 1024
            ) /
            64 * 64;
        std::vector<int> assignments(data.rows());
        const auto assign_block = [&](Eigen::Index begin) {
            const Eigen::Index rows = std::min<Eigen::Index>(block_rows, data.rows() - begin);
            RowMatrix distances = data.middleRows(begin, rows) * centroids.transpose();
            if (!spherical) {
                distances *= -2.0f;
                distances.rowwise() += norms;
            }
            if (!distances.allFinite())
                throw std::invalid_argument("Clustering centroid scores overflowed");
            for (Eigen::Index row = 0; row < rows; ++row) {
                Eigen::Index cluster;
                if (spherical)
                    distances.row(row).maxCoeff(&cluster);
                else
                    distances.row(row).minCoeff(&cluster);
                assignments[begin + row] = static_cast<int>(cluster);
            }
        };

        // Tree ensembles already parallelize their independent fits.
        if (!parallel || data.rows() <= block_rows) {
            for (Eigen::Index begin = 0; begin < data.rows(); begin += block_rows)
                assign_block(begin);
            return assignments;
        }

        std::exception_ptr error;
#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
        for (Eigen::Index begin = 0; begin < data.rows(); begin += block_rows) {
            try {
                assign_block(begin);
            } catch (...) {
#ifdef _OPENMP
#pragma omp critical(clustering_assignment_error)
#endif
                {
                    if (!error)
                        error = std::current_exception();
                }
            }
        }
        if (error)
            std::rethrow_exception(error);
        return assignments;
    }

    static RowMatrix fit(
        const Eigen::Ref<const RowMatrix>& data,
        const std::vector<int>& seeds,
        int& iterations,
        bool spherical = false,
        int max_iterations = default_max_iterations,
        bool parallel = true
    ) {
        const int n_clusters = static_cast<int>(seeds.size());
        RowMatrix centroids(n_clusters, data.cols());
        for (int cluster = 0; cluster < n_clusters; ++cluster) {
            if (seeds[cluster] < 0 || seeds[cluster] >= data.rows())
                throw std::invalid_argument("Invalid clustering seed row");
            centroids.row(cluster) = data.row(seeds[cluster]);
            if (spherical)
                centroids.row(cluster).stableNormalize();
        }

        std::vector<int> assignments;
        std::vector<int> counts;
        iterations =
            refine(data, centroids, assignments, counts, spherical, max_iterations, parallel);
        return centroids;
    }

    // Assignments/counts describe the last centroid update, without an extra final probe.
    // Spherical initial centroids must already be normalized (zero vectors are allowed).
    static int refine(
        const Eigen::Ref<const RowMatrix>& data,
        RowMatrix& centroids,
        std::vector<int>& assignments,
        std::vector<int>& counts,
        bool spherical = false,
        int max_iterations = default_max_iterations,
        bool parallel = true
    ) {
        if (max_iterations <= 0 || data.rows() == 0 || data.cols() == 0 || centroids.rows() == 0 ||
            centroids.cols() != data.cols())
            throw std::invalid_argument("Invalid clustering dimensions or iteration limit");
        const int n_clusters = static_cast<int>(centroids.rows());
        assignments.clear();
        counts.resize(n_clusters);
        RowMatrix sums(n_clusters, data.cols());

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            auto next = assign(data, centroids, spherical, parallel);
            if (next == assignments)
                return iteration;
            assignments = std::move(next);
            sums.setZero();
            std::fill(counts.begin(), counts.end(), 0);
            for (Eigen::Index row = 0; row < data.rows(); ++row) {
                const int cluster = assignments[row];
                sums.row(cluster) += data.row(row);
                ++counts[cluster];
            }
            for (int cluster = 0; cluster < n_clusters; ++cluster) {
                // Keeping an empty centroid preserves Lloyd descent and permits
                // convergence even when fewer than K distinct points exist.
                if (counts[cluster] > 0) {
                    if (spherical) {
                        centroids.row(cluster) = sums.row(cluster);
                        centroids.row(cluster).stableNormalize();
                    } else {
                        centroids.row(cluster) = sums.row(cluster) / float(counts[cluster]);
                    }
                }
            }
            if (!centroids.allFinite())
                throw std::invalid_argument("Clustering centroid means overflowed");
        }
        return max_iterations;
    }
};

} // namespace mlann_detail
