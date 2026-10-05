#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "neighbor-query.h"

namespace mlann_detail {
namespace exact_search_detail {

using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <OneToManyMetric metric>
struct Heap {
    ScoredCandidate* items;
    int keep;
    int used = 0;

    void add(uint32_t id, float score) {
        const NeighborTopKOrder<metric> better;
        const ScoredCandidate item{score, id};
        if (used < keep) {
            items[used++] = item;
            if (used == keep)
                std::make_heap(items, items + keep, better);
            return;
        }
        if (!better(item, items[0]))
            return;
        int hole = 0;
        size_t child = 1;
        while (child < size_t(keep)) {
            if (child + 1 < size_t(keep) && better(items[child], items[child + 1]))
                ++child;
            if (!better(item, items[child]))
                break;
            items[hole] = items[child];
            hole = int(child);
            child = 2 * size_t(hole) + 1;
        }
        items[hole] = item;
    }
};

template <OneToManyMetric metric>
inline void write_result(ScoredCandidate* heap, int keep, int k, int* out, float* distances) {
    miniselect::pdqsort_branchless(heap, heap + keep, NeighborTopKOrder<metric>{});
    for (int j = 0; j < k; ++j) {
        out[j] = j < keep ? int(heap[j].label) : -1;
        if (distances) {
            if (j >= keep)
                distances[j] = -1.f;
            else if constexpr (metric == OneToManyMetric::L2)
                distances[j] = std::sqrt(heap[j].score);
            else
                distances[j] = heap[j].score;
        }
    }
}

template <OneToManyMetric metric>
void search(
    const float* data,
    int n,
    int dim,
    const float* queries,
    int nq,
    int k,
    int* out,
    float* distances
) {
    if (!nq)
        return;
    constexpr int query_block = 128;
    constexpr int corpus_block = 4096;
    const Eigen::Map<const Matrix> x(data, n, dim), q(queries, nq, dim);
    const int keep = std::min(k, n);
    const int query_blocks = 1 + (nq - 1) / query_block;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    threads = int(std::min<int64_t>(threads, int64_t(query_blocks) * n));
    const int shards = std::min(n, (threads + query_blocks - 1) / query_blocks);

    // Small query batches also parallelize over the corpus. Only those batches
    // need shared partial heaps; their size is bounded by O(threads * query_block * k).
    // Large batches write each completed query block directly into the output.
    std::vector<ScoredCandidate> partial(shards > 1 ? size_t(nq) * shards * keep : 0);
    std::vector<float> x_norms;
    if constexpr (metric == OneToManyMetric::L2) {
        x_norms.resize(n);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int i = 0; i < n; ++i)
            x_norms[i] = x.row(i).squaredNorm();
    }

    std::exception_ptr error;
    std::atomic<bool> failed{false};
    const auto record_error = [&] {
#pragma omp critical(mlann_exact_search_error)
        {
            if (!error)
                error = std::current_exception();
        }
        failed.store(true, std::memory_order_relaxed);
    };
#pragma omp parallel num_threads(threads)
    {
        // Column-major C x Q keeps the score scan for each query contiguous.
        Eigen::MatrixXf scores;
        Eigen::VectorXf q_norms;
        std::vector<ScoredCandidate> local;
        std::vector<Heap<metric>> heaps;
#pragma omp for schedule(static)
        for (int task = 0; task < query_blocks * shards; ++task) {
            if (failed.load(std::memory_order_relaxed))
                continue;
            try {
                const int qi = (task / shards) * query_block;
                const int qs = std::min(query_block, nq - qi);
                const int shard = task % shards;
                const int begin = int(int64_t(n) * shard / shards);
                const int end = int(int64_t(n) * (shard + 1) / shards);
                if (shards == 1)
                    local.resize(size_t(qs) * keep);
                heaps.clear();
                for (int j = 0; j < qs; ++j) {
                    ScoredCandidate* storage =
                        shards == 1 ? local.data() + size_t(j) * keep
                                    : partial.data() + (size_t(qi + j) * shards + shard) * keep;
                    heaps.push_back({storage, std::min(keep, end - begin), 0});
                }
                if constexpr (metric == OneToManyMetric::L2)
                    q_norms = q.middleRows(qi, qs).rowwise().squaredNorm();
                for (int ci = begin; ci < end;) {
                    const int cs = std::min(corpus_block, end - ci);
                    scores.resize(cs, qs);
                    scores.noalias() = x.middleRows(ci, cs) * q.middleRows(qi, qs).transpose();
                    for (int j = 0; j < qs; ++j) {
                        const float* column = scores.col(j).data();
                        for (int i = 0; i < cs; ++i) {
                            float value = column[i];
                            if constexpr (metric == OneToManyMetric::L2)
                                value = std::max(0.f, x_norms[ci + i] + q_norms[j] - 2.f * value);
                            heaps[j].add(uint32_t(ci + i), value);
                        }
                    }
                    ci += cs;
                }
                if (shards == 1) {
                    for (int j = 0; j < qs; ++j) {
                        const size_t offset = size_t(qi + j) * k;
                        write_result<metric>(
                            heaps[j].items,
                            keep,
                            k,
                            out + offset,
                            distances ? distances + offset : nullptr
                        );
                    }
                }
            } catch (...) {
                // Exceptions must not escape an OpenMP worker.
                record_error();
            }
        }
        if (shards > 1) {
#pragma omp for schedule(static)
            for (int qi = 0; qi < nq; ++qi) {
                if (failed.load(std::memory_order_relaxed))
                    continue;
                try {
                    local.resize(keep);
                    Heap<metric> heap{local.data(), keep, 0};
                    for (int shard = 0; shard < shards; ++shard) {
                        const int begin = int(int64_t(n) * shard / shards);
                        const int end = int(int64_t(n) * (shard + 1) / shards);
                        const ScoredCandidate* source =
                            partial.data() + (size_t(qi) * shards + shard) * keep;
                        for (int j = 0; j < std::min(keep, end - begin); ++j)
                            heap.add(source[j].label, source[j].score);
                    }
                    const size_t offset = size_t(qi) * k;
                    write_result<metric>(
                        local.data(),
                        keep,
                        k,
                        out + offset,
                        distances ? distances + offset : nullptr
                    );
                } catch (...) {
                    record_error();
                }
            }
        }
    }
    if (error)
        std::rethrow_exception(error);
}

} // namespace exact_search_detail

inline void exact_search_batch(
    const float* data,
    int n,
    int dim,
    const float* queries,
    int nq,
    int k,
    OneToManyMetric metric,
    int* out,
    float* distances = nullptr
) {
    if (metric == OneToManyMetric::IP)
        exact_search_detail::search<OneToManyMetric::IP>(
            data, n, dim, queries, nq, k, out, distances
        );
    else
        exact_search_detail::search<OneToManyMetric::L2>(
            data, n, dim, queries, nq, k, out, distances
        );
}

} // namespace mlann_detail
