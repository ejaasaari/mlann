#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#include "miniselect/pdqselect.h"
#include "one-to-many.h"

namespace mlann_detail {
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
#define MLANN_NEIGHBOR_INLINE inline __attribute__((always_inline))
#define MLANN_NEIGHBOR_REPEAT_4(OP) OP(0) OP(1) OP(2) OP(3)
struct NeighborOneToManyKernel {
  private:
    static MLANN_NEIGHBOR_INLINE __m512
    multiply_add(const __m512 sum, const __m512 left, const __m512 right) {
#if defined(__FMA__)
        return _mm512_fmadd_ps(left, right, sum);
#else
        return _mm512_add_ps(sum, _mm512_mul_ps(left, right));
#endif
    }

    template <OneToManyMetric metric>
    static MLANN_NEIGHBOR_INLINE float run_one(
        const float* query,
        const float* row,
        const std::size_t dim
    ) {
        __m512 sum = _mm512_setzero_ps();
        std::size_t j = 0;
        if constexpr (metric == OneToManyMetric::L2) {
            for (; j + 16 <= dim; j += 16) {
                const __m512 diff =
                    _mm512_sub_ps(_mm512_loadu_ps(query + j), _mm512_loadu_ps(row + j));
                sum = multiply_add(sum, diff, diff);
            }
        } else {
            for (; j + 16 <= dim; j += 16) {
                sum = multiply_add(sum, _mm512_loadu_ps(query + j), _mm512_loadu_ps(row + j));
            }
        }

        float scalar = _mm512_reduce_add_ps(sum);
        if constexpr (metric == OneToManyMetric::L2) {
            for (; j < dim; ++j) {
                const float diff = query[j] - row[j];
                scalar += diff * diff;
            }
            return scalar;
        } else {
            for (; j < dim; ++j)
                scalar += query[j] * row[j];
            return scalar;
        }
    }

  public:
    template <OneToManyMetric metric, typename Output>
    static MLANN_NEIGHBOR_INLINE void run(
        const float* query,
        const float* data,
        const std::size_t dim,
        const std::uint32_t* indices,
        const std::size_t count,
        Output output
    ) {
        std::size_t candidate = 0;
        for (; count - candidate >= 4; candidate += 4) {
            for (std::size_t future = candidate + 32; future < count && future < candidate + 32 + 4;
                 ++future) {
                const float* row = data + static_cast<std::size_t>(indices[future]) * dim;
                for (std::size_t j = 0; j < dim; j += 16)
                    __builtin_prefetch(row + j, 0, 2);
            }

#define MLANN_NEIGHBOR_AVX512_ROW(i)                                                               \
    const float* const row##i = data + static_cast<std::size_t>(indices[candidate + (i)]) * dim;
            MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_ROW)
#undef MLANN_NEIGHBOR_AVX512_ROW

#define MLANN_NEIGHBOR_AVX512_SUM(i) __m512 sum##i = _mm512_setzero_ps();
            MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_SUM)
#undef MLANN_NEIGHBOR_AVX512_SUM

            std::size_t j = 0;
            if constexpr (metric == OneToManyMetric::L2) {
                for (; j + 16 <= dim; j += 16) {
                    const __m512 query_vector = _mm512_loadu_ps(query + j);
#define MLANN_NEIGHBOR_AVX512_L2(i)                                                                \
    const __m512 diff##i = _mm512_sub_ps(query_vector, _mm512_loadu_ps(row##i + j));               \
    sum##i = multiply_add(sum##i, diff##i, diff##i);
                    MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_L2)
#undef MLANN_NEIGHBOR_AVX512_L2
                }
            } else {
                for (; j + 16 <= dim; j += 16) {
                    const __m512 query_vector = _mm512_loadu_ps(query + j);
#define MLANN_NEIGHBOR_AVX512_IP(i)                                                                \
    sum##i = multiply_add(sum##i, query_vector, _mm512_loadu_ps(row##i + j));
                    MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_IP)
#undef MLANN_NEIGHBOR_AVX512_IP
                }
            }

#define MLANN_NEIGHBOR_AVX512_REDUCE(i) float scalar##i = _mm512_reduce_add_ps(sum##i);
            MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_REDUCE)
#undef MLANN_NEIGHBOR_AVX512_REDUCE

            if constexpr (metric == OneToManyMetric::L2) {
                for (; j < dim; ++j) {
                    const float query_value = query[j];
#define MLANN_NEIGHBOR_AVX512_TAIL_L2(i)                                                           \
    const float diff##i = query_value - row##i[j];                                                 \
    scalar##i += diff##i * diff##i;
                    MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_TAIL_L2)
#undef MLANN_NEIGHBOR_AVX512_TAIL_L2
                }
#define MLANN_NEIGHBOR_AVX512_STORE_L2(i) output[candidate + (i)] = scalar##i;
                MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_STORE_L2)
#undef MLANN_NEIGHBOR_AVX512_STORE_L2
            } else {
                for (; j < dim; ++j) {
                    const float query_value = query[j];
#define MLANN_NEIGHBOR_AVX512_TAIL_IP(i) scalar##i += query_value * row##i[j];
                    MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_TAIL_IP)
#undef MLANN_NEIGHBOR_AVX512_TAIL_IP
                }
#define MLANN_NEIGHBOR_AVX512_STORE_IP(i) output[candidate + (i)] = scalar##i;
                MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_STORE_IP)
#undef MLANN_NEIGHBOR_AVX512_STORE_IP
            }
        }

        for (; candidate < count; ++candidate) {
            const float* const row = data + static_cast<std::size_t>(indices[candidate]) * dim;
            output[candidate] = run_one<metric>(query, row, dim);
        }
    }
};

#undef MLANN_NEIGHBOR_REPEAT_4
#undef MLANN_NEIGHBOR_INLINE
#endif

template <typename Output>
inline void compute_neighbor_one_to_many(
    const float* query,
    const float* data,
    size_t dim,
    const uint32_t* indices,
    size_t count,
    OneToManyMetric metric,
    Output output
) {
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
    if (metric == OneToManyMetric::L2)
        NeighborOneToManyKernel::run<OneToManyMetric::L2>(query, data, dim, indices, count, output);
    else
        NeighborOneToManyKernel::run<OneToManyMetric::IP>(query, data, dim, indices, count, output);
#else
    compute_one_to_many(query, data, dim, indices, count, metric, output);
#endif
}

inline void compute_neighbor_scores(
    const float* query,
    const float* data,
    size_t dim,
    const uint32_t* indices,
    size_t count,
    OneToManyMetric metric,
    StridedFloatOutput output
) {
    compute_neighbor_one_to_many(query, data, dim, indices, count, metric, output);
}

// A top-k heap bounds score storage independently of the candidate count.
template <OneToManyMetric metric>
struct NeighborTopKOrder {
    bool operator()(const ScoredCandidate& left, const ScoredCandidate& right) const {
        if (left.score == right.score)
            return left.label < right.label;
        if constexpr (metric == OneToManyMetric::IP)
            return left.score > right.score;
        else
            return left.score < right.score;
    }
};

template <OneToManyMetric metric>
struct NeighborTopKState {
    ScoredCandidate* heap;
    const uint32_t* indices;
    size_t keep;
    size_t used = 0;

    void add(size_t index, float score) {
        const NeighborTopKOrder<metric> better;
        const ScoredCandidate item{score, indices[index]};
        if (used < keep) {
            heap[used++] = item;
            if (used == keep)
                std::make_heap(heap, heap + keep, better);
            return;
        }
        if (!better(item, heap[0]))
            return;
        size_t hole = 0, child = 1;
        while (child < keep) {
            if (child + 1 < keep && better(heap[child], heap[child + 1]))
                ++child;
            if (!better(item, heap[child]))
                break;
            heap[hole] = heap[child];
            hole = child;
            child = 2 * hole + 1;
        }
        heap[hole] = item;
    }
};

template <OneToManyMetric metric>
struct NeighborTopKOutput {
    NeighborTopKState<metric>* state;
    struct Slot {
        NeighborTopKState<metric>* state;
        size_t index;
        void operator=(float score) { state->add(index, score); }
    };
    Slot operator[](size_t index) const { return {state, index}; }
};

template <OneToManyMetric metric>
inline void compute_neighbor_topk_impl(
    const float* query,
    const float* data,
    size_t dim,
    const uint32_t* indices,
    size_t count,
    size_t k,
    ScoredCandidate* output
) {
    const size_t keep = std::min(k, count);
    if (!keep)
        return;
    NeighborTopKState<metric> state{output, indices, keep};
    compute_neighbor_one_to_many(
        query, data, dim, indices, count, metric, NeighborTopKOutput<metric>{&state}
    );
    miniselect::pdqsort_branchless(output, output + keep, NeighborTopKOrder<metric>{});
}

inline void compute_neighbor_topk(
    const float* query,
    const float* data,
    size_t dim,
    const uint32_t* indices,
    size_t count,
    size_t k,
    OneToManyMetric metric,
    ScoredCandidate* output
) {
    if (metric == OneToManyMetric::IP)
        compute_neighbor_topk_impl<OneToManyMetric::IP>(
            query, data, dim, indices, count, k, output
        );
    else
        compute_neighbor_topk_impl<OneToManyMetric::L2>(
            query, data, dim, indices, count, k, output
        );
}

// Unique labels within each leaf allow SIMD updates without conflicting writes.
template <bool unit_votes, typename Weight = float>
inline void accumulate_leaf_votes(
    const std::vector<uint32_t>& labels,
    const Weight* weights,
    float* votes,
    float threshold,
    std::vector<uint32_t>& elected
) {
    size_t i = 0;
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
    const __m512 limit = _mm512_set1_ps(threshold);
    const __m512 sentinel = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (; i + 16 <= labels.size(); i += 16) {
        for (size_t j = i + 32; j < labels.size() && j < i + 48; ++j)
            __builtin_prefetch(votes + labels[j], 1, 1);
        const __m512i ids = _mm512_loadu_si512(labels.data() + i);
        __m512 weight;
        if constexpr (unit_votes) {
            weight = _mm512_set1_ps(1.f);
        } else if constexpr (std::is_same_v<Weight, uint16_t>) {
            weight = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + i))
            ));
        } else {
            weight = _mm512_loadu_ps(weights + i);
        }
        const __m512 updated = _mm512_add_ps(_mm512_i32gather_ps(ids, votes, 4), weight);
        unsigned selected = _mm512_cmp_ps_mask(updated, limit, _CMP_GE_OQ);
        _mm512_i32scatter_ps(votes, ids, _mm512_mask_mov_ps(updated, selected, sentinel), 4);
        while (selected) {
            const unsigned lane = __builtin_ctz(selected);
            elected.push_back(labels[i + lane]);
            selected &= selected - 1;
        }
    }
#endif
    for (; i < labels.size(); ++i) {
#if defined(__GNUC__) || defined(__clang__)
        if (i + 32 < labels.size())
            __builtin_prefetch(votes + labels[i + 32], 1, 1);
#endif
        if ((votes[labels[i]] += (unit_votes ? 1.f : weights[i])) >= threshold) {
            elected.push_back(labels[i]);
            votes[labels[i]] = -std::numeric_limits<float>::infinity();
        }
    }
}
template <typename Weight>
inline void accumulate_neighbor_votes(
    const std::vector<uint32_t>& labels,
    const std::vector<Weight>& weights,
    float* votes,
    float threshold,
    std::vector<uint32_t>& elected
) {
    accumulate_leaf_votes<false>(labels, weights.data(), votes, threshold, elected);
}

inline void accumulate_unit_votes(
    const std::vector<uint32_t>& ids,
    float* votes,
    float threshold,
    std::vector<uint32_t>& elected
) {
    accumulate_leaf_votes<true, float>(ids, nullptr, votes, threshold, elected);
}
} // namespace mlann_detail
