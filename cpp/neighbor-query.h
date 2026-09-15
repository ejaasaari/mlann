#pragma once

#include "one-to-many.h"

namespace mlann_detail {
// Same arithmetic and output buffers as the native scorer. Smaller batches and
// look-ahead loads reduce stalls while reading full corpus vectors.
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
#define MLANN_NEIGHBOR_INLINE inline __attribute__((always_inline))
#define MLANN_NEIGHBOR_REPEAT_4(OP) OP(0) OP(1) OP(2) OP(3)
struct NeighborOneToManyKernel {
 private:
  static MLANN_NEIGHBOR_INLINE __m512 multiply_add(const __m512 sum, const __m512 left,
                                                     const __m512 right) {
#if defined(__FMA__)
    return _mm512_fmadd_ps(left, right, sum);
#else
    return _mm512_add_ps(sum, _mm512_mul_ps(left, right));
#endif
  }

  template <OneToManyMetric metric>
  static MLANN_NEIGHBOR_INLINE float run_one(const float *query, const float *row,
                                               const std::size_t dim) {
    __m512 sum = _mm512_setzero_ps();
    std::size_t j = 0;
    if constexpr (metric == OneToManyMetric::L2) {
      for (; j + 16 <= dim; j += 16) {
        const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(query + j), _mm512_loadu_ps(row + j));
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
      for (; j < dim; ++j) scalar += query[j] * row[j];
      return scalar;
    }
  }

 public:
  template <OneToManyMetric metric, typename Output>
  static MLANN_NEIGHBOR_INLINE void run(const float *query, const float *data,
                                          const std::size_t dim, const std::uint32_t *indices,
                                          const std::size_t count, Output output) {
    std::size_t candidate = 0;
    for (; count - candidate >= 4; candidate += 4) {
      for (std::size_t future = candidate + 32;
           future < count && future < candidate + 32 + 4; ++future) {
        const float *row = data + static_cast<std::size_t>(indices[future]) * dim;
        for (std::size_t j = 0; j < dim; j += 16) __builtin_prefetch(row + j, 0, 2);
      }

#define MLANN_NEIGHBOR_AVX512_ROW(i) \
  const float *const row##i = data + static_cast<std::size_t>(indices[candidate + i]) * dim;
      MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_ROW)
#undef MLANN_NEIGHBOR_AVX512_ROW

#define MLANN_NEIGHBOR_AVX512_SUM(i) __m512 sum##i = _mm512_setzero_ps();
      MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_SUM)
#undef MLANN_NEIGHBOR_AVX512_SUM

      std::size_t j = 0;
      if constexpr (metric == OneToManyMetric::L2) {
        for (; j + 16 <= dim; j += 16) {
          const __m512 query_vector = _mm512_loadu_ps(query + j);
#define MLANN_NEIGHBOR_AVX512_L2(i)                                                     \
  const __m512 diff##i = _mm512_sub_ps(query_vector, _mm512_loadu_ps(row##i + j)); \
  sum##i = multiply_add(sum##i, diff##i, diff##i);
          MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_L2)
#undef MLANN_NEIGHBOR_AVX512_L2
        }
      } else {
        for (; j + 16 <= dim; j += 16) {
          const __m512 query_vector = _mm512_loadu_ps(query + j);
#define MLANN_NEIGHBOR_AVX512_IP(i) \
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
#define MLANN_NEIGHBOR_AVX512_TAIL_L2(i)              \
  const float diff##i = query_value - row##i[j]; \
  scalar##i += diff##i * diff##i;
          MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_TAIL_L2)
#undef MLANN_NEIGHBOR_AVX512_TAIL_L2
        }
#define MLANN_NEIGHBOR_AVX512_STORE_L2(i) output[candidate + i] = scalar##i;
        MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_STORE_L2)
#undef MLANN_NEIGHBOR_AVX512_STORE_L2
      } else {
        for (; j < dim; ++j) {
          const float query_value = query[j];
#define MLANN_NEIGHBOR_AVX512_TAIL_IP(i) scalar##i += query_value * row##i[j];
          MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_TAIL_IP)
#undef MLANN_NEIGHBOR_AVX512_TAIL_IP
        }
#define MLANN_NEIGHBOR_AVX512_STORE_IP(i) output[candidate + i] = scalar##i;
        MLANN_NEIGHBOR_REPEAT_4(MLANN_NEIGHBOR_AVX512_STORE_IP)
#undef MLANN_NEIGHBOR_AVX512_STORE_IP
      }
    }

    for (; candidate < count; ++candidate) {
      const float *const row = data + static_cast<std::size_t>(indices[candidate]) * dim;
      output[candidate] = run_one<metric>(query, row, dim);
    }
  }
};

#undef MLANN_NEIGHBOR_REPEAT_4
#undef MLANN_NEIGHBOR_INLINE
#endif

template <typename Output>
inline void compute_neighbor_one_to_many(const float *query, const float *data,
                                         size_t dim, const uint32_t *indices,
                                         size_t count, OneToManyMetric metric, Output output) {
#if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
  if (metric == OneToManyMetric::L2)
    NeighborOneToManyKernel::run<OneToManyMetric::L2>(query, data, dim, indices, count, output);
  else
    NeighborOneToManyKernel::run<OneToManyMetric::IP>(query, data, dim, indices, count, output);
#else
  compute_one_to_many(query, data, dim, indices, count, metric, output);
#endif
}

inline void compute_neighbor_scores(const float *query, const float *data,
                                    size_t dim, const uint32_t *indices,
                                    size_t count, OneToManyMetric metric,
                                    StridedFloatOutput output) {
  compute_neighbor_one_to_many(query, data, dim, indices, count, metric, output);
}
}  // namespace mlann_detail
