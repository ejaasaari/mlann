#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

#include "distance.h"
#include "miniselect/pdqselect.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> UIntRowMatrix;

class MLANN {
 public:
  MLANN(const float *corpus_, int n_corpus_, int dim_)
      : corpus(Eigen::Map<const RowMatrix>(corpus_, n_corpus_, dim_)),
        n_corpus(n_corpus_),
        dim(dim_) {}

  virtual ~MLANN() = default;

  virtual void grow(int n_trees_, int depth_, const Eigen::Ref<const UIntRowMatrix> &knn_,
                    const Eigen::Ref<const RowMatrix> &train_, float density_ = -1.0, int b_ = 1) {}

  virtual void query(const float *data, int k, float vote_threshold, int *out, Distance dist = L2,
                     float *out_distances = nullptr, int *out_n_elected = nullptr) const {}

  void query(const Eigen::Ref<const Eigen::RowVectorXf> &q, int k, float vote_threshold, int *out,
             Distance dist = L2, float *out_distances = nullptr,
             int *out_n_elected = nullptr) const {
    query(q.data(), k, vote_threshold, out, dist, out_distances, out_n_elected);
  }

  static void exact_knn(const float *q_data, const float *X_data, int n_corpus, int dim, int k,
                        int *out, Distance dist = L2, float *out_distances = nullptr) {
    const Eigen::Map<const RowMatrix> corpus(X_data, n_corpus, dim);
    const Eigen::Map<const Eigen::RowVectorXf> q(q_data, dim);

    Eigen::VectorXf distances(n_corpus);
    if (dist == L2) {
      for (int i = 0; i < n_corpus; ++i) {
        distances(i) = squared_euclidean(corpus.row(i).data(), q.data(), dim);
      }
    } else {
      for (int i = 0; i < n_corpus; ++i) {
        distances(i) = dot_product(corpus.row(i).data(), q.data(), dim);
      }
    }

    if (k == 1) {
      Eigen::MatrixXf::Index index;

      if (dist == L2) {
        distances.minCoeff(&index);
        out[0] = index;
        if (out_distances) out_distances[0] = std::sqrt(distances(index));
      } else {
        distances.maxCoeff(&index);
        out[0] = index;
        if (out_distances) out_distances[0] = distances(index);
      }

      return;
    }

    Eigen::VectorXi idx(n_corpus);
    std::iota(idx.data(), idx.data() + n_corpus, 0);

    if (dist == L2) {
      miniselect::pdqpartial_sort_branchless(
          idx.data(), idx.data() + k, idx.data() + n_corpus,
          [&distances](int i1, int i2) { return distances(i1) < distances(i2); });
    } else {
      miniselect::pdqpartial_sort_branchless(
          idx.data(), idx.data() + k, idx.data() + n_corpus,
          [&distances](int i1, int i2) { return distances(i1) > distances(i2); });
    }

    for (int i = 0; i < k; ++i) out[i] = idx(i);

    if (out_distances) {
      if (dist == L2) {
        for (int i = 0; i < k; ++i) out_distances[i] = std::sqrt(distances(idx(i)));
      } else {
        for (int i = 0; i < k; ++i) out_distances[i] = distances(idx(i));
      }
    }
  }

  static void exact_knn(const Eigen::Ref<const Eigen::RowVectorXf> &q,
                        const Eigen::Ref<const RowMatrix> &corpus, int k, int *out,
                        Distance dist = L2, float *out_distances = nullptr) {
    MLANN::exact_knn(q.data(), corpus.data(), corpus.rows(), corpus.cols(), k, out, dist,
                     out_distances);
  }

  void exact_knn(const float *q, int k, int *out, Distance dist = L2,
                 float *out_distances = nullptr) const {
    MLANN::exact_knn(q, corpus.data(), n_corpus, dim, k, out, dist, out_distances);
  }

  void exact_knn(const Eigen::Ref<const Eigen::RowVectorXf> &q, int k, int *out, Distance dist = L2,
                 float *out_distances = nullptr) const {
    MLANN::exact_knn(q.data(), corpus.data(), n_corpus, dim, k, out, dist, out_distances);
  }

  bool empty() const { return n_trees == 0; }

 protected:
  void exact_knn(const Eigen::Map<const Eigen::RowVectorXf> &q, int k,
                 const std::vector<uint32_t> &indices, int *out, Distance dist = L2,
                 float *out_distances = nullptr) const {
    if (indices.empty()) {
      for (int i = 0; i < k; ++i) out[i] = -1;
      if (out_distances) {
        for (int i = 0; i < k; ++i) out_distances[i] = -1;
      }
      return;
    }

    int n_elected = indices.size();
    Eigen::VectorXf distances(n_elected);

    if (dist == L2) {
      for (int i = 0; i < n_elected; ++i) {
        distances(i) = squared_euclidean(corpus.row(indices[i]).data(), q.data(), dim);
      }
    } else {
      for (int i = 0; i < n_elected; ++i) {
        distances(i) = dot_product(corpus.row(indices[i]).data(), q.data(), dim);
      }
    }

    if (k == 1) {
      Eigen::MatrixXf::Index index;

      if (dist == L2) {
        distances.minCoeff(&index);
        out[0] = indices[index];
        if (out_distances) out_distances[0] = std::sqrt(distances(index));
      } else {
        distances.maxCoeff(&index);
        out[0] = indices[index];
        if (out_distances) out_distances[0] = distances(index);
      }

      return;
    }

    int n_to_sort = n_elected > k ? k : n_elected;
    Eigen::VectorXi idx(n_elected);
    std::iota(idx.data(), idx.data() + n_elected, 0);

    if (dist == L2) {
      miniselect::pdqpartial_sort_branchless(
          idx.data(), idx.data() + n_to_sort, idx.data() + n_elected,
          [&distances](int i1, int i2) { return distances(i1) < distances(i2); });
    } else {
      miniselect::pdqpartial_sort_branchless(
          idx.data(), idx.data() + n_to_sort, idx.data() + n_elected,
          [&distances](int i1, int i2) { return distances(i1) > distances(i2); });
    }

    for (int i = 0; i < k; ++i) out[i] = i < n_elected ? indices[idx(i)] : -1;

    if (out_distances) {
      if (dist == L2) {
        for (int i = 0; i < k; ++i) {
          out_distances[i] = i < n_elected ? std::sqrt(distances(idx(i))) : -1;
        }
      } else {
        for (int i = 0; i < k; ++i) {
          out_distances[i] = i < n_elected ? distances(idx(i)) : -1;
        }
      }
    }
  }

  const Eigen::Map<const RowMatrix> corpus;  // corpus from which nearest neighbors are searched
  Eigen::MatrixXf split_points;              // all split points in all the trees
  Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      split_dimensions;  // all split dimensions in all the trees
  std::vector<std::vector<std::vector<uint32_t>>> labels_all;
  std::vector<std::vector<std::vector<float>>> votes_all;

  const int n_corpus;    // size of corpus
  const int dim;         // dimension of data
  int n_trees = 0;       // number of RP-trees
  int depth = 0;         // depth of an RP-tree with median split
  float density = -1.0;  // expected ratio of non-zero components in a projection matrix
  int n_pool = 0;        // amount of random vectors needed for all the RP-trees
  int n_array = 0;       // length of the one RP-tree as array
  int b = 0;
  int n_inner_nodes = 0;
  int n_leaves = 0;
};
