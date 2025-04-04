#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

#include "miniselect/pdqselect.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> IntRowMatrix;

class MLANN {
 public:
  MLANN(const float *corpus_, int n_corpus_, int dim_)
      : corpus(Eigen::Map<const RowMatrix>(corpus_, n_corpus_, dim_)),
        n_corpus(n_corpus_),
        dim(dim_) {}

  virtual ~MLANN() = default;

  virtual void grow(int n_trees_, int depth_, const Eigen::Ref<const IntRowMatrix> &knn_,
                    const Eigen::Ref<const RowMatrix> &train_, float density_ = -1.0, int b_ = 1) {}

  virtual void query(const float *data, int k, float vote_threshold, int *out,
                     float *out_distances = nullptr, int *out_n_elected = nullptr) const {}

  void query(const Eigen::Ref<const Eigen::VectorXf> &q, int k, float vote_threshold, int *out,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const {
    query(q.data(), k, vote_threshold, out, out_distances, out_n_elected);
  }

  static void exact_knn(const float *q_data, const float *X_data, int n_corpus, int dim, int k,
                        int *out, float *out_distances = nullptr) {
    const Eigen::Map<const RowMatrix> corpus(X_data, n_corpus, dim);
    const Eigen::Map<const Eigen::VectorXf> q(q_data, dim);

    if (k < 1 || k > n_corpus) {
      throw std::out_of_range(
          "k must be positive and no greater than the sample size of data corpus.");
    }

    Eigen::VectorXf distances(n_corpus);

    for (int i = 0; i < n_corpus; ++i) distances(i) = (corpus.row(i) - q).squaredNorm();

    if (k == 1) {
      Eigen::MatrixXf::Index index;
      distances.minCoeff(&index);
      out[0] = index;

      if (out_distances) out_distances[0] = std::sqrt(distances(index));

      return;
    }

    Eigen::VectorXi idx(n_corpus);
    std::iota(idx.data(), idx.data() + n_corpus, 0);
    miniselect::pdqpartial_sort_branchless(
        idx.data(), idx.data() + k, idx.data() + n_corpus,
        [&distances](int i1, int i2) { return distances(i1) < distances(i2); });

    for (int i = 0; i < k; ++i) out[i] = idx(i);

    if (out_distances) {
      for (int i = 0; i < k; ++i) out_distances[i] = std::sqrt(distances(idx(i)));
    }
  }

  static void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q,
                        const Eigen::Ref<const RowMatrix> &corpus, int k, int *out,
                        float *out_distances = nullptr) {
    MLANN::exact_knn(q.data(), corpus.data(), corpus.rows(), corpus.cols(), k, out, out_distances);
  }

  void exact_knn(const float *q, int k, int *out, float *out_distances = nullptr) const {
    MLANN::exact_knn(q, corpus.data(), n_corpus, dim, k, out, out_distances);
  }

  void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q, int k, int *out,
                 float *out_distances = nullptr) const {
    MLANN::exact_knn(q.data(), corpus.data(), n_corpus, dim, k, out, out_distances);
  }

  bool empty() const { return n_trees == 0; }

 protected:
  std::pair<std::vector<int>, std::vector<float>> count_votes(
      std::vector<int>::iterator leaf_begin, std::vector<int>::iterator leaf_end,
      const Eigen::Ref<const IntRowMatrix> &knn) {
    int k_build = knn.cols();
    std::unordered_map<int, int> votes;
    for (auto it = leaf_begin; it != leaf_end; ++it) {
      const Eigen::VectorXi knn_crnt = knn.row(*it);
      for (int j = 0; j < k_build; ++j) ++votes[knn_crnt(j)];
    }

    std::vector<int> out_labels;
    std::vector<float> out_votes;

    int n_votes = 0;
    for (const auto &v : votes)
      if (v.second >= b) {
        out_labels.push_back(v.first);
        out_votes.push_back(v.second);
        n_votes += v.second;
      }

    for (int i = 0; i < out_votes.size(); ++i) out_votes[i] /= (n_votes * n_trees);

    return {out_labels, out_votes};
  }

  void exact_knn(const Eigen::Map<const Eigen::VectorXf> &q, int k, const std::vector<int> &indices,
                 int *out, float *out_distances = nullptr) const {
    if (indices.empty()) {
      for (int i = 0; i < k; ++i) out[i] = -1;
      if (out_distances) {
        for (int i = 0; i < k; ++i) out_distances[i] = -1;
      }
      return;
    }

    int n_elected = indices.size();
    Eigen::VectorXf distances(n_elected);

    for (int i = 0; i < n_elected; ++i) distances(i) = (corpus.row(indices[i]) - q).squaredNorm();

    if (k == 1) {
      Eigen::MatrixXf::Index index;
      distances.minCoeff(&index);
      out[0] = n_elected ? indices[index] : -1;

      if (out_distances) out_distances[0] = n_elected ? std::sqrt(distances(index)) : -1;

      return;
    }

    int n_to_sort = n_elected > k ? k : n_elected;
    Eigen::VectorXi idx(n_elected);
    std::iota(idx.data(), idx.data() + n_elected, 0);
    miniselect::pdqpartial_sort_branchless(
        idx.data(), idx.data() + n_to_sort, idx.data() + n_elected,
        [&distances](int i1, int i2) { return distances(i1) < distances(i2); });

    for (int i = 0; i < k; ++i) out[i] = i < n_elected ? indices[idx(i)] : -1;

    if (out_distances) {
      for (int i = 0; i < k; ++i)
        out_distances[i] = i < n_elected ? std::sqrt(distances(idx(i))) : -1;
    }
  }

  const Eigen::Map<const RowMatrix> corpus;  // corpus from which nearest neighbors are searched
  Eigen::MatrixXf split_points;              // all split points in all the trees
  Eigen::MatrixXi split_dimensions;          // all split dimensions in all the trees
  std::vector<std::vector<std::vector<int>>> labels_all;
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