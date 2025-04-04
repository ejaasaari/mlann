#ifndef CPP_MLANN_H_
#define CPP_MLANN_H_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
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

  /**
   * Build a normal index.
   *
   * @param n_trees_ number of trees to be grown
   * @param depth_ depth of the trees; in the set
   * \f$\{1,2, \dots ,\lfloor \log_2 (n) \rfloor \}\f$, where \f$n \f$ is the number
   * of data points
   * @param knn_ Eigen ref to the knn matrix of training set; a column is a training set point,
   * and a row is k:th neighbor
   * @param train_ Eigen ref to the training set; a column is a training set point
   * @param density_ expected proportion of non-zero components in the
   * random vectors; on the interval \f$(0,1]\f$; default value sets density to
   * \f$ 1 / \sqrt{d} \f$, where \f$d\f$ is the dimension of the data
   * a default value 0 initializes the rng randomly with std::random_device
   */
  void grow(int n_trees_, int depth_, const Eigen::Ref<const IntRowMatrix> &knn_,
            const Eigen::Ref<const RowMatrix> &train_, float density_ = -1.0, int b_ = 1,
            int n_subsample = 100, float tol_ = 0.001) {
    if (!empty()) {
      throw std::logic_error("The index has already been grown.");
    }

    if (n_trees_ <= 0) {
      throw std::out_of_range("The number of trees must be positive.");
    }

    if (density_ < -1.0001 || density_ > 1.0001 || (density_ > -0.9999 && density_ < -0.0001)) {
      throw std::out_of_range("The density must be on the interval (0,1].");
    }

    int n_train = train_.rows();

    n_trees = n_trees_;
    depth = depth_;
    n_inner_nodes = (1 << depth_) - 1;
    n_leaves = 1 << depth_;
    b = b_;
    n_pool = n_trees_ * depth_;
    n_array = 1 << (depth_ + 1);
    tol = tol_;

    if (density_ < 0) {
      density = 1.0 / std::sqrt(dim);
    } else {
      density = density_;
    }

    const Eigen::Map<const IntRowMatrix> knn(knn_.data(), knn_.rows(), knn_.cols());
    const Eigen::Map<const RowMatrix> train(train_.data(), train_.rows(), train_.cols());

    split_points = Eigen::MatrixXf(n_array, n_trees);
    split_dimensions = Eigen::MatrixXi(n_array, n_trees);
    labels_all = std::vector<std::vector<std::vector<int>>>(n_trees);
    votes_all = std::vector<std::vector<std::vector<float>>>(n_trees);

    std::random_device gen;
    std::mt19937 r = std::mt19937(gen());

    const auto random_dims_all = generate_random_directions(r);

#pragma omp parallel for
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      labels_all[n_tree] = std::vector<std::vector<int>>(n_leaves);
      votes_all[n_tree] = std::vector<std::vector<float>>(n_leaves);

      std::random_device gen;
      std::mt19937 rand_gen(gen());
      std::vector<int> indices(n_train);
      std::iota(indices.begin(), indices.end(), 0);
      grow_subtree(indices.begin(), indices.end(), 0, 0, n_tree, labels_all[n_tree],
                   votes_all[n_tree], train, knn, random_dims_all[n_tree], rand_gen, n_subsample);
    }
  }

  /**@}*/

  /**@}*/

  /** @name Approximate k-nn search
   * A query using a non-autotuned index. Finds k approximate nearest neighbors
   * from a data set corpus for a query point q. Because the index is not autotuned,
   * k and vote threshold are set manually. The indices of k nearest neighbors
   * are written to a buffer out, which has to be preallocated to have at least
   * length k. Optionally also Euclidean distances to these k nearest points
   * are written to a buffer out_distances. If there are less than k points in
   * the candidate set, -1 is written to the remaining locations of the
   * output buffers.
   */

  /**
   * Approximate k-nn search using a normal index.
   *
   * @param data pointer to an array containing the query point
   * @param k number of nearest neighbors searched for
   * @param vote_threshold - number of votes required for a query point to be included in the
   * candidate set
   * @param out output buffer (size = k) for the indices of k approximate nearest neighbors
   * @param out_distances optional output buffer (size = k) for distances to k approximate nearest
   * neighbors
   * @param out_n_elected optional output parameter (size = 1) for the candidate set size
   */
  void query(const float *data, int k, float vote_threshold, int *out,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const {
    if (k <= 0 || k > n_corpus) {
      throw std::out_of_range("k must belong to the set {1, ..., n_corpus}.");
    }

    if (empty()) {
      throw std::logic_error("The index must be built before making queries.");
    }

    const Eigen::Map<const Eigen::VectorXf> q(data, dim);

    std::vector<int> found_leaves(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      int idx_tree = 0;
      int d = 0;
      for (; d < depth; ++d) {
        const int idx_left = 2 * idx_tree + 1;
        const int idx_right = idx_left + 1;
        const float split_point = split_points(idx_tree, n_tree);
        const int split_dimension = split_dimensions(idx_tree, n_tree);
        if (split_dimension == -1) {  // if branch stops before maximum depth
          break;
        }
        if (q(split_dimension) <= split_point) {
          idx_tree = idx_left;
        } else {
          idx_tree = idx_right;
        }
      }
      const int levels2leaf = depth - d;
      found_leaves[n_tree] = (1 << levels2leaf) * (idx_tree + 1) - 1 - n_inner_nodes;
    }

    std::vector<int> elected;
    Eigen::VectorXf votes_total = Eigen::VectorXf::Zero(n_corpus);

    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      int leaf_idx = found_leaves[n_tree];
      const std::vector<int> &labels = labels_all[n_tree][leaf_idx];
      const std::vector<float> &votes = votes_all[n_tree][leaf_idx];
      int n_labels = labels.size();
      for (int i = 0; i < n_labels; ++i) {
        if ((votes_total(labels[i]) += votes[i]) >= vote_threshold) {
          elected.push_back(labels[i]);
          // votes_total(labels[i]) = std::numeric_limits<float>::min();
          votes_total(labels[i]) = -10000;
        }
      }
    }

    if (out_n_elected) *out_n_elected = elected.size();

    exact_knn(q, k, elected, out, out_distances);
  }

  /**
   *  Approximate k-nn search using a normal index.
   *
   * @param q Eigen ref to the query point
   * @param k number of nearest neighbors searched for
   * @param vote_threshold number of votes required for a query point to be included in the
   * candidate set
   * @param out output buffer (size = k) for the indices of k approximate nearest neighbors
   * @param out_distances optional output buffer (size = k) for distances to k approximate nearest
   * neighbors
   * @param out_n_elected optional output parameter (size = 1) for the candidate set size
   */
  void query(const Eigen::Ref<const Eigen::VectorXf> &q, int k, float vote_threshold, int *out,
             float *out_distances = nullptr, int *out_n_elected = nullptr) const {
    query(q.data(), k, vote_threshold, out, out_distances, out_n_elected);
  }

  /**@}*/

  /** @name Exact k-nn search
   * Functions for fast exact k-nn search: find k nearest neighbors for a
   * query point q from a data set corpus_. The indices of k nearest neighbors are
   * written to a buffer out, which has to be preallocated to have at least
   * length k. Optionally also the Euclidean distances to these k nearest points
   * are written to a buffer out_distances. There are both static and member
   * versions.
   */

  /**
   * @param q_data pointer to an array containing the query point
   * @param X_data pointer to an array containing the data set
   * @param n_corpus number of points in a data set
   * @param dim dimension of data
   * @param k number of neighbors searched for
   * @param out output buffer (size = k) for the indices of k nearest neighbors
   * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
   */
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

  /**
   * @param q Eigen ref to a query point
   * @param corpus Eigen ref to a data set
   * @param k number of neighbors searched for
   * @param out output buffer (size = k) for the indices of k nearest neighbors
   * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
   */
  static void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q,
                        const Eigen::Ref<const RowMatrix> &corpus, int k, int *out,
                        float *out_distances = nullptr) {
    MLANN::exact_knn(q.data(), corpus.data(), corpus.rows(), corpus.cols(), k, out, out_distances);
  }

  /**
   * @param q pointer to an array containing the query point
   * @param k number of neighbors searched for
   * @param out output buffer (size = k) for the indices of k nearest neighbors
   * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
   */
  void exact_knn(const float *q, int k, int *out, float *out_distances = nullptr) const {
    MLANN::exact_knn(q, corpus.data(), n_corpus, dim, k, out, out_distances);
  }

  /**
   * @param q pointer to an array containing the query point
   * @param k number of points searched for
   * @param out output buffer (size = k) for the indices of k nearest neighbors
   * @param out_distances optional output buffer (size = k) for the distances to k nearest neighbors
   */
  void exact_knn(const Eigen::Ref<const Eigen::VectorXf> &q, int k, int *out,
                 float *out_distances = nullptr) const {
    MLANN::exact_knn(q.data(), corpus.data(), n_corpus, dim, k, out, out_distances);
  }

  static Eigen::MatrixXi exact_all_pairs(const float *corpus, size_t n_corpus, size_t dim, size_t k,
                                         const float *training_data, size_t n_train) {
    Eigen::MatrixXi true_knn(k, n_train);
    for (size_t i = 0; i < n_train; ++i) {
      MLANN::exact_knn(training_data + i * dim, corpus, n_corpus, dim, k, true_knn.data() + i * k);
    }
    return true_knn;
  }
  /**@}*/

  /** @name Utility functions
   */

  /**
   * Is the index is already constructed or not?
   *
   * @return - is the index empty?
   */
  bool empty() const { return n_trees == 0; }

  /**@}*/

 private:
  static std::tuple<int, float, float> split(const std::vector<int>::iterator &begin,
                                             const std::vector<int>::iterator &end,
                                             const std::vector<int> &random_dims,
                                             const Eigen::Ref<const RowMatrix> &train,
                                             const Eigen::Ref<const IntRowMatrix> &knn, float tol,
                                             int n_corpus, std::mt19937 &r, int n_subsample) {
    int n = end - begin;
    int max_dim = -1;
    float max_gain = 0, max_split = 0;

    if (n <= 1) return std::make_tuple(max_dim, max_split, max_gain);

    std::vector<int> indices;
    if (n_subsample > 0 && n_subsample < n) {
      std::vector<int> indices_org(n);
      n = n_subsample;
      std::iota(indices_org.begin(), indices_org.end(), 0);
      std::shuffle(indices_org.begin(), indices_org.end(), r);
      indices = std::vector<int>(indices_org.begin(), indices_org.begin() + n_subsample);
    } else {
      indices = std::vector<int>(n);
      std::iota(indices.begin(), indices.end(), 0);
    }

    Eigen::VectorXf left_entropies(n), right_entropies(n);
    const float *data = train.data();
    const int cols = train.cols();
    const int k_build = knn.cols();

    for (const auto &d : random_dims) {
      std::sort(indices.begin(), indices.end(), [data, cols, d, begin](int i1, int i2) {
        return data[*(begin + i1) * cols + d] < data[*(begin + i2) * cols + d];
      });

      // std::unordered_map<int, int> votes;
      std::vector<int> votes(n_corpus, 0);

      float entropy = 0;
      for (int ii = 0; ii < n; ++ii) {
        const int i = indices[ii];
        const Eigen::VectorXi knn_crnt = knn.row(*(begin + i));
        for (int j = 0; j < k_build; ++j) {
          int v = ++votes[knn_crnt(j)];
          if (v > 1) entropy -= (v - 1) * log2(v - 1);
          entropy += v * log2(v);
        }
        left_entropies[ii] = k_build * log2(ii + 1) - entropy / (ii + 1);
      }

      for (int ii = 0; ii < n - 1; ++ii) {
        const int i = indices[ii];
        const Eigen::VectorXi knn_crnt = knn.row(*(begin + i));
        for (int j = 0; j < k_build; ++j) {
          int v = --votes[knn_crnt(j)];
          entropy -= (v + 1) * log2(v + 1);
          if (v) entropy += v * log2(v);
        }
        right_entropies[ii] = k_build * log2(n - ii - 1) - entropy / (n - ii - 1);
      }
      right_entropies[n - 1] = 0;

      for (int ii = 0; ii < n - 1; ++ii) {
        const int i = indices[ii];
        if (train(*(begin + i), d) == train(*(begin + indices[ii + 1]), d)) continue;
        float left = static_cast<float>(ii + 1) / n * left_entropies[ii];
        float right = static_cast<float>(n - ii - 1) / n * right_entropies[ii];
        float gain = left_entropies[n - 1] - (left + right);
        if (gain > max_gain + tol) {
          max_gain = gain;
          max_dim = d;
          max_split = (train(*(begin + i), d) + train(*(begin + indices[ii + 1]), d)) / 2.0;
        }
      }
    }
    return std::make_tuple(max_dim, max_split, max_gain);
  }

  std::vector<std::vector<std::vector<int>>> generate_random_directions(std::mt19937 &r) {
    const int n_random_dim = density * dim;
    std::vector<std::vector<std::vector<int>>> dims_all(n_trees);
    for (int n_tree = 0; n_tree < n_trees; ++n_tree) {
      for (int tree_level = 0; tree_level < depth; ++tree_level) {
        std::vector<int> dims(dim);
        std::iota(dims.begin(), dims.end(), 0);
        std::shuffle(dims.begin(), dims.end(), r);
        dims_all[n_tree].push_back(std::vector<int>(dims.begin(), dims.begin() + n_random_dim));
      }
    }
    return dims_all;
  }

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

  /**
   * Builds a single random projection tree. The tree is constructed by recursively
   * projecting the data on a random vector and splitting into two by the median.
   */
  void grow_subtree(std::vector<int>::iterator begin, std::vector<int>::iterator end,
                    int tree_level, int i, int n_tree, std::vector<std::vector<int>> &labels_tree,
                    std::vector<std::vector<float>> &votes_tree,
                    const Eigen::Ref<const RowMatrix> &train,
                    const Eigen::Ref<const IntRowMatrix> &knn,
                    const std::vector<std::vector<int>> &random_dims, std::mt19937 &r,
                    int n_subsample) {
    if (tree_level == depth) {
      const int index_leaf = i - n_inner_nodes;
      const auto ret = count_votes(begin, end, knn);
      labels_tree[index_leaf] = ret.first;
      votes_tree[index_leaf] = ret.second;
      return;
    }

    const auto s =
        split(begin, end, random_dims[tree_level], train, knn, tol, n_corpus, r, n_subsample);
    const int max_dim = std::get<0>(s);
    const float max_split = std::get<1>(s);

    if (max_dim == -1) {
      split_dimensions(i, n_tree) = -1;
      const int levels2leaf = depth - tree_level;
      const int index_leaf = (1 << levels2leaf) * (i + 1) - 1 - n_inner_nodes;
      const auto ret = count_votes(begin, end, knn);
      labels_tree[index_leaf] = ret.first;
      votes_tree[index_leaf] = ret.second;
      return;
    }

    const float *data = train.data();
    const int cols = train.cols();
    auto mid = std::partition(begin, end, [data, cols, max_dim, max_split](const int em) {
      return data[em * cols + max_dim] <= max_split;
    });

    split_points(i, n_tree) = max_split;
    split_dimensions(i, n_tree) = max_dim;

    const int idx_left = 2 * i + 1;
    const int idx_right = idx_left + 1;
    grow_subtree(begin, mid, tree_level + 1, idx_left, n_tree, labels_tree, votes_tree, train, knn,
                 random_dims, r, n_subsample);
    grow_subtree(mid, end, tree_level + 1, idx_right, n_tree, labels_tree, votes_tree, train, knn,
                 random_dims, r, n_subsample);
  }

  /**
   * Find k nearest neighbors from data for the query point
   */
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
  float tol = 0.001;
};

#endif  // CPP_MLANN_H_
