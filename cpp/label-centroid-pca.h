#pragma once

// Full-input spectral orientations, followed by the original hard PAL loss.
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include "mlann.h"

namespace label_centroid_pca_detail {
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
struct Sample {
  Matrix x;                       // Centered queries, all input coordinates.
  std::vector<int> labels, counts; // Compact corpus IDs, row major N x K.
  int k = 0;
  int n() const { return x.rows(); }
};
// Single-precision fitting, including centering, products and spectral solves.
inline Eigen::MatrixXf covariance(const Sample &a) {
  return (a.x.transpose() * a.x) / a.n();
}
inline bool disjoint_labels(const Sample &a) {
  return std::all_of(a.counts.begin(), a.counts.end(), [](int count) { return count <= 1; });
}
// B = X' Y diag(counts)^-1 Y' X / (K N). Use the smaller of
// label-space scatter and sample-space overlap; never allocate N*K by dim.
inline Eigen::MatrixXf between(const Sample &a, const Matrix &x) {
  if (disjoint_labels(a)) return (x.transpose() * x) / a.n();
  if (a.counts.size() <= size_t(a.n())) {
    Matrix sums = Matrix::Zero(a.counts.size(), x.cols());
    for (int i = 0; i < a.n(); ++i)
      for (int j = 0; j < a.k; ++j) sums.row(a.labels[i * a.k + j]) += x.row(i);
    for (int j = 0; j < sums.rows(); ++j)
      if (a.counts[j]) sums.row(j) /= std::sqrt(float(a.counts[j]));
    return (sums.transpose() * sums) / (float(a.k) * a.n());
  }
  std::vector<int> offsets(a.counts.size() + 1), rows(a.labels.size());
  std::partial_sum(a.counts.begin(), a.counts.end(), offsets.begin() + 1);
  auto next = offsets;
  for (int i = 0; i < a.n(); ++i)
    for (int j = 0; j < a.k; ++j) rows[next[a.labels[i * a.k + j]]++] = i;
  Matrix overlap = Matrix::Zero(a.n(), a.n());
  for (size_t label = 0; label < a.counts.size(); ++label) {
    if (!a.counts[label]) continue;
    const float weight = 1.0f / a.counts[label];
    for (int i = offsets[label]; i < offsets[label + 1]; ++i)
      for (int j = offsets[label]; j < offsets[label + 1]; ++j)
        overlap(rows[i], rows[j]) += weight;
  }
  const Matrix smoothed = (overlap * x);
  return (x.transpose() * smoothed) / (float(a.k) * a.n());
}
inline Eigen::MatrixXf between(const Sample &a) { return between(a, a.x); }
inline Eigen::VectorXf leading_dense(const Eigen::MatrixXf &matrix) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(matrix);
  if (solver.info() != Eigen::Success) throw std::runtime_error("Spectral solve failed");
  if (!(solver.eigenvalues().tail(1)[0] > 0)) return Eigen::VectorXf::Zero(matrix.rows());
  return solver.eigenvectors().col(matrix.rows() - 1);
}
struct LeadingEigenStats { int iterations = 0; bool fallback = false; };
// Compute only the largest eigenpair: Householder tridiagonalization, Sturm
// bisection, then inverse iteration. Never form the dense Householder Q or a
// complete eigenvector basis. Small problems and failed checks use Eigen's
// original solver. No random probes or fixed-budget approximate directions.
inline Eigen::VectorXf leading_bisect(const Eigen::MatrixXf &matrix, LeadingEigenStats *stats = nullptr) {
  if (stats) *stats = {};
  const int n = matrix.rows();
  if (n <= 8) return leading_dense(matrix);
  const float scale = matrix.cwiseAbs().maxCoeff();
  if (!(scale > 0)) return Eigen::VectorXf::Zero(n);
  const Eigen::MatrixXf scaled = matrix / scale;
  Eigen::Tridiagonalization<Eigen::MatrixXf> reduction(scaled);
  Eigen::VectorXf d = reduction.diagonal(), e = reduction.subDiagonal();
  // A diagonal sign change makes every off-diagonal nonnegative. Starting
  // inverse iteration with positive entries then cannot be orthogonal to the
  // leading eigenspace, including reducible / repeated-eigenvalue problems.
  Eigen::VectorXf signs = Eigen::VectorXf::Ones(n);
  for (int i = 1; i < n; ++i) signs[i] = e[i - 1] < 0 ? -signs[i - 1] : signs[i - 1];
  e = e.cwiseAbs();
  float norm = 0;
  for (int i = 0; i < n; ++i)
    norm = std::max(norm, std::abs(d[i]) + (i ? e[i - 1] : 0) + (i + 1 < n ? e[i] : 0));
  if (!(norm > 0)) return Eigen::VectorXf::Zero(n);
  d /= norm; e /= norm;
  constexpr float eps = std::numeric_limits<float>::epsilon();
  constexpr float pivot_min = 16 * std::numeric_limits<float>::min();
  constexpr float tolerance = 2e-6f;
  float lower = d.maxCoeff(), upper = lower;
  for (int i = 0; i < n; ++i)
    upper = std::max(upper, d[i] + (i ? e[i - 1] : 0) + (i + 1 < n ? e[i] : 0));
  upper += 8 * eps;
  // Count eigenvalues below the trial shift using the signs of LDL' pivots.
  for (int step = 0; step < 64 && upper - lower > 8 * eps; ++step) {
    const float shift = lower + (upper - lower) / 2;
    float pivot = d[0] - shift;
    if (std::abs(pivot) < pivot_min) pivot = -pivot_min;
    int below = pivot < 0;
    for (int i = 1; i < n; ++i) {
      pivot = d[i] - shift - e[i - 1] * (e[i - 1] / pivot);
      if (std::abs(pivot) < pivot_min) pivot = -pivot_min;
      below += pivot < 0;
    }
    if (below < n) lower = shift; else upper = shift;
  }
  auto fallback = [&]() -> Eigen::VectorXf {
    if (stats) stats->fallback = true;
    return leading_dense(matrix);
  };
  if (upper <= 0) return Eigen::VectorXf::Zero(n);
  // Keep the shifted matrix positive definite, even after bisection rounding.
  const float shift = upper + 8 * eps;
  Eigen::VectorXf pivots(n), factors(n - 1);
  pivots[0] = shift - d[0];
  for (int i = 1; i < n; ++i) {
    if (!(pivots[i - 1] > 0)) return fallback();
    factors[i - 1] = -e[i - 1] / pivots[i - 1];
    pivots[i] = shift - d[i] + factors[i - 1] * e[i - 1];
  }
  if (!(pivots[n - 1] > 0)) return fallback();
  Eigen::VectorXf v = Eigen::VectorXf::Ones(n), product(n);
  for (int iteration = 0; iteration < 8; ++iteration) {
    if (stats) ++stats->iterations;
    for (int i = 1; i < n; ++i) v[i] -= factors[i - 1] * v[i - 1];
    v.array() /= pivots.array();
    for (int i = n - 2; i >= 0; --i) v[i] -= factors[i] * v[i + 1];
    if (!v.allFinite() || !(v.norm() > 0)) return fallback();
    v.normalize();
    product = d.array() * v.array();
    for (int i = 0; i + 1 < n; ++i) {
      product[i] += e[i] * v[i + 1]; product[i + 1] += e[i] * v[i];
    }
    const float value = v.dot(product);
    if (std::abs(upper - value) <= tolerance && (product - value * v).norm() <= tolerance) {
      if (!(value > 0)) return fallback();
      Eigen::VectorXf result = reduction.matrixQ() * (signs.array() * v.array()).matrix();
      result.normalize();
      const Eigen::VectorXf residual = scaled.selfadjointView<Eigen::Lower>() * result - (value * norm) * result;
      if (residual.norm() <= 4 * tolerance * norm) return result;
      return fallback();
    }
  }
  return fallback();
}
// Lanczos with two-pass full reorthogonalization, checked against the original
// matrix. Small systems or unconverged iterations use the selected direct solve.
inline Eigen::VectorXf leading(const Eigen::MatrixXf &matrix, LeadingEigenStats *stats = nullptr) {
  const int n = matrix.rows();
  if (n < 48) return leading_bisect(matrix, stats);
  if (stats) *stats = {};
  const float scale = matrix.cwiseAbs().maxCoeff();
  if (!(scale > 0)) return Eigen::VectorXf::Zero(n);
  const Eigen::MatrixXf a = (matrix / scale).selfadjointView<Eigen::Lower>();
  const int limit = std::min(n, 48);
  Eigen::MatrixXf basis(n, limit);
  Eigen::VectorXf diagonal(limit), off_diagonal(limit), v(n);
  std::minstd_rand generator(193U + n);
  for (int i = 0; i < n; ++i) v[i] = std::uniform_real_distribution<float>(-1, 1)(generator);
  v.normalize();
  const float tolerance = 1e-6f * a.norm();
  for (int j = 0; j < limit; ++j) {
    basis.col(j) = v;
    Eigen::VectorXf residual = a * v;
    diagonal[j] = v.dot(residual);
    residual -= diagonal[j] * v;
    if (j) residual -= off_diagonal[j - 1] * basis.col(j - 1);
    for (int pass = 0; pass < 2; ++pass) {
      const Eigen::VectorXf coefficients = basis.leftCols(j + 1).transpose() * residual;
      residual.noalias() -= basis.leftCols(j + 1) * coefficients;
    }
    off_diagonal[j] = residual.norm();
    const bool breakdown = off_diagonal[j] <= 32 * std::numeric_limits<float>::epsilon();
    if ((j + 1) % 8 == 0 || breakdown || j + 1 == limit) {
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> ritz;
      ritz.computeFromTridiagonal(diagonal.head(j + 1), off_diagonal.head(j));
      if (ritz.info() != Eigen::Success) break;
      const float value = ritz.eigenvalues()[j];
      Eigen::VectorXf result = basis.leftCols(j + 1) * ritz.eigenvectors().col(j);
      result.normalize();
      if (value > 0 && (a * result - value * result).norm() <= tolerance) {
        if (stats) stats->iterations = j + 1;
        return result;
      }
      if (breakdown) break;
    }
    v = residual / off_diagonal[j];
  }
  Eigen::VectorXf result = leading_bisect(matrix, stats);
  if (stats) stats->fallback = true;
  return result;
}
// An orthonormal change of basis, with no truncation or feature selection.
// When N < dim, every nonzero spectral direction lies in this row space.
struct QueryBasis {
  Eigen::HouseholderQR<Eigen::MatrixXf> qr;
  Eigen::MatrixXf basis;
  Matrix coordinates;
  bool reduced = false;
  explicit QueryBasis(const Matrix &x, bool implicit = true) {
    if (x.rows() < x.cols()) {
      reduced = true;
      if (implicit) {
        // Keep the reflectors and apply them only to the fitted direction.
        qr.compute(x.transpose());
        coordinates = qr.matrixQR().topRows(x.rows()).template triangularView<Eigen::Upper>().transpose();
      } else {
        Eigen::HouseholderQR<Eigen::MatrixXf> explicit_qr(x.transpose());
        basis = explicit_qr.householderQ() * Eigen::MatrixXf::Identity(x.cols(), x.rows());
        coordinates = explicit_qr.matrixQR().topRows(x.rows()).template triangularView<Eigen::Upper>().transpose();
      }
    }
  }
  const Matrix &get(const Matrix &x) const { return reduced ? coordinates : x; }
  Eigen::VectorXf expand(const Eigen::VectorXf &v) const {
    Eigen::VectorXf result;
    if (!reduced) result = v;
    else if (basis.size()) result = basis * v;
    else {
      Eigen::VectorXf padded = Eigen::VectorXf::Zero(qr.rows());
      padded.head(v.size()) = v;
      result = qr.householderQ() * padded;
    }
    if (result.squaredNorm() > 0) result.normalize();
    return result;
  }
};
inline Eigen::VectorXf pca(const Sample &a) {
  if (a.n() >= a.x.cols()) return leading(covariance(a));
  const Eigen::VectorXf v = leading((a.x * a.x.transpose()) / a.n());
  Eigen::VectorXf result = a.x.transpose() * v;
  if (result.squaredNorm() > 0) result.normalize();
  return result;
}
inline Eigen::VectorXf label_centroid(const Sample &a) {
  if (disjoint_labels(a)) return pca(a);
  QueryBasis q(a.x);
  return q.expand(leading(between(a, q.get(a.x))));
}
inline float clogc(int count) { return count ? count * std::log(float(count)) : 0; }
struct PALTables {
  std::vector<float> label, mass, delta;
  PALTables(int n, int k) : label(n + 1), mass(n + 1), delta(n + 1) {
    for (int i = 1; i <= n; ++i) {
      label[i] = clogc(i); mass[i] = clogc(k * i);
      delta[i] = label[i] - label[i - 1];
    }
  }
};
struct Split { float threshold = 0, loss = INFINITY; bool valid = false; float gain = 0; };
struct ThresholdScratch { std::vector<int> order, left, right; };
inline Split threshold(const Sample &a, const Eigen::VectorXf &projection,
                       const PALTables &tables, ThresholdScratch &scratch) {
  auto &order = scratch.order; auto &left = scratch.left; auto &right = scratch.right;
  order.resize(a.n()); left.assign(a.counts.size(), 0); right = a.counts;
  std::iota(order.begin(), order.end(), 0);
  miniselect::pdqsort_branchless(order.begin(), order.end(), [&](int i, int j) {
    return projection[i] < projection[j] || (projection[i] == projection[j] && i < j);
  });
  float sl = 0, sr = 0;
  for (int c : a.counts) sr += tables.label[c];
  const float parent = tables.mass[a.n()] - sr;
  Split best;
  for (int p = 0; p + 1 < a.n(); ++p) {
    for (int j = 0; j < a.k; ++j) {
      const int label = a.labels[order[p] * a.k + j];
      sl += tables.delta[++left[label]]; sr -= tables.delta[right[label]--];
    }
    const float lo = projection[order[p]], hi = projection[order[p + 1]];
    if (!(lo < hi)) continue;
    const float loss = tables.mass[p + 1] + tables.mass[a.n() - p - 1] - sl - sr;
    if (loss < best.loss) {
      float b = lo + (hi - lo) / 2;
      if (!(b < hi)) b = lo;
      best = {b, loss, true, parent - loss};
    }
  }
  return best;
}
inline Split threshold(const Sample &a, const Eigen::VectorXf &projection) {
  const PALTables tables(a.n(), a.k); ThresholdScratch scratch;
  return threshold(a, projection, tables, scratch);
}
inline uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
// Partial Fisher-Yates draws with sparse swaps, O(k), no scan through a large node.
template <typename Generator>
inline std::vector<int> sample(int n, int k, Generator &rng) {
  std::vector<int> result;
  result.reserve(k);
  if (k < n / 4) {
    std::unordered_set<int> selected;
    selected.reserve(k);
    for (int i = n - k; i < n; ++i) {
      int j = std::uniform_int_distribution<int>(0, i)(rng);
      if (!selected.insert(j).second) { selected.insert(i); j = i; }
      result.push_back(j);
    }
  } else {
    result.resize(n);
    std::iota(result.begin(), result.end(), 0);
    for (int i = 0; i < k; ++i)
      std::swap(result[i], result[std::uniform_int_distribution<int>(i, n - 1)(rng)]);
    result.resize(k);
  }
  return result;
}
}  // namespace label_centroid_pca_detail

class LabelCentroidPCA : public MLANN {
 public:
  struct Options { int sample = 300; uint64_t seed = 17; };
  LabelCentroidPCA(const float *data, int n, int d)
      : LabelCentroidPCA(data, n, d, Options{}) {}
  LabelCentroidPCA(const float *data, int n, int d, Options options)
      : MLANN(data, n, d), options_(options) {
    if (options.sample < 2) throw std::invalid_argument("sample must be >= 2");
  }
  using MLANN::query;
  void grow(int trees, int max_depth, const Eigen::Ref<const UIntRowMatrix> &knn,
            const Eigen::Ref<const RowMatrix> &train, float density_ = -1, int b_ = 1) override {
    if (!empty()) throw std::logic_error("Index already grown");
    if (dim < 1 || n_corpus < 1 || trees <= 0 || train.rows() < 2 || max_depth < 1 || max_depth > 29 ||
        max_depth > std::log2(train.rows()) || b_ < 1 || knn.cols() < 1 ||
        knn.rows() != train.rows() || train.cols() != dim || !train.allFinite() ||
        !corpus.allFinite() || knn.maxCoeff() >= uint32_t(n_corpus))
      throw std::invalid_argument("Invalid forest data or dimensions");
    // Reuse one sorting buffer per worker instead of allocating for every row.
    int duplicates = 0;
#pragma omp parallel reduction(|:duplicates)
    {
      std::vector<uint32_t> ids(knn.cols());
#pragma omp for schedule(static)
      for (int i = 0; i < knn.rows(); ++i) {
        std::copy_n(knn.row(i).data(), knn.cols(), ids.begin());
        miniselect::pdqsort_branchless(ids.begin(), ids.end());
        duplicates |= std::adjacent_find(ids.begin(), ids.end()) != ids.end();
      }
    }
    if (duplicates) throw std::invalid_argument("Neighbor IDs must be distinct within each row");
    // Accepted for compatibility with MLANN::grow; these methods use every feature.
    (void)density_;
    n_trees = trees; depth = max_depth; b = b_;
    forests_.resize(trees); leaves_.resize(trees);
    std::vector<std::vector<float>> tree_projections(trees);
    const label_centroid_pca_detail::PALTables tables(std::min<int>(options_.sample, train.rows()), knn.cols());
#pragma omp parallel
    {
      Scratch scratch(n_corpus);
#pragma omp for schedule(dynamic, 1)
      for (int t = 0; t < trees; ++t) {
        std::vector<int> rows(train.rows());
        std::iota(rows.begin(), rows.end(), 0);
        forests_[t].reserve(std::min<size_t>((size_t(1) << (depth + 1)) - 1, 2 * train.rows()));
        scratch.projections.clear();
        scratch.generator.seed(uint32_t(label_centroid_pca_detail::mix(options_.seed ^ label_centroid_pca_detail::mix(t))));
        grow_node(rows.begin(), rows.end(), 0, t, train, knn, tables, scratch);
        forests_[t].shrink_to_fit(); leaves_[t].shrink_to_fit();
        tree_projections[t] = std::move(scratch.projections);
      }
    }
    pack_projections(tree_projections);
  }
  void query(const float *data, int k, float threshold, int *out, Distance dist = L2,
             float *distances = nullptr, int *elected_count = nullptr) const override {
    Eigen::VectorXf votes = Eigen::VectorXf::Zero(n_corpus);
    std::vector<uint32_t> elected;
    std::array<int, routing_batch_size> leaves;
    for (int first = 0; first < n_trees; first += routing_batch_size) {
      const int count = std::min(routing_batch_size, n_trees - first);
      route_batch(data, first, count, leaves.data());
      for (int t = 0; t < count; ++t) {
        const auto &leaf = leaves_[first + t][leaves[t]];
        for (size_t i = 0; i < leaf.labels.size(); ++i)
          if ((votes[leaf.labels[i]] += leaf.votes[i]) >= threshold) {
            elected.push_back(leaf.labels[i]); votes[leaf.labels[i]] = -9999999;
          }
      }
    }
    if (elected_count) *elected_count = elected.size();
    exact_knn(Eigen::Map<const Eigen::RowVectorXf>(data, dim), k, elected, out, dist, distances);
  }
 protected:
  struct Node {
    uint32_t projection = 0;
    float threshold = 0;
    int left = -1, right = -1, leaf = -1;
  };
  struct Leaf { std::vector<uint32_t> labels; std::vector<float> votes; };
  std::vector<std::vector<Node>> forests_;
  std::vector<std::vector<Leaf>> leaves_;
  RowMatrix projections_;
  static constexpr int routing_batch_size = 64;
  void route_batch(const float *query, int first, int count, int *leaves) const {
    std::array<int, routing_batch_size> nodes{}, active;
    std::array<uint32_t, routing_batch_size> rows;
    std::array<float, routing_batch_size> scores;
    int remaining = 0;
    for (int t = 0; t < count; ++t)
      if (forests_[first + t][0].leaf < 0) active[remaining++] = t;
    while (remaining) {
      for (int i = 0; i < remaining; ++i) {
        const int t = active[i];
        rows[i] = forests_[first + t][nodes[t]].projection;
      }
      mlann_detail::compute_one_to_many(query, projections_.data(), dim, rows.data(),
                                       remaining, mlann_detail::OneToManyMetric::IP, scores.data());
      int next = 0;
      for (int i = 0; i < remaining; ++i) {
        const int t = active[i];
        const auto &node = forests_[first + t][nodes[t]];
        nodes[t] = scores[i] <= node.threshold ? node.left : node.right;
        if (forests_[first + t][nodes[t]].leaf < 0) active[next++] = t;
      }
      remaining = next;
    }
    for (int t = 0; t < count; ++t) leaves[t] = forests_[first + t][nodes[t]].leaf;
  }
 private:
  using It = std::vector<int>::iterator;
  struct Scratch {
    std::vector<int> map;
    std::vector<uint32_t> touched;
    std::minstd_rand generator;
    std::vector<float> projections;
    label_centroid_pca_detail::ThresholdScratch threshold;
    explicit Scratch(int n) : map(n, 0) {}
    void reset() { for (auto id : touched) map[id] = 0; touched.clear(); }
  };
  void pack_projections(std::vector<std::vector<float>> &trees) {
    size_t rows = 0;
    for (const auto &values : trees) rows += values.size() / dim;
    if (rows > std::numeric_limits<uint32_t>::max())
      throw std::length_error("Too many oblique projections");
    projections_.resize(rows, dim);
    size_t offset = 0;
    for (int t = 0; t < n_trees; ++t) {
      auto &values = trees[t];
      if (!values.empty()) {
        std::copy(values.begin(), values.end(), projections_.data() + offset * dim);
        for (auto &node : forests_[t]) if (node.leaf < 0) node.projection += uint32_t(offset);
        offset += values.size() / dim;
      }
      std::vector<float>().swap(values);
    }
  }
  // Fitting, full-node partitioning and querying share the same float SIMD
  // reduction, including its feature tails. No boundary fallback is needed.
  static float project(const Eigen::VectorXf &normal, const float *row) {
    const uint32_t zero = 0; float score;
    mlann_detail::compute_one_to_many(row, normal.data(), normal.size(), &zero, 1,
                                     mlann_detail::OneToManyMetric::IP, &score);
    return score;
  }
  void leaf(Node &node, int tree, It begin, It end, const Eigen::Ref<const UIntRowMatrix> &knn, Scratch &s) {
    node.leaf = leaves_[tree].size(); leaves_[tree].emplace_back();
    auto &output = leaves_[tree].back();
    s.touched.reserve(std::min<size_t>(n_corpus, size_t(end - begin) * knn.cols()));
    for (auto it = begin; it != end; ++it) for (int j = 0; j < knn.cols(); ++j) {
      const uint32_t id = knn(*it, j);
      if (s.map[id]++ == 0) s.touched.push_back(id);
    }
    output.labels.reserve(s.touched.size()); output.votes.reserve(s.touched.size());
    uint64_t total = 0;
    for (auto id : s.touched) if (s.map[id] >= b) {
      output.labels.push_back(id); output.votes.push_back(s.map[id]); total += s.map[id];
    }
    if (total) {
      const float inv = 1.f / (float(total) * float(n_trees));
      for (auto &v : output.votes) v *= inv;
    }
    s.reset();
  }
  int grow_node(It begin, It end, int level, int tree,
                const Eigen::Ref<const RowMatrix> &train, const Eigen::Ref<const UIntRowMatrix> &knn,
                const label_centroid_pca_detail::PALTables &tables,
                Scratch &scratch) {
    using namespace label_centroid_pca_detail;
    const int index = forests_[tree].size();
    forests_[tree].emplace_back();
    Node node;
    const int count = end - begin;
    if (level == depth || count < 2) {
      leaf(node, tree, begin, end, knn, scratch); forests_[tree][index] = node; return index;
    }
    const int n = std::min(options_.sample, count);
    const auto local = sample(count, n, scratch.generator);
    Sample a;
    a.k = knn.cols(); a.x.resize(n, dim); a.labels.resize(size_t(n) * a.k);
    a.counts.reserve(std::min<size_t>(n_corpus, a.labels.size()));
    for (int i = 0; i < n; ++i) {
      const int row = begin[local[i]];
      a.x.row(i) = train.row(row);
      for (int j = 0; j < a.k; ++j) {
        const auto label = knn(row, j);
        if (!scratch.map[label]) { scratch.touched.push_back(label); a.counts.push_back(0); scratch.map[label] = a.counts.size(); }
        const int compact = scratch.map[label] - 1;
        a.labels[i * a.k + j] = compact; ++a.counts[compact];
      }
    }
    scratch.reset();
    a.x.rowwise() -= a.x.colwise().mean().eval();
    const Eigen::VectorXf direction = label_centroid(a);
    const Eigen::VectorXf normal = direction;
    Eigen::VectorXf projections(n);
    for (int i = 0; i < n; ++i) projections[i] = project(normal, train.row(begin[local[i]]).data());
    const Split fit = threshold(a, projections, tables, scratch.threshold);
    if (!fit.valid || !(fit.gain > 1e-9f)) {
      leaf(node, tree, begin, end, knn, scratch);
    } else {
      node.threshold = fit.threshold;
      const auto mid = std::partition(begin, end, [&](int row) { return project(normal, train.row(row).data()) <= node.threshold; });
      if (mid == begin || mid == end) leaf(node, tree, begin, end, knn, scratch);
      else {
        node.projection = scratch.projections.size() / dim;
        scratch.projections.insert(scratch.projections.end(), normal.data(), normal.data() + dim);
        node.left = grow_node(begin, mid, level + 1, tree, train, knn, tables, scratch);
        node.right = grow_node(mid, end, level + 1, tree, train, knn, tables, scratch);
      }
    }
    forests_[tree][index] = node;
    return index;
  }
  Options options_;
};