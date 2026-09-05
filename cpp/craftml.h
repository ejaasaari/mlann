#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlann.h"

struct CraftMLOptions {
  int n_trees = 10;
  int max_depth = 20;
  int branching_factor = 10;
  int leaf_size = 32;
  int label_dim = 128;
  int feature_dim = 0;  // Zero means A = I, including for raw L2 data.
  int iterations = 2;
  int node_sample_size = 1000;
  uint32_t seed = 42;
  Distance distance = L2;
};

// CRAFTML (Siblini et al., ICML 2018), adapted to corpus-ID multilabel targets.
// SxSy signed hashing, spherical label k-means++, and feature-centroid routing.
// All training queries participate in every tree; only split fitting is sampled.
class CraftML : public MLANN {
 public:
  struct LabelScore {
    uint32_t id;
    float score;
  };
  struct Node {
    RowMatrix centroids;
    std::vector<uint32_t> children;
    std::vector<LabelScore> labels;
    int training_size = 0;
    bool is_leaf() const { return children.empty(); }
  };
  struct Tree {
    uint64_t feature_seed = 0;
    uint64_t label_seed = 0;
    std::vector<Node> nodes;
  };
  struct QueryStats {
    size_t visited_labels = 0;
    size_t unique_labels = 0;
    int candidates = 0;
    bool fallback = false;
  };

  CraftML(const float *data, int n, int d) : MLANN(data, n, d) {
    if (!data || n <= 0 || d <= 0) throw std::invalid_argument("Corpus must be nonempty");
  }

  const std::vector<Tree> &trees() const { return forest_; }
  Distance distance() const { return options_.distance; }

  void grow(int count, int max_depth, const Eigen::Ref<const UIntRowMatrix> &knn,
            const Eigen::Ref<const RowMatrix> &train, float density = -1, int b = 1) override {
    if (density != -1 || b != 1)
      throw std::invalid_argument("Use CraftML::build to configure feature hashing and leaves");
    CraftMLOptions options;
    options.n_trees = count;
    options.max_depth = max_depth;
    build(knn, train, options);
  }

  void build(const Eigen::Ref<const UIntRowMatrix> &knn,
             const Eigen::Ref<const RowMatrix> &train, const CraftMLOptions &options = {}) {
    if (!empty()) throw std::logic_error("The index has already been built");
    validate(knn, train, options);
    options_ = options;
    std::vector<Tree> forest(options.n_trees);
    std::exception_ptr error;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int t = 0; t < options.n_trees; ++t) {
      try {
        Tree &tree = forest[t];
        tree.feature_seed = mix(uint64_t(options.seed) + 2ULL * t);
        tree.label_seed = mix(uint64_t(options.seed) + 2ULL * t + 1);
        std::mt19937 generator(static_cast<uint32_t>(mix(tree.label_seed)));
        RowMatrix projected;
        if (options.feature_dim > 0) {
          projected.resize(train.rows(), options.feature_dim);
          for (int i = 0; i < train.rows(); ++i)
            project(train.row(i).data(), dim, projected.row(i).data(), options.feature_dim,
                    tree.feature_seed);
        }
        const Eigen::Map<const RowMatrix, 0, Eigen::OuterStride<>> features(
            options.feature_dim > 0 ? projected.data() : train.data(), train.rows(),
            options.feature_dim > 0 ? options.feature_dim : dim,
            Eigen::OuterStride<>(options.feature_dim > 0 ? projected.outerStride() : train.outerStride()));
        std::vector<int> ids(train.rows());
        std::iota(ids.begin(), ids.end(), 0);
        grow_node(tree, ids, 0, ids.size(), 0, features, knn, generator);
      } catch (...) {
#ifdef _OPENMP
#pragma omp critical(craftml_build_error)
#endif
        { if (!error) error = std::current_exception(); }
      }
    }
    if (error) std::rethrow_exception(error);
    forest_ = std::move(forest);
    n_trees = options.n_trees;
  }

  // Unpruned forest probabilities in original corpus-ID space, sorted by ID.
  std::vector<LabelScore> predict(const float *q) const {
    check_query(q);
    Scratch scratch;
    accumulate(q, scratch, nullptr);
    std::vector<LabelScore> result;
    result.reserve(scratch.touched.size());
    for (size_t slot : scratch.touched) result.push_back(scratch.table[slot]);
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.id < b.id; });
    return result;
  }

  using MLANN::query;
  void query(const float *q, int k, float threshold, int *out, Distance dist = L2,
             float *distances = nullptr, int *elected = nullptr) const override {
    search(q, k, -1, threshold, out, dist, distances, elected);
  }

  // budget == -1 selects strict score > threshold; otherwise budget must be >= k.
  // Too few supported candidates triggers exact full-corpus search, never short output.
  void search(const float *q, int k, int budget, float threshold, int *out, Distance dist = L2,
              float *distances = nullptr, int *elected = nullptr, QueryStats *stats = nullptr) const {
    check_query(q);
    if (k <= 0 || k > n_corpus || !out) throw std::invalid_argument("k must be in [1, corpus size]");
    if (dist != options_.distance) throw std::invalid_argument("Search metric must match build metric");
    if (budget != -1 && budget < k) throw std::invalid_argument("candidate_budget must be >= k");
    if (budget == -1 && (!std::isfinite(threshold) || threshold < 0 || threshold > 1))
      throw std::invalid_argument("Probability threshold must be in [0, 1]");
    static thread_local Scratch scratch;
    if (stats) *stats = {};
    accumulate(q, scratch, stats);
    scratch.ranked.clear();
    for (size_t slot : scratch.touched) {
      const auto &entry = scratch.table[slot];
      if (budget >= 0 || entry.score > threshold) scratch.ranked.push_back(entry);
    }
    if (budget >= 0 && scratch.ranked.size() > static_cast<size_t>(budget)) {
      std::nth_element(scratch.ranked.begin(), scratch.ranked.begin() + budget, scratch.ranked.end(),
                       [](const auto &a, const auto &b) {
                         return a.score > b.score || (a.score == b.score && a.id < b.id);
                       });
      scratch.ranked.resize(budget);
    }
    const bool fallback = scratch.ranked.size() < static_cast<size_t>(k);
    const int count = fallback ? n_corpus : static_cast<int>(scratch.ranked.size());
    if (elected) *elected = count;
    if (stats) { stats->candidates = count; stats->fallback = fallback; }
    if (fallback) {
      MLANN::exact_knn(q, k, out, dist, distances);
      return;
    }
    scratch.candidates.clear();
    for (const auto &entry : scratch.ranked) scratch.candidates.push_back(entry.id);
    // Sequential corpus access also makes the k=1 tie break agree with exact search.
    std::sort(scratch.candidates.begin(), scratch.candidates.end());
    const Eigen::Map<const Eigen::RowVectorXf> query(q, dim);
    MLANN::exact_knn(query, k, scratch.candidates, out, dist, distances);
  }

 private:
  CraftMLOptions options_;
  std::vector<Tree> forest_;

  static uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  // Open addressing with touched-slot clearing: memory and reset work depend on
  // observed support, not corpus size. Each query thread owns its scratch storage.
  struct Scratch {
    std::vector<LabelScore> table = std::vector<LabelScore>(512, {UINT32_MAX, 0});
    std::vector<size_t> touched;
    std::vector<LabelScore> ranked;
    std::vector<uint32_t> candidates;
    Eigen::RowVectorXf projected;
    void clear() {
      for (size_t slot : touched) table[slot] = {UINT32_MAX, 0};
      touched.clear();
    }
    void add(uint32_t id, float value) {
      if ((touched.size() + 1) * 2 > table.size()) {
        auto old_table = std::move(table);
        auto old_touched = std::move(touched);
        table.assign(old_table.size() * 2, {UINT32_MAX, 0});
        touched.clear();
        touched.reserve(old_touched.size() * 2);
        for (size_t slot : old_touched) add(old_table[slot].id, old_table[slot].score);
      }
      size_t slot = mix(id) & (table.size() - 1);
      while (table[slot].id != UINT32_MAX && table[slot].id != id)
        slot = (slot + 1) & (table.size() - 1);
      if (table[slot].id == UINT32_MAX) {
        table[slot] = {id, 0};
        touched.push_back(slot);
      }
      table[slot].score += value;
    }
  };

  static void project(const float *x, int n, float *out, int p, uint64_t seed) {
    std::fill(out, out + p, 0);
    for (int j = 0; j < n; ++j) {
      const uint64_t hash = mix(seed ^ uint64_t(j));
      out[hash % p] += (hash >> 63) ? x[j] : -x[j];
    }
  }

  void validate(const Eigen::Ref<const UIntRowMatrix> &knn,
                const Eigen::Ref<const RowMatrix> &train, const CraftMLOptions &o) const {
    if (o.n_trees <= 0 || o.max_depth < 0 || o.max_depth > 64 || o.branching_factor < 2 ||
        o.leaf_size <= 0 || o.label_dim <= 0 || o.feature_dim < 0 || o.iterations <= 0 ||
        o.node_sample_size < o.branching_factor || (o.distance != IP && o.distance != L2))
      throw std::invalid_argument("Invalid CraftML build parameters");
    if (!train.rows() || train.cols() != dim || train.rows() > std::numeric_limits<int>::max() ||
        knn.rows() != train.rows() || knn.cols() <= 0 || knn.cols() > n_corpus)
      throw std::invalid_argument("Incompatible training features or neighbor labels");
    if (!train.allFinite() || !corpus.allFinite())
      throw std::invalid_argument("Corpus and training features must be finite");
    std::vector<uint32_t> row(knn.cols());
    for (int i = 0; i < knn.rows(); ++i) {
      std::copy(knn.row(i).data(), knn.row(i).data() + knn.cols(), row.begin());
      std::sort(row.begin(), row.end());
      if (row.back() >= static_cast<uint32_t>(n_corpus) ||
          std::adjacent_find(row.begin(), row.end()) != row.end())
        throw std::invalid_argument("Neighbor labels must be distinct valid corpus IDs per query");
    }
  }

  void check_query(const float *q) const {
    if (empty()) throw std::logic_error("Cannot query before building index");
    if (!q || !Eigen::Map<const Eigen::RowVectorXf>(q, dim).allFinite())
      throw std::invalid_argument("Query features must be finite");
  }

  int route(const float *q, const RowMatrix &centroids) const {
    const Eigen::Map<const Eigen::RowVectorXf> x(q, centroids.cols());
    int best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < centroids.rows(); ++c) {
      const float score = options_.distance == IP ? x.dot(centroids.row(c))
                                                 : -(x - centroids.row(c)).squaredNorm();
      if (score > best_score) { best_score = score; best = c; }
    }
    return best;
  }

  void accumulate(const float *q, Scratch &scratch, QueryStats *stats) const {
    scratch.clear();
    for (const Tree &tree : forest_) {
      const float *features = q;
      if (options_.feature_dim > 0) {
        scratch.projected.resize(options_.feature_dim);
        project(q, dim, scratch.projected.data(), options_.feature_dim, tree.feature_seed);
        features = scratch.projected.data();
      }
      uint32_t node = 0;
      while (!tree.nodes[node].is_leaf()) {
        const Node &current = tree.nodes[node];
        node = current.children[route(features, current.centroids)];
      }
      const auto &labels = tree.nodes[node].labels;
      if (stats) stats->visited_labels += labels.size();
      for (const auto &entry : labels) scratch.add(entry.id, entry.score);
    }
    // Normalize once, after summing. This also makes strict threshold boundaries
    // identical between predict() and search(), independent of FMA contraction.
    for (size_t slot : scratch.touched) scratch.table[slot].score /= n_trees;
    if (stats) stats->unique_labels = scratch.touched.size();
  }

  void make_leaf(Node &node, const std::vector<int> &ids, size_t begin, size_t end,
                 const Eigen::Ref<const UIntRowMatrix> &knn) const {
    std::vector<uint32_t> labels;
    labels.reserve((end - begin) * knn.cols());
    for (size_t i = begin; i < end; ++i)
      labels.insert(labels.end(), knn.row(ids[i]).data(), knn.row(ids[i]).data() + knn.cols());
    std::sort(labels.begin(), labels.end());
    for (size_t i = 0; i < labels.size();) {
      size_t j = i + 1;
      while (j < labels.size() && labels[j] == labels[i]) ++j;
      node.labels.push_back({labels[i], static_cast<float>(j - i) / (end - begin)});
      i = j;
    }
    node.labels.shrink_to_fit();
  }

  uint32_t grow_node(Tree &tree, std::vector<int> &ids, size_t begin, size_t end, int level,
                     const Eigen::Ref<const RowMatrix> &features,
                     const Eigen::Ref<const UIntRowMatrix> &knn, std::mt19937 &generator) const {
    const uint32_t node_id = static_cast<uint32_t>(tree.nodes.size());
    tree.nodes.emplace_back();
    tree.nodes[node_id].training_size = static_cast<int>(end - begin);
    auto leaf = [&]() {
      make_leaf(tree.nodes[node_id], ids, begin, end, knn);
      return node_id;
    };
    if (end - begin <= static_cast<size_t>(options_.leaf_size) || level >= options_.max_depth)
      return leaf();

    const int sample_size = static_cast<int>(std::min(end - begin, size_t(options_.node_sample_size)));
    // A partial Fisher-Yates shuffle samples without replacement in O(sample size).
    for (int i = 0; i < sample_size; ++i) {
      std::uniform_int_distribution<size_t> pick(begin + i, end - 1);
      std::swap(ids[begin + i], ids[pick(generator)]);
    }
    RowMatrix labels = RowMatrix::Zero(sample_size, options_.label_dim);
    for (int i = 0; i < sample_size; ++i) {
      for (int j = 0; j < knn.cols(); ++j) {
        const uint64_t hash = mix(tree.label_seed ^ uint64_t(knn(ids[begin + i], j)));
        labels(i, hash % options_.label_dim) += (hash >> 63) ? 1.0f : -1.0f;
      }
      labels.row(i).stableNormalize();  // Zero sketches remain zero.
    }
    const int branches = std::min(options_.branching_factor, sample_size);
    RowMatrix centers(branches, options_.label_dim);
    std::uniform_int_distribution<int> first(0, sample_size - 1);
    centers.row(0) = labels.row(first(generator));
    Eigen::VectorXf nearest = Eigen::VectorXf::Constant(sample_size,
                                                       std::numeric_limits<float>::infinity());
    int count = 1;
    while (count < branches) {
      for (int i = 0; i < sample_size; ++i)
        nearest(i) = std::min(nearest(i), (labels.row(i) - centers.row(count - 1)).squaredNorm());
      const double total = nearest.cast<double>().sum();
      if (total <= 1e-12) break;
      std::uniform_real_distribution<double> pick(0, total);
      double target = pick(generator);
      int selected = sample_size - 1;
      for (int i = 0; i < sample_size; ++i) {
        target -= nearest(i);
        if (target < 0) { selected = i; break; }
      }
      centers.row(count++) = labels.row(selected);
    }
    if (count == 1) return leaf();
    centers.conservativeResize(count, Eigen::NoChange);
    std::vector<int> assignment(sample_size), sizes(count);
    RowMatrix sums(count, options_.label_dim);
    for (int iteration = 0; iteration < options_.iterations; ++iteration) {
      sums.setZero();
      std::fill(sizes.begin(), sizes.end(), 0);
      const RowMatrix similarity = labels * centers.transpose();
      for (int i = 0; i < sample_size; ++i) {
        Eigen::Index cluster;
        similarity.row(i).maxCoeff(&cluster);
        assignment[i] = static_cast<int>(cluster);
        ++sizes[cluster];
        sums.row(cluster) += labels.row(i);
      }
      for (int c = 0; c < count; ++c) {
        if (sizes[c]) { centers.row(c) = sums.row(c); centers.row(c).stableNormalize(); }
      }
    }
    RowMatrix classifier = RowMatrix::Zero(count, features.cols());
    for (int i = 0; i < sample_size; ++i)
      classifier.row(assignment[i]) += features.row(ids[begin + i]);
    int fitted = 0;
    for (int c = 0; c < count; ++c) {
      if (!sizes[c]) continue;
      classifier.row(c) /= sizes[c];
      if (options_.distance == IP) classifier.row(c).stableNormalize();
      classifier.row(fitted++) = classifier.row(c);
    }
    classifier.conservativeResize(fitted, Eigen::NoChange);
    if (fitted < 2) return leaf();

    // Route ALL node examples with the learned feature router before making leaves.
    std::vector<std::vector<int>> groups(fitted);
    for (size_t i = begin; i < end; ++i)
      groups[route(features.row(ids[i]).data(), classifier)].push_back(ids[i]);
    int occupied = 0;
    for (const auto &group : groups) occupied += !group.empty();
    if (occupied < 2) return leaf();
    tree.nodes[node_id].centroids.resize(occupied, features.cols());
    tree.nodes[node_id].children.resize(occupied);
    size_t offset = begin;
    int child = 0;
    for (int c = 0; c < fitted; ++c) {
      if (groups[c].empty()) continue;
      tree.nodes[node_id].centroids.row(child) = classifier.row(c);
      std::copy(groups[c].begin(), groups[c].end(), ids.begin() + offset);
      const size_t next = offset + groups[c].size();
      // Recursive growth can reallocate tree.nodes: never retain a Node reference.
      const uint32_t result = grow_node(tree, ids, offset, next, level + 1, features, knn, generator);
      tree.nodes[node_id].children[child++] = result;
      offset = next;
    }
    return node_id;
  }
};
