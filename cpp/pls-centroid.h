#pragma once

#include "rf-class-depth.h"
#include <array>
#include <unordered_set>

// Full-support covariance-guided proposals fitted to projected neighbor centroids. The
// original RF PAL objective selects splits; leaf voting and reranking are shared.
class PLSCentroid : public RFClass {
 public:
  struct Options {
    int sample = 300, candidates = 6, sketch_dim = 64;
    uint32_t seed = 17;
  };

  PLSCentroid(const float *data, int n, int d) : PLSCentroid(data, n, d, Options{}) {}
  PLSCentroid(const float *data, int n, int d, Options options_)
      : RFClass(data, n, d), options(options_) {
    if (options.sample < 2 || options.candidates < 1 ||
        options.sketch_dim < 8 || options.sketch_dim % 2)
      throw std::invalid_argument("Invalid PLSCentroid options");
  }

  void grow(int trees, int depth_, const Eigen::Ref<const UIntRowMatrix> &knn,
            const Eigen::Ref<const RowMatrix> &train, float density_ = -1, int b_ = 1) override {
    if (!empty()) throw std::logic_error("The index has already been grown");
    if (trees < 1 || train.rows() < 2 || depth_ < 1 || depth_ > std::log2(train.rows()) ||
        train.cols() != dim || knn.rows() != train.rows() || knn.cols() < 1 || b_ < 1 ||
        knn.maxCoeff() >= uint32_t(n_corpus)) throw std::invalid_argument("Invalid training data");
    n_trees = trees; depth = depth_; b = b_;
    labels_all.resize(trees); votes_all.resize(trees); forest.resize(trees);
    std::vector<std::vector<float>> tree_projections(trees);

    // Targets use training neighbors only and are released after fitting.
    RowMatrix targets = make_targets(knn);
    log2_tbl.resize(train.rows() + 1); t_tbl.resize(train.rows() + 1);
    for (int i = 1; i <= train.rows(); ++i) log2_tbl[i] = std::log2(float(i));
    for (int i = 0; i <= train.rows(); ++i) t_tbl[i] = i * log2_tbl[i];
    for (int i = train.rows(); i > 0; --i) t_tbl[i] -= t_tbl[i - 1];
#pragma omp parallel for schedule(dynamic, 1)
    for (int tree = 0; tree < trees; ++tree) {
      Scratch scratch;
      scratch.ensure_corpus(n_corpus);
      scratch.generator.seed(options.seed + 104729U * (tree + 1));
      std::vector<int> rows(train.rows());
      std::iota(rows.begin(), rows.end(), 0);
      forest[tree].reserve(2 * (1 << std::min(depth, 16)) - 1);
      grow_node(rows.begin(), rows.end(), 0, tree, train, knn, targets, scratch);
      forest[tree].shrink_to_fit();
      tree_projections[tree] = std::move(scratch.projections);
    }
    targets.resize(0, 0);
    pack_projections(tree_projections);
  }

  void query(const float *data, int k, float threshold, int *out, Distance dist = L2,
             float *distances = nullptr, int *n_elected = nullptr) const override {
    std::vector<uint32_t> elected;
    Eigen::VectorXf votes_total = Eigen::VectorXf::Zero(n_corpus);
    std::array<int, routing_batch_size> leaves;
    for (int first = 0; first < n_trees; first += routing_batch_size) {
      const int count = std::min(routing_batch_size, n_trees - first);
      route_batch(data, first, count, leaves.data());
      // Preserve tree order when accumulating votes and electing candidates.
      for (int t = 0; t < count; ++t) {
        const auto &labels = labels_all[first + t][leaves[t]];
        const auto &votes = votes_all[first + t][leaves[t]];
        for (size_t i = 0; i < labels.size(); ++i) {
          if ((votes_total(labels[i]) += votes[i]) >= threshold) {
            elected.push_back(labels[i]);
            votes_total(labels[i]) = -9999999;
          }
        }
      }
    }
    if (n_elected) *n_elected = elected.size();
    exact_knn(Eigen::Map<const Eigen::RowVectorXf>(data, dim), k, elected, out, dist, distances);
  }

 protected:
  struct Normal {
    Eigen::VectorXf weights;
    float threshold = 0, gain = 0;
    float project(const float *x) const {
      return weights.dot(Eigen::Map<const Eigen::VectorXf>(x, weights.size()));
    }
  };
  struct Node {
    uint32_t projection = 0;
    float threshold = 0;
    int left = -1, right = -1, leaf = -1;
  };
  struct Scratch : SplitScratch {
    std::minstd_rand generator;
    std::vector<float> projections;
  };
  Options options;
  std::vector<std::vector<Node>> forest;
  RowMatrix projections;
  static constexpr int routing_batch_size = 64;

  void pack_projections(std::vector<std::vector<float>> &tree_projections) {
    size_t rows = 0;
    for (const auto &values : tree_projections) rows += values.size() / dim;
    if (rows > std::numeric_limits<uint32_t>::max())
      throw std::length_error("Too many PLSCentroid projections");
    projections.resize(rows, dim);
    size_t offset = 0;
    for (int tree = 0; tree < n_trees; ++tree) {
      auto &values = tree_projections[tree];
      if (!values.empty()) {
        std::copy(values.begin(), values.end(), projections.data() + offset * dim);
        for (auto &node : forest[tree])
          if (node.leaf < 0) node.projection += static_cast<uint32_t>(offset);
        offset += values.size() / dim;
      }
      std::vector<float>().swap(values);
    }
  }

  void route_batch(const float *query, int first, int count, int *leaves) const {
    std::array<int, routing_batch_size> nodes{}, active;
    std::array<uint32_t, routing_batch_size> rows;
    std::array<float, routing_batch_size> scores;
    int remaining = 0;
    for (int t = 0; t < count; ++t)
      if (forest[first + t][0].leaf < 0) active[remaining++] = t;

    while (remaining) {
      for (int i = 0; i < remaining; ++i) {
        const int t = active[i];
        rows[i] = forest[first + t][nodes[t]].projection;
      }
      mlann_detail::compute_one_to_many(query, projections.data(), dim, rows.data(),
                                       remaining, mlann_detail::OneToManyMetric::IP,
                                       scores.data());
      int next = 0;
      for (int i = 0; i < remaining; ++i) {
        const int t = active[i];
        const auto &node = forest[first + t][nodes[t]];
        nodes[t] = scores[i] <= node.threshold ? node.left : node.right;
        if (forest[first + t][nodes[t]].leaf < 0) active[next++] = t;
      }
      remaining = next;
    }
    for (int t = 0; t < count; ++t) leaves[t] = forest[first + t][nodes[t]].leaf;
  }

  static std::vector<uint32_t> sample_unique(int n, int k, std::minstd_rand &generator) {

    std::vector<uint32_t> reservoir;
    reservoir.reserve(k);
    if (k * 4 < n) {
      // Floyd's algorithm: uniform subset, O(k), no duplicates.
      std::unordered_set<uint32_t> selected;
      for (int i = n - k; i < n; ++i) {
        uint32_t j = std::uniform_int_distribution<int>(0, i)(generator);
        if (!selected.insert(j).second) { selected.insert(i); j = i; }
        reservoir.push_back(j);
      }
    } else {
      reservoir.resize(n);
      std::iota(reservoir.begin(), reservoir.end(), 0);
      for (int i = 0; i < k; ++i) {
        int j = std::uniform_int_distribution<int>(i, n - 1)(generator);
        std::swap(reservoir[i], reservoir[j]);
      }
      reservoir.resize(k);
    }

    return reservoir;
  }

  RowMatrix make_targets(const Eigen::Ref<const UIntRowMatrix> &knn) const {
    const int r = options.sketch_dim;
    std::minstd_rand rng(options.seed + 271828U);
    RowMatrix embedding = RowMatrix::Zero(n_corpus, r);
    // The fixed target sketch is independent of the full-support split normals.
    // Retain 16 coordinates: the 40-tree sketch sweep found no consistent gain
    // from changing this count (benchmarks/pls_centroid_sketch_support_t40).
    struct Projection { std::vector<uint32_t> dims; Eigen::VectorXf weights; };
    std::vector<Projection> projections(r);
    for (auto &projection : projections) {
      projection.dims = sample_unique(dim, std::min(16, dim), rng);
      projection.weights.resize(projection.dims.size());
      std::normal_distribution<float> gaussian;
      for (int j = 0; j < projection.weights.size(); ++j) projection.weights[j] = gaussian(rng);
      projection.weights.normalize();
    }
#pragma omp parallel for
    for (int i = 0; i < n_corpus; ++i)
      for (int j = 0; j < r; ++j) {
        const auto &projection = projections[j];
        for (size_t k = 0; k < projection.dims.size(); ++k)
          embedding(i,j) += projection.weights[k] * corpus(i,projection.dims[k]);
      }
    RowMatrix targets = RowMatrix::Zero(knn.rows(), r);
#pragma omp parallel for
    for (int i = 0; i < knn.rows(); ++i) {
      for (int j = 0; j < knn.cols(); ++j) targets.row(i) += embedding.row(knn(i,j));
      targets.row(i) /= float(knn.cols());
    }
    return targets;
  }

  Normal scan(Normal normal, const std::vector<int> &rows,
              const Eigen::Ref<const RowMatrix> &train, const UIntRowMatrix &labels,
              int n_labels) const {
    const int n = rows.size(), k = labels.cols();
    std::vector<SplitEntry> order(n);
    for (int i = 0; i < n; ++i) order[i] = {normal.project(train.row(rows[i]).data()), i};
    miniselect::pdqsort_branchless(order.begin(), order.end(), [](auto &a, auto &c) { return a.key < c.key; });
    normal.gain = 0;

    std::vector<int> counts(n_labels,0);
    std::vector<float> left_ent(n);
    float entropy = 0;
    for (int pos = 0; pos < n; ++pos) {
      for (int j = 0; j < k; ++j) entropy += t_tbl[++counts[labels(order[pos].index,j)]];
      left_ent[pos] = k * log2_tbl[pos+1] - entropy / float(pos+1);
    }
    const float base = left_ent[n-1];
    for (int pos = 0; pos < n-1; ++pos) {
      for (int j = 0; j < k; ++j) entropy -= t_tbl[counts[labels(order[pos].index,j)]--];
      const int remain = n-pos-1;
      if (order[pos].key == order[pos+1].key) continue;
      const float right_ent = k * log2_tbl[remain] - entropy / float(remain);
      const float gain = base - ((pos+1)*(1.f/n)*left_ent[pos] + remain*(1.f/n)*right_ent);
      if (gain > normal.gain + tol) {
        normal.gain = gain; normal.threshold = midpoint(order[pos].key, order[pos+1].key);
      }
    }
    return normal;
  }
  static float midpoint(float a, float c) {
    const float mid = a + (c-a)*.5f;
    return mid < c ? mid : a; // Keep distinct adjacent floats separated.
  }
  UIntRowMatrix compact(const std::vector<int> &rows, const Eigen::Ref<const UIntRowMatrix> &knn,
                        int &n_labels, SplitScratch &scratch) const {
    UIntRowMatrix labels(rows.size(), knn.cols());
    n_labels = 0;
    std::vector<uint32_t> touched;
    for (size_t i = 0; i < rows.size(); ++i) for (int j = 0; j < knn.cols(); ++j) {
      uint32_t id = knn(rows[i],j);
      if (!scratch.votes[id]) { scratch.votes[id] = ++n_labels; touched.push_back(id); }
      labels(i,j) = scratch.votes[id]-1;
    }
    for (auto id : touched) scratch.votes[id] = 0;
    return labels;
  }
  std::vector<Normal> proposals(const std::vector<int> &rows,
                                const Eigen::Ref<const RowMatrix> &train, const RowMatrix &z,
                                std::minstd_rand &rng) {
    // All candidates share one full-input cross-covariance matrix.
    Eigen::MatrixXf x(rows.size(), dim);
    for (size_t i = 0; i < rows.size(); ++i) x.row(i) = train.row(rows[i]);
    x.rowwise() -= x.colwise().mean().eval();
    const Eigen::MatrixXf zc = z.rowwise() - z.colwise().mean();
    const Eigen::MatrixXf map = x.transpose() * zc / float(rows.size());
    std::vector<Normal> pool;
    pool.reserve(options.candidates);
    for (int c = 0; c < options.candidates; ++c) {
      Eigen::VectorXf a(map.cols());
      std::normal_distribution<float> gaussian;
      for (int j = 0; j < a.size(); ++j) a[j] = gaussian(rng);
      a.normalize();
      Normal normal;
      normal.weights = map * a;
      if (normal.weights.norm() > 1e-10f) normal.weights.normalize();
      pool.push_back(std::move(normal));
    }
    return pool;
  }

  int grow_node(std::vector<int>::iterator begin, std::vector<int>::iterator end, int level, int tree,
                const Eigen::Ref<const RowMatrix> &train, const Eigen::Ref<const UIntRowMatrix> &knn,
                const RowMatrix &targets, Scratch &scratch) {
    const int node = forest[tree].size();
    forest[tree].emplace_back();
    const int n = end-begin;
    Normal best;
    if (level < depth && n > 1) {
      auto sampled = sample_unique(n,std::min(n,options.sample),scratch.generator);
      std::vector<int> rows; rows.reserve(sampled.size());
      for (auto i : sampled) rows.push_back(begin[i]);
      RowMatrix z(rows.size(), targets.cols());
      for (size_t i = 0; i < rows.size(); ++i) z.row(i) = targets.row(rows[i]);
      int n_labels;
      auto labels = compact(rows,knn,n_labels,scratch);
      auto candidates = proposals(rows,train,z,scratch.generator);
      for (auto &candidate : candidates) {
        if (candidate.weights.squaredNorm() < 1e-12f) continue;
        auto evaluated = scan(std::move(candidate),rows,train,labels,n_labels);
        if (evaluated.gain > best.gain + tol) best = std::move(evaluated);
      }
    }
    if (best.weights.size() == 0) {
      const int leaf = labels_all[tree].size();

      auto votes = count_votes(begin,end,knn,scratch);
      labels_all[tree].push_back(std::move(votes.first)); votes_all[tree].push_back(std::move(votes.second));
      forest[tree][node].leaf = leaf;
      return node;
    }
    auto mid = std::partition(begin,end,[&](int row) { return best.project(train.row(row).data()) <= best.threshold; });
    if (mid == begin || mid == end) throw std::logic_error("Hard scan produced an empty full-node child");

    forest[tree][node].projection = scratch.projections.size() / dim;
    forest[tree][node].threshold = best.threshold;
    scratch.projections.insert(scratch.projections.end(), best.weights.data(),
                               best.weights.data() + dim);
    const int left = grow_node(begin,mid,level+1,tree,train,knn,targets,scratch);
    const int right = grow_node(mid,end,level+1,tree,train,knn,targets,scratch);
    forest[tree][node].left = left; forest[tree][node].right = right;
    return node;
  }

};
