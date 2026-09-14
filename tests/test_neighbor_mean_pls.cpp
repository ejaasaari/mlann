#include <cassert>
#include <iostream>
#include <set>
#include <omp.h>
#include "../cpp/neighbor-mean-pls.h"
using namespace neighbor_mean_pls_detail;

double oracle_clogc(int n) { return n ? n * std::log(double(n)) : 0; }

double oracle(const Sample &a, const std::vector<bool> &left) {
  std::vector<int> l(a.counts.size()), r(l.size()); int nl = 0;
  for (int i = 0; i < a.n(); ++i) {
    nl += left[i];
    for (int j = 0; j < a.k; ++j) ++(left[i] ? l : r)[a.labels[i * a.k + j]];
  }
  double loss = oracle_clogc(nl * a.k) + oracle_clogc((a.n() - nl) * a.k);
  for (size_t j = 0; j < l.size(); ++j) loss -= oracle_clogc(l[j]) + oracle_clogc(r[j]);
  return loss;
}
template <typename Base>
struct InspectIndex : Base {
  using Base::Base;
  using Base::forests_;
  using Base::n_corpus;
  using Base::b;
  using Base::n_trees;
  using Base::leaves_;
  using Base::projections_;
  using Base::route_batch;
  using Base::routing_batch_size;
  void verify(const RowMatrix &train, const UIntRowMatrix &labels) {
    assert(projections_.cols() == train.cols());
    std::vector<std::vector<std::vector<int>>> counts(n_trees);
    for (int t = 0; t < n_trees; ++t)
      counts[t].assign(leaves_[t].size(), std::vector<int>(n_corpus));
    for (int r = 0; r < train.rows(); ++r) {
      for (int first = 0; first < n_trees; first += routing_batch_size) {
        const int size = std::min(routing_batch_size, n_trees - first);
        int routed[routing_batch_size]; route_batch(train.row(r).data(), first, size, routed);
        for (int t = 0; t < size; ++t) {
          // A single tree must agree with the actual batched query router.
          int single; route_batch(train.row(r).data(), first + t, 1, &single);
          assert(single == routed[t]);
          for (int j = 0; j < labels.cols(); ++j) ++counts[first+t][routed[t]][labels(r,j)];
        }
      }
    }
    for (int t = 0; t < n_trees; ++t) {
      for (size_t id = 0; id < leaves_[t].size(); ++id) {
        const auto &leaf = leaves_[t][id];
        int total = 0, retained = 0;
        for (int c : counts[t][id]) if (c >= b) { total += c; ++retained; }
        assert(retained == int(leaf.labels.size()));
        for (size_t j = 0; j < leaf.labels.size(); ++j)
          assert(std::abs(leaf.votes[j] - float(counts[t][id][leaf.labels[j]]) / (total * n_trees)) < 1e-7);
      }
    }
  }
};
using Inspect = InspectIndex<NeighborMeanPLS>;
template <typename Base>
void verify_wide_forest(int dim) {
  RowMatrix data=RowMatrix::Random(80,dim), train=RowMatrix::Random(64,dim);
  UIntRowMatrix labels(64,3);
  for (int i=0; i<64; ++i) for (int j=0; j<3; ++j) labels(i,j)=(i+j)%80;
  NeighborMeanPLS::Options o; o.sample=dim == 200 ? 64 : 12;
  InspectIndex<Base> index(data.data(),80,dim,o);
  index.grow(65,4,labels,train,0.001f,2);
  index.verify(train,labels);
  // The same tree seed must produce the same model with different worker counts.
  omp_set_num_threads(1);
  InspectIndex<Base> other(data.data(),80,dim,o);
  other.grow(65,4,labels,train,1.f,2);
  assert(index.projections_ == other.projections_);
  for (int t=0; t<65; ++t) {
    assert(index.forests_[t].size() == other.forests_[t].size());
    for (size_t i=0; i<index.forests_[t].size(); ++i) {
      const auto &a=index.forests_[t][i], &b=other.forests_[t][i];
      assert(a.threshold==b.threshold && a.left==b.left && a.right==b.right && a.leaf==b.leaf);
    }
  }
  omp_set_num_threads(2);
}
void test_projection_batches() {
  for (int dim : {1,7,8,9,15,16,17,200,203}) {
    RowMatrix weights=RowMatrix::Random(65,dim);
    Eigen::VectorXf query=Eigen::VectorXf::Random(dim);
    std::vector<uint32_t> rows(65); std::iota(rows.begin(),rows.end(),0);
    float single[65], batch[65];
    for (int i=0; i<65; ++i)
      mlann_detail::compute_one_to_many(query.data(),weights.data(),dim,&rows[i],1,
                                       mlann_detail::OneToManyMetric::IP,&single[i]);
    for (int count : {1,7,8,9,15,16,17,63,64,65}) {
      mlann_detail::compute_one_to_many(query.data(),weights.data(),dim,rows.data(),count,
                                       mlann_detail::OneToManyMetric::IP,batch);
      for (int i=0; i<count; ++i) assert(single[i]==batch[i]);
    }
  }
}
void test_leading_eigenpair() {
  int selected = 0, fallbacks = 0;
  for (bool iterative : {false, true}) for (int n : {1, 8, 9, 17, 64, 200}) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Eigen::MatrixXd::Random(n, n));
    const Eigen::MatrixXd rotation = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
    for (int kind = 0; kind < 7; ++kind) {
      Eigen::VectorXd values = Eigen::VectorXd::LinSpaced(n, 0.01, 0.8);
      values[n - 1] = 1;
      if (kind == 1) values.setOnes();                      // repeated leading eigenspace
      if (kind == 2 && n > 1) values[n - 2] = 1 - 1e-14;   // nearly repeated top eigenvalue
      if (kind == 3) { values.setZero(); values[n - 1] = 1; }
      if (kind == 4) { values.setConstant(-3); values[n - 1] = 1; }
      if (kind == 5) values.setConstant(-1);
      if (kind == 6) values.setZero();
      for (bool diagonal : {false, true}) {
        Eigen::MatrixXd matrix = values.asDiagonal();
        if (!diagonal) matrix = rotation * matrix * rotation.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> reference(matrix);
        assert(reference.info() == Eigen::Success);
        for (double scale : {1e-20, 1.0, 1e20}) {
          const Eigen::MatrixXf input = (matrix * scale).cast<float>();
          LeadingEigenStats stats;
          const Eigen::VectorXd v = (iterative ? leading(input, &stats) : leading_bisect(input, &stats)).cast<double>();
          selected += stats.iterations > 0; fallbacks += stats.fallback;
          if (kind >= 5) { assert(v.isZero()); continue; }
          assert(std::abs(v.norm() - 1) < 2e-6);
          const double value = v.dot(matrix * v), top = reference.eigenvalues()[n - 1];
          assert(std::abs(value - top) < 1e-5);
          const double residual_bound = 1e-5 * matrix.norm() + 1e-6;
          assert((matrix * v - value * v).norm() < residual_bound);
          if (kind != 1 && kind != 2)
            assert(std::abs(v.dot(reference.eigenvectors().col(n - 1))) > 1 - 1e-5);
          assert(v == (iterative ? leading(input) : leading_bisect(input)).cast<double>());
        }
      }
    }
  }
  assert(selected > 0);
  std::cout << "Selected-eigenpair checks: " << selected << " solves, " << fallbacks << " fallbacks\n";
}
int main() {
  omp_set_num_threads(2);
  test_projection_batches();
  test_leading_eigenpair();
  for (int dim : {7,17,200}) verify_wide_forest<NeighborMeanPLS>(dim);
  std::mt19937_64 rng(42);
  for (int trial = 0; trial < 24; ++trial) {
    Sample a; a.k=3; a.x=Matrix::Random(17,7); a.counts.assign(9,0);
    for (int i=0; i<a.n(); ++i) for (int j : sample(9,3,rng)) {
      a.labels.push_back(j); ++a.counts[j];
    }
    Eigen::VectorXf z(a.n()); for (int i=0; i<a.n(); ++i) z[i]=int(rng()%9)-4;
    const auto fit=threshold(a,z); assert(fit.valid);
    double best=INFINITY;
    for (int i=0; i<a.n(); ++i) {
      std::vector<bool> left(a.n()); int count=0;
      for (int j=0; j<a.n(); ++j) { left[j]=z[j]<=z[i]; count+=left[j]; }
      if (count && count<a.n()) best=std::min(best,oracle(a,left));
    }
    std::vector<bool> left(a.n()); for (int i=0;i<a.n();++i) left[i]=z[i]<=fit.threshold;
    assert(std::abs(fit.loss-best)<2e-5*std::max(1.0,std::abs(best)));
    assert(oracle(a,left)<=best+2e-5*std::max(1.0,std::abs(best)));
  }
  RowMatrix data=RowMatrix::Random(80,9), constant=RowMatrix::Ones(64,9);
  UIntRowMatrix labels(64,3);
  for (int i=0;i<64;++i) for (int j=0;j<3;++j) labels(i,j)=(i+j)%80;
  Inspect index(data.data(),80,9); index.grow(2,4,labels,constant); index.verify(constant,labels);
  std::cout << "NeighborMeanPLS numerical, PAL, full-input, leaf-vote, routing and determinism checks passed\n";
}
