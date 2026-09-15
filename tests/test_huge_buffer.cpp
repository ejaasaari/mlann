#include "../cpp/huge-buffer.h"
#include <cassert>

int main() {
  mlann_detail::HugeBuffer<float> original;
  original.resize(1024 * 1024 + 17);
  for (size_t i = 0; i < original.size(); ++i) original[i] = float(i % 1023);
  auto copied = original;
  assert(copied.data() != original.data());
  mlann_detail::HugeBuffer<float> moved = std::move(copied);
  for (size_t i = 0; i < moved.size(); ++i) assert(moved[i] == original[i]);
  copied = moved;
  copied.resize(7);
  std::fill_n(copied.data(), 7, 3.f);
  original = std::move(copied);
  assert(original.size() == 7);
  for (int i = 0; i < 7; ++i) assert(original[i] == 3.f);
  std::vector<float> corpus(2 * 1024 * 1024 + 17);
  for (size_t i = 0; i < corpus.size(); ++i) corpus[i] = float(i % 1023);
  const auto *address = corpus.data();
  const auto capacity = corpus.capacity();
  mlann_detail::promote_existing_corpus_pages(corpus.data() + 7,
      (corpus.size() - 14) * sizeof(float));
  assert(corpus.data() == address && corpus.capacity() == capacity);
  for (size_t i = 0; i < corpus.size(); ++i) assert(corpus[i] == float(i % 1023));
  moved.resize(0);
  assert(moved.size() == 0);
}
