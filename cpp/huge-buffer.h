#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace mlann_detail {
// Storage for trivial values. Linux allocations request transparent huge pages;
// other platforms and failed mappings use ordinary vector storage. Resizing to
// a different size discards contents. No process-wide memory policy is changed.
template <class T>
class HugeBuffer {
  static_assert(std::is_trivial<T>::value, "HugeBuffer requires trivial values");
  T *data_ = nullptr;
  size_t size_ = 0, mapped_ = 0;
  std::vector<T> fallback_;

  void release() {
#ifdef __linux__
    if (mapped_) munmap(data_, mapped_);
#endif
    data_ = nullptr;
    size_ = mapped_ = 0;
    fallback_.clear();
  }

 public:
  HugeBuffer() = default;
  HugeBuffer(const HugeBuffer &other) {
    resize(other.size_);
    if (size_) std::copy_n(other.data_, size_, data_);
  }
  HugeBuffer(HugeBuffer &&other) noexcept { swap(other); }
  HugeBuffer &operator=(const HugeBuffer &other) {
    if (this != &other) {
      HugeBuffer copy(other);
      swap(copy);
    }
    return *this;
  }
  HugeBuffer &operator=(HugeBuffer &&other) noexcept {
    if (this != &other) {
      release();
      swap(other);
    }
    return *this;
  }
  ~HugeBuffer() { release(); }

  void swap(HugeBuffer &other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(mapped_, other.mapped_);
    fallback_.swap(other.fallback_);
  }

  void resize(size_t n) {
    if (n == size_) return;
    release();
    if (!n) return;
#ifdef __linux__
    constexpr size_t huge = 2 * 1024 * 1024;
    if (n > (SIZE_MAX - 2 * huge) / sizeof(T)) throw std::bad_alloc();
    if (n * sizeof(T) >= huge) {
      const size_t bytes = (n * sizeof(T) + huge - 1) & ~(huge - 1);
      void *raw = mmap(nullptr, bytes + huge, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (raw != MAP_FAILED) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(raw);
        const uintptr_t aligned = (base + huge - 1) & ~uintptr_t(huge - 1);
        const size_t prefix = aligned - base, suffix = huge - prefix;
        if (prefix) munmap(raw, prefix);
        if (suffix) munmap(reinterpret_cast<void *>(aligned + bytes), suffix);
        data_ = reinterpret_cast<T *>(aligned);
        mapped_ = bytes;
        size_ = n;
        // Advice can be declined by the kernel; ordinary pages remain correct.
        madvise(data_, mapped_, MADV_HUGEPAGE);
        std::uninitialized_default_construct_n(data_, n);
        return;
      }
    }
#endif
    fallback_.resize(n);
    data_ = fallback_.data();
    size_ = n;
  }

  T *data() { return data_; }
  const T *data() const { return data_; }
  size_t size() const { return size_; }
  T &operator[](size_t i) { return data_[i]; }
  const T &operator[](size_t i) const { return data_[i]; }
};

// Promote only complete huge-page spans inside the existing corpus allocation.
// No corpus copy or index metadata is created; unsupported advice is harmless.
inline void promote_existing_corpus_pages(const void *data, size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  constexpr uintptr_t page = 2 * 1024 * 1024;
  const uintptr_t address = reinterpret_cast<uintptr_t>(data);
  if (bytes > UINTPTR_MAX - address || address > UINTPTR_MAX - (page - 1)) return;
  const uintptr_t begin = (address + page - 1) & ~(page - 1);
  const uintptr_t end = (address + bytes) & ~(page - 1);
  if (end <= begin) return;
  madvise(reinterpret_cast<void *>(begin), end - begin, MADV_HUGEPAGE);
#ifdef MADV_COLLAPSE
  madvise(reinterpret_cast<void *>(begin), end - begin, MADV_COLLAPSE);
#endif
#endif
}
}  // namespace mlann_detail
