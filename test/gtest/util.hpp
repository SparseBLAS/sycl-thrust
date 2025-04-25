#pragma once

#include <limits>
#include <random>
#include <ranges>

namespace util {

template <std::contiguous_iterator Iter, typename T = int>
void fill_random(Iter first, Iter last, T a = 0,
                 T b = std::numeric_limits<T>::max()) {
  std::mt19937 g(0);
  std::uniform_int_distribution<T> d(0, 100);

  for (auto iter = first; iter != last; ++iter) {
    *iter = d(g);
  }
}

template <typename T, typename Allocator>
bool is_equal(const std::vector<T>& a,
              const thrust::device_vector<T, Allocator>& d_b) {
  if (a.size() != d_b.size()) {
    return false;
  }

  std::vector<T> b(d_b.size());
  thrust::copy(d_b.begin(), d_b.end(), b.begin());

  for (std::size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

} // namespace util
