#include <gtest/gtest.h>

#include <random>
#include <ranges>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "util.hpp"

TEST(DeviceVector, Construct) {
  using T = int;

  for (std::size_t n : {3, 45, 823, 1000}) {
    std::vector<T> v(n);

    util::fill_random(v.begin(), v.end());

    thrust::device_vector<T> d_v1(v);
    thrust::device_vector<T> d_v2(v.begin(), v.end());

    ASSERT_EQ(v.size(), d_v1.size());
    ASSERT_EQ(v.size(), d_v2.size());

    for (std::size_t i = 0; i < v.size(); i++) {
      T value = v[i];
      T d_value1 = d_v1[i];
      T d_value2 = d_v2[i];
      ASSERT_EQ(value, d_value1);
      ASSERT_EQ(value, d_value2);
    }
  }
}

TEST(DeviceVector, Copy) {
  using T = int;

  std::mt19937 g(0);

  for (std::size_t n : {3, 45, 823, 1000, 9823, 384241, 1824981}) {
    std::vector<T> v(n);

    util::fill_random(v.begin(), v.end());

    std::vector<T> v2(v);
    thrust::device_vector<T> d_v(v);

    std::size_t n_copies = 10;

    std::uniform_int_distribution<std::size_t> first_d(0, n);

    for (std::size_t i = 0; i < n_copies; i++) {
      std::size_t first = first_d(g);
      std::uniform_int_distribution<std::size_t> last_d(first, n);
      std::size_t last = last_d(g);

      std::uniform_int_distribution<std::size_t> d_first_d(0,
                                                           n - (last - first));
      std::size_t d_first = d_first_d(g);

      std::copy(v.begin() + first, v.begin() + last, v2.begin() + d_first);

      thrust::copy(v.begin() + first, v.begin() + last, d_v.begin() + d_first);

      ASSERT_TRUE(util::is_equal(v2, d_v));
    }
  }
}
