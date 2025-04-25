#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <fmt/core.h>
#include <fmt/ranges.h>

int main(int argc, char** argv) {

  thrust::device_vector<int> d_v(10, 1);

  thrust::fill(d_v.begin(), d_v.begin() + 7, 7);

  std::vector<int> v(5);

  thrust::copy(d_v.begin(), d_v.begin() + 5, v.begin());

  std::fill(v.begin(), v.end(), 12);

  thrust::copy(v.begin(), v.end(), d_v.begin());

  fmt::print("{}\n", d_v);

  return 0;
}
