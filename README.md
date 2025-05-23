# sycl-thrust

sycl-thrust is an implementation of the Thrust library using SYCL.  This allows
you to run Thrust programs using standard C++ data structures and algorithms on
Intel GPUs.

> ⚠️ **Warning:** sycl-thrust is experimental, incomplete, and under active development.

## Implemented Features

The current state implements just enough to allocate data on the device, initialize it,
and copy back and forth to the host.  It also implements some basic execution policies.

| Feature          | Status     |
|------------------|------------|
| `device_vector`  | ✅ Implemented |
| `device_allocator` | ✅ Implemented |
| `copy`           | ✅ Implemented |
| `fill`           | ✅ Implemented |
| `sort`           | ❌ Missing     |
| `reduce`         | ❌ Missing     |
| *...others*      | ❌ Missing     |

## Example

```cpp
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

#include <fmt/ranges.h>

int main(int argc, char** argv) {
  // Create a device vector initialized with 10
  // elements, all equal to 1.
  thrust::device_vector<int> d_v(10, 1);

  // Fill the value "7" into the first 7 elements.
  thrust::fill(d_v.begin(), d_v.begin() + 7, 7);

  // Create a std::vector on the CPU.
  std::vector<int> v(10);

  // Copy our device vector to the CPU.
  thrust::copy(d_v.begin(), d_v.begin() + 10, v.begin());

  std::fill(v.begin(), v.end(), 12);

  thrust::copy(v.begin(), v.end(), d_v.begin());

  fmt::print("{}\n", d_v);

  return 0;
}
```

## Default Device Behavior
By default, sycl-thrust will use the `sycl::default_selector_v` selector to pick
the default device for both `device_allocator` and the `device` execution policy.
The `sycl::default_selector_v` will generally pick a GPU device if available,
falling back to a CPU device if none is available.  This allows code to run
successfully even if no GPU is available.  If you would like to force sycl-thrust
to use a GPU, you can build with CMake flag `-DTHRUST_DEFAULT_GPU=ON`, or by directly
setting the compile directive `-DTHRUST_DEFAULT_GPU` if you are not using CMake.
This may be necessary to run on NVIDIA or AMD GPUs using the Codeplay oneAPI plugin,
as the default selector may fail to select these GPUs.
