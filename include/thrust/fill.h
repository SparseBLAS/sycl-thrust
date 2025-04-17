#pragma once

#include <sycl/sycl.hpp>

#include <type_traits>

#include <thrust/detail/get_pointer_device.hpp>
#include <thrust/device_ptr.h>

namespace thrust {

template <typename T>
  requires(std::is_trivially_copyable_v<T>)
void fill(device_ptr<T> first, device_ptr<T> last, const T& value) {
  sycl::queue q = __detail::get_pointer_queue(first.get());

  q.fill(first.get(), value, std::distance(first, last)).wait();
}

template <typename ExecutionPolicy, typename T>
  requires(std::is_trivially_copyable_v<T>)
void fill(ExecutionPolicy&& policy, device_ptr<T> first, device_ptr<T> last,
          const T& value) {
  policy.get_queue()
      .fill(first.get(), value, std::distance(first, last))
      .wait();
}

} // namespace thrust
