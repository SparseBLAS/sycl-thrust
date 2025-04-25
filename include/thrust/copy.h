#pragma once

#include <sycl/sycl.hpp>

#include <iterator>
#include <type_traits>

#include <thrust/detail/get_pointer_device.hpp>
#include <thrust/device_ptr.h>

namespace thrust {

template <std::contiguous_iterator I, typename T>
  requires(std::is_same_v<std::iter_value_t<I>, std::remove_const_t<T>> &&
           std::is_trivially_copyable_v<T>)
void copy(I first, I last, device_ptr<T> d_first) {
  sycl::queue q = __detail::get_pointer_queue(d_first.get());

  q.memcpy(d_first.get(), std::to_address(first),
           std::distance(first, last) * sizeof(T))
      .wait();
}

template <typename T, std::contiguous_iterator O>
  requires(std::is_same_v<std::iter_value_t<O>, std::remove_const_t<T>> &&
           std::is_trivially_copyable_v<T>)
void copy(device_ptr<T> first, device_ptr<T> last, O d_first) {
  sycl::queue q = __detail::get_pointer_queue(first.get());

  q.memcpy(std::to_address(d_first), first.get(),
           std::distance(first, last) * sizeof(T))
      .wait();
}

template <typename ExecutionPolicy, std::contiguous_iterator I, typename T>
  requires(std::is_same_v<std::iter_value_t<I>, std::remove_const_t<T>> &&
           std::is_trivially_copyable_v<T>)
void copy(ExecutionPolicy&& policy, I first, I last, device_ptr<T> d_first) {
  policy.get_queue()
      .memcpy(d_first.get(), std::to_address(first),
              std::distance(first, last) * sizeof(T))
      .wait();
}

template <typename ExecutionPolicy, typename T, std::contiguous_iterator O>
  requires(std::is_same_v<std::iter_value_t<O>, std::remove_const_t<T>> &&
           std::is_trivially_copyable_v<T>)
void copy(ExecutionPolicy&& policy, device_ptr<T> first, device_ptr<T> last,
          O d_first) {
  policy.get_queue()
      .memcpy(std::to_address(d_first), first.get(),
              std::distance(first, last) * sizeof(T))
      .wait();
}

} // namespace thrust
