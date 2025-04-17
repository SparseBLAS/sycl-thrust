#pragma once

#include <sycl/sycl.hpp>

#include <type_traits>

#include <thrust/detail/get_pointer_device.hpp>

namespace thrust {

template <typename T>
  requires(std::is_trivially_copyable_v<T> || std::is_void_v<T>)
class device_reference {
public:
  device_reference() = delete;
  ~device_reference() = default;

  constexpr device_reference(const device_reference&) = default;

  constexpr device_reference(T* pointer) : pointer_(pointer) {}

  constexpr operator T() const {
#ifdef __SYCL_DEVICE_ONLY__
    return *pointer_;
#else
    auto&& q = __detail::get_pointer_queue(pointer_);
    char buffer[sizeof(T)] __attribute__((aligned(sizeof(T))));
    q.memcpy(reinterpret_cast<std::remove_const_t<T>*>(buffer), pointer_,
             sizeof(T))
        .wait();
    return *reinterpret_cast<T*>(buffer);
#endif
  }

  constexpr device_reference operator=(const T& value) const
    requires(!std::is_const_v<T>)
  {
#ifdef __SYCL_DEVICE_ONLY__
    *pointer_ = value;
#else
    auto&& q = __detail::get_pointer_queue(pointer_);
    q.memcpy(pointer_, &value, sizeof(T)).wait();
#endif
    return *this;
  }

  constexpr device_reference operator=(const device_reference& other) const {
#ifdef __SYCL_DEVICE_ONLY__
    *pointer_ = *other.pointer_;
#else
    T value = other;
    *this = value;
#endif

    return *this;
  }

private:
  T* pointer_;
};

} // namespace thrust

#if __has_include(<fmt/ostream.h>)

#include <fmt/ostream.h>

template <typename T>
struct fmt::formatter<thrust::device_reference<T>> : fmt::ostream_formatter {};

#endif
