#pragma once

#include <thrust/detail/vector_base.h>
#include <thrust/device_allocator.h>

namespace thrust {

template <typename T, typename Allocator = thrust::device_allocator<T>>
  requires(std::is_trivially_copyable_v<T> &&
           std::is_trivially_destructible_v<T>)
class device_vector : public detail::vector_base<T, Allocator> {
private:
  using base = detail::vector_base<T, Allocator>;

public:
  using value_type = T;
  using allocator_type = Allocator;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = typename std::allocator_traits<allocator_type>::pointer;
  using const_pointer =
      typename std::allocator_traits<allocator_type>::const_pointer;
  using reference = decltype(*std::declval<pointer>());
  using const_reference = decltype(*std::declval<const_pointer>());
  using iterator = pointer;
  using const_iterator = const_pointer;

  device_vector() noexcept(noexcept(Allocator())) : base() {}

  explicit device_vector(const Allocator& allocator) noexcept
      : base(allocator) {}

  explicit device_vector(size_type count, const T& value,
                         const Allocator& alloc = Allocator())
      : base(count, value, alloc) {}

  explicit device_vector(size_type count, const Allocator& alloc = Allocator())
      : base(count, alloc) {}

  template <std::forward_iterator Iter>
  constexpr device_vector(Iter first, Iter last,
                          const Allocator& alloc = Allocator())
      : base(first, last, alloc) {}

  template <typename OtherAllocator>
  constexpr device_vector(const std::vector<T, OtherAllocator>& other)
      : base(other) {}

  device_vector(const device_vector& other) : base(other) {}

  device_vector(const device_vector& other, const Allocator& alloc)
      : base(other, alloc) {}

  device_vector(device_vector&& other) noexcept
    requires(std::is_trivially_move_constructible_v<T>)
      : base(std::move(other)) {}

  device_vector(device_vector&& other, const Allocator& alloc) noexcept
    requires(std::is_trivially_move_constructible_v<T>)
      : base(std::move(other), alloc) {}

  device_vector(std::initializer_list<T> init,
                const Allocator& alloc = Allocator())
      : base(init, alloc) {}
};

} // namespace thrust
