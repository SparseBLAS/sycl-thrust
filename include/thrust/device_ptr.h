#pragma once

#include <type_traits>

#include <thrust/device_reference.h>

namespace thrust {

template <typename T>
  requires(std::is_trivially_copyable_v<T> || std::is_void_v<T>)
class device_ptr {
public:
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = device_ptr<T>;
  using const_pointer = device_ptr<std::add_const_t<T>>;
  using nonconst_pointer = device_ptr<std::remove_const_t<T>>;
  using iterator_category = std::random_access_iterator_tag;
  using reference = device_reference<T>;

  constexpr device_ptr(T* pointer) noexcept : pointer_(pointer) {}
  constexpr device_ptr() noexcept = default;
  constexpr ~device_ptr() noexcept = default;
  constexpr device_ptr(const device_ptr&) noexcept = default;
  constexpr device_ptr& operator=(const device_ptr&) noexcept = default;

  constexpr device_ptr(std::nullptr_t) noexcept : pointer_(nullptr) {}

  constexpr device_ptr& operator=(std::nullptr_t) noexcept {
    pointer_ = nullptr;
    return *this;
  }

  constexpr operator device_ptr<void>() const noexcept
    requires(!std::is_void_v<T>)
  {
    return device_ptr<void>(reinterpret_cast<void*>(pointer_));
  }

  constexpr operator device_ptr<const void>() const noexcept
    requires(!std::is_void_v<T>)
  {
    return device_ptr<const void>(reinterpret_cast<const void*>(pointer_));
  }

  constexpr operator const_pointer() const noexcept
    requires(!std::is_const_v<T>)
  {
    return const_pointer(pointer_);
  }

  constexpr bool operator==(std::nullptr_t) const noexcept {
    return pointer_ == nullptr;
  }
  constexpr bool operator!=(std::nullptr_t) const noexcept {
    return pointer_ != nullptr;
  }

  constexpr bool operator==(const device_ptr&) const noexcept = default;
  constexpr bool operator!=(const device_ptr&) const noexcept = default;

  constexpr pointer operator+(difference_type offset) const noexcept {
    return pointer(pointer_ + offset);
  }
  constexpr pointer operator-(difference_type offset) const noexcept {
    return pointer(pointer_ - offset);
  }

  constexpr difference_type operator-(const_pointer other) const noexcept
    requires(!std::is_const_v<T>)
  {
    return pointer_ - other.pointer_;
  }
  constexpr difference_type operator-(pointer other) const noexcept {
    return pointer_ - other.pointer_;
  }

  constexpr bool operator<(const_pointer other) const noexcept {
    return pointer_ < other.pointer_;
  }
  constexpr bool operator>(const_pointer other) const noexcept {
    return pointer_ > other.pointer_;
  }
  constexpr bool operator<=(const_pointer other) const noexcept {
    return pointer_ <= other.pointer_;
  }
  constexpr bool operator>=(const_pointer other) const noexcept {
    return pointer_ >= other.pointer_;
  }

  constexpr pointer& operator++() noexcept {
    ++pointer_;
    return *this;
  }

  constexpr pointer operator++(int) noexcept {
    pointer other = *this;
    ++(*this);
    return other;
  }

  constexpr pointer& operator--() noexcept {
    --pointer_;
    return *this;
  }

  constexpr pointer operator--(int) noexcept {
    pointer other = *this;
    --(*this);
    return other;
  }

  constexpr pointer& operator+=(difference_type offset) noexcept {
    pointer_ += offset;
    return *this;
  }

  constexpr pointer& operator-=(difference_type offset) noexcept {
    pointer_ -= offset;
    return *this;
  }

  constexpr reference operator*() const noexcept {
    return reference(pointer_);
  }

  constexpr reference operator[](difference_type offset) const noexcept {
    return reference(*(*this + offset));
  }

  constexpr T* get() const noexcept {
    return pointer_;
  }

  constexpr friend pointer operator+(difference_type n, pointer iter) {
    return iter + n;
  }

  friend const_pointer;
  friend nonconst_pointer;

private:
  T* pointer_;
};

} // namespace thrust
