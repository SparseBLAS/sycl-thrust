#pragma once

#include <memory>
#include <type_traits>

#include <thrust/copy.h>

namespace thrust {

namespace detail {

template <typename T, typename Allocator = std::allocator<T>>
  requires(std::is_trivially_copyable_v<T> &&
           std::is_trivially_destructible_v<T>)
class vector_base {
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

  vector_base() noexcept(noexcept(Allocator())) : vector_base(Allocator()) {}

  explicit vector_base(const Allocator& allocator) noexcept
      : allocator_(allocator) {}

  explicit vector_base(size_type count, const T& value,
                       const Allocator& alloc = Allocator())
      : allocator_(alloc) {
    change_capacity_impl_(count);
    using namespace std;
    fill(data(), data() + size(), value);
  }

  explicit vector_base(size_type count, const Allocator& alloc = Allocator())
      : allocator_(alloc) {
    change_capacity_impl_(count);
    using namespace std;
    fill(data(), data() + size(), T{});
  }

  template <std::forward_iterator Iter>
  constexpr vector_base(Iter first, Iter last,
                        const Allocator& alloc = Allocator())
      : allocator_(alloc) {
    change_capacity_impl_(std::distance(first, last));
    using namespace std;
    copy(first, last, begin());
  }

  template <typename OtherAllocator>
  constexpr vector_base(const std::vector<T, OtherAllocator>& other)
      : vector_base(other.begin(), other.end()) {}

  vector_base(const vector_base& other) : allocator_(other.get_allocator()) {
    change_capacity_impl_(other.size());
    using namespace std;
    copy(other.begin(), other.end(), begin());
  }

  vector_base(const vector_base& other, const Allocator& alloc)
      : allocator_(alloc) {
    change_capacity_impl_(other.size());
    using namespace std;
    copy(other.begin(), other.end(), begin());
  }

  vector_base(vector_base&& other) noexcept
    requires(std::is_trivially_move_constructible_v<T>)
      : allocator_(other.get_allocator()) {
    data_ = other.data_;
    other.data_ = nullptr;
    size_ = other.size_;
    other.size_ = 0;
    capacity_ = other.capacity_;
    other.capacity_ = 0;
  }

  vector_base(vector_base&& other, const Allocator& alloc) noexcept
    requires(std::is_trivially_move_constructible_v<T>)
      : allocator_(alloc) {
    data_ = other.data_;
    other.data_ = nullptr;
    size_ = other.size_;
    other.size_ = 0;
    capacity_ = other.capacity_;
    other.capacity_ = 0;
  }

  vector_base(std::initializer_list<T> init,
              const Allocator& alloc = Allocator())
      : allocator_(alloc) {
    change_capacity_impl_(init.size());
    using namespace std;
    copy(init.begin(), init.end(), begin());
  }

  vector_base& operator=(const vector_base& other) {
    assign(other.begin(), other.end());
    return *this;
  }

  vector_base& operator=(vector_base&& other) {
    data_ = other.data_;
    other.data_ = nullptr;
    size_ = other.size_;
    other.size_ = 0;
    capacity_ = other.capacity_;
    other.capacity_ = 0;
    return *this;
  }

  template <std::forward_iterator Iter>
  void assign(Iter first, Iter last) {
    auto new_size = std::distance(first, last);
    reserve(new_size);
    using namespace std;
    copy(first, last, begin());
    size_ = new_size;
  }

  ~vector_base() noexcept {
    if (data() != nullptr) {
      allocator_.deallocate(data(), capacity());
    }
  }

  size_type size() const noexcept {
    return size_;
  }

  bool empty() const noexcept {
    return size() == 0;
  }

  size_type capacity() const noexcept {
    return capacity_;
  }

  pointer data() noexcept {
    return data_;
  }

  const_pointer data() const noexcept {
    return data_;
  }

  allocator_type get_allocator() const noexcept {
    return allocator_;
  }

  iterator begin() noexcept {
    return data_;
  }

  iterator end() noexcept {
    return begin() + size();
  }

  const_iterator begin() const noexcept {
    return data_;
  }

  const_iterator end() const noexcept {
    return begin() + size();
  }

  reference operator[](size_type pos) {
    return *(begin() + pos);
  }

  const_reference operator[](size_type pos) const {
    return *(begin() + pos);
  }

  void reserve(size_type new_cap) {
    if (new_cap > capacity()) {
      pointer new_data = get_allocator().allocate(new_cap);
      using namespace std;
      if (begin() != end()) {
        using namespace std;
        copy(begin(), end(), new_data);
      }
      if (data_ != nullptr) {
        get_allocator().deallocate(data_, capacity());
      }
      data_ = new_data;
      capacity_ = new_cap;
    }
  }

  void push_back(const T& value) {
    if (size() + 1 > capacity()) {
      size_type new_capacity = next_highest_power_of_two_impl_(capacity());
      reserve(new_capacity);
    }

    data()[size()] = value;
    ++size_;
  }

  void push_back(T&& value) {
    if (size() + 1 > capacity()) {
      size_type new_capacity = next_highest_power_of_two_impl_(capacity());
      reserve(new_capacity);
    }

    data()[size()] = std::move(value);
    ++size_;
  }

  bool try_push_back(const T& value) {
    if (size() + 1 <= capacity()) {
      data()[size()] = value;
      ++size_;
      return true;
    }
    return false;
  }

  void resize(size_type count, const value_type& value) {
    if (count > capacity()) {
      reserve(count);
    }
    if (count > size()) {
      using namespace std;
      fill(end(), begin() + count, value);
    }
    size_ = count;
  }

  void resize(size_type count) {
    resize(count, value_type{});
  }

private:
  // For use only inside constructors and assignment operators
  void change_capacity_impl_(size_type count) {
    if (data_ != nullptr && capacity_ != count) {
      allocator_.deallocate(data_, capacity());
    }
    size_ = capacity_ = count;
    data_ = size_ ? allocator_.allocate(count) : nullptr;
  }

  // NOTE: algorithm copied from "Bit Twiddling Hacks"
  //       (Public domain)
  constexpr size_type next_highest_power_of_two_impl_(size_type n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    if constexpr (sizeof(size_type) > 2)
      n |= n >> 16;
    if constexpr (sizeof(size_type) > 4)
      n |= n >> 32;
    n++;
    return n;
  }

  pointer data_ = nullptr;
  size_type size_ = 0;
  size_type capacity_ = 0;
  allocator_type allocator_;
};

} // namespace detail

} // namespace thrust
