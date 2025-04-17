#pragma once

#include <sycl/sycl.hpp>

namespace thrust {

class execution_policy {
public:
  execution_policy() : queue_(sycl::default_selector_v) {}

  execution_policy(const sycl::queue& queue) : queue_(queue) {}

  template <typename Selector>
  execution_policy(Selector&& selector)
      : queue_(std::forward<Selector>(selector)) {}

  sycl::queue& get_queue() {
    return queue_;
  }

  const sycl::queue& get_queue() const {
    return queue_;
  }

  sycl::device get_device() const {
    return queue_.get_device();
  }

  sycl::context get_context() const {
    return queue_.get_context();
  }

private:
  sycl::queue queue_;
};

// TODO: support allocators, setting stream with par.

inline execution_policy device(sycl::default_selector_v);
inline execution_policy host(sycl::cpu_selector_v);
inline execution_policy par(sycl::default_selector_v);

} // namespace thrust
