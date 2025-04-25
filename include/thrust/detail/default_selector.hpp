#pragma once

#include <sycl/sycl.hpp>

namespace thrust {

#ifndef SYCL_THRUST_DEFAULT_GPU
inline auto default_selector_v = sycl::default_selector_v;
#else
inline auto default_selector_v = sycl::gpu_selector_v;
#endif

}; // namespace thrust
