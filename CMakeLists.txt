cmake_minimum_required(VERSION 3.20)
project(sycl_thrust)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SYCL_THRUST_DEFAULT_GPU "Force sycl-thrust to select a GPU by default rather than potentially selecting a CPU device." OFF)

add_library(sycl_thrust INTERFACE)
add_subdirectory(include)

target_compile_options(sycl_thrust INTERFACE -fsycl)
target_link_options(sycl_thrust INTERFACE -fsycl)

if (SYCL_THRUST_DEFAULT_GPU)
  target_compile_options(sycl_thrust INTERFACE -DSYCL_THRUST_DEFAULT_GPU)
endif()

install(DIRECTORY include/thrust DESTINATION include)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(FetchContent)
    FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.1.4)
  FetchContent_MakeAvailable(fmt)
  add_subdirectory(examples)
endif()
