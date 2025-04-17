cmake_minimum_required(VERSION 3.20)
project(sycl_thrust)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(sycl_thrust INTERFACE)
add_subdirectory(include)

target_compile_options(sycl_thrust INTERFACE -fsycl)
target_link_options(sycl_thrust INTERFACE -fsycl)

install(DIRECTORY include/thrust DESTINATION include)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(FetchContent)
  set(CMAKE_CXX_FLAGS "-O3 -march=native")
    FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.1.4)
  FetchContent_MakeAvailable(fmt)
  add_subdirectory(examples)
endif()
