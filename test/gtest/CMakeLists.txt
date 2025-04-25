enable_testing()

  add_executable(
    thrust-tests
    device_vector_test.cpp
  )

target_link_libraries(thrust-tests sycl_thrust fmt GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(thrust-tests)
