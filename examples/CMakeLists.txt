function(add_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_link_libraries(${example_name} sycl_thrust fmt)
endfunction()

add_example(vector_example)
