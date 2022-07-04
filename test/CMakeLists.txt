include_directories(${SHADERPP_INCLUDE_DIR})
link_libraries(singlenet)

add_executable(test_mnist test_mnist.cc)
add_executable(test_calculation_int test_calculation_int.cc)
add_executable(test_calculation_float test_calculation_float.cc)
add_executable(test_module test_module.cc)
add_executable(test_transpose test_transpose.cc)

add_test(test_mnist test_mnist)
add_test(test_calculation_int test_calculation_int)
add_test(test_calculation_float test_calculation_float)
add_test(test_module test_module)
add_test(test_transpose test_transpose)
