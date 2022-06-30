include_directories(${SHADERPP_INCLUDE_DIR})
link_libraries(singlenet)

add_executable(test_mnist test_mnist.cc)
add_executable(test_tensor test_tensor.cc)
add_executable(test_module test_module.cc)

include(CTest)
add_test(test_mnist test_mnist)
add_test(test_tensor test_tensor)
add_test(test_module test_module)
