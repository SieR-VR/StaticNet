#include <cassert>
#include "Tensor.h"

int main()
{
    using namespace SingleNet;

    Tensor<int, 2, 2> int_test1 = {{1, 2}, {3, 4}};
    Tensor<int, 2, 2> int_test2 = {{5, 6}, {7, 8}};
    int int_test3 = 3;

    Tensor<int, 2, 2> int_test_add_result = {{6, 8}, {10, 12}};
    Tensor<int, 2, 2> int_test_sub_result = {{-4, -4}, {-4, -4}};
    Tensor<int, 2, 2> int_test_hadamard_result = {{5, 12}, {21, 32}};
    Tensor<int, 2, 2> int_test_dot_result = {{19, 22}, {43, 50}};
    Tensor<int, 2, 2> int_test_mul_result = {{3, 6}, {9, 12}};
    Tensor<int, 2, 2> int_test_div_result = {{0, 0}, {1, 1}};
    Tensor<int, 2> int_test_reduce_result = {4, 6};

    assert(int_test1[0][0] == 1);
    assert(int_test1[0][1] == 2);
    assert(int_test1[1][0] == 3);
    assert(int_test1[1][1] == 4);

    assert(int_test1 == int_test1);
    assert(int_test1 != int_test2);

    assert(int_test1 + int_test2 == int_test_add_result);
    assert(int_test1 - int_test2 == int_test_sub_result);
    assert(hadamard(int_test1, int_test2) == int_test_hadamard_result);
    assert(dot(int_test1, int_test2) == int_test_dot_result);
    assert(int_test1 * int_test3 == int_test_mul_result);
    assert(int_test1 / int_test3 == int_test_div_result);
    assert(int_test1.reduce() == int_test_reduce_result);
}