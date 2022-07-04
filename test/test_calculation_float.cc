#include <cassert>
#include "Tensor.h"

int main()
{
    using namespace SingleNet;

    Tensor<float, 2, 2> float_test1 = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    Tensor<float, 2, 2> float_test2 = {{0.5f, 0.6f}, {0.7f, 0.8f}};
    float float_test3 = 3.f;

    Tensor<float, 2, 2> float_test_add_result = {{0.6f, 0.8f}, {1.0f, 1.2f}};
    Tensor<float, 2, 2> float_test_sub_result = {{-0.4f, -0.4f}, {-0.4f, -0.4f}};
    Tensor<float, 2, 2> float_test_hadamard_result = {{0.05f, 0.12f}, {0.21f, 0.32f}};
    Tensor<float, 2, 2> float_test_dot_result = {{0.19f, 0.22f}, {0.43f, 0.50f}};
    Tensor<float, 2, 2> float_test_mul_result = {{0.03f, 0.06f}, {0.09f, 0.12f}};
    Tensor<float, 2, 2> float_test_div_result = {{0.1f / 3.f, 0.2f / 3.f}, {0.3f / 3.f, 0.4f / 3.f}};

    assert(float_test1[0][0] == 1);
    assert(float_test1[0][1] == 2);
    assert(float_test1[1][0] == 3);
    assert(float_test1[1][1] == 4);

    assert(float_test1 == float_test1);
    assert(float_test1 != float_test2);

    assert(float_test1 + float_test2 == float_test_add_result);
    assert(float_test1 - float_test2 == float_test_sub_result);
    assert(hadamard(float_test1, float_test2) == float_test_hadamard_result);
    assert(dot(float_test1, float_test2) == float_test_dot_result);
    assert(float_test1 * float_test3 == float_test_mul_result);
    assert(float_test1 / float_test3 == float_test_div_result);
}