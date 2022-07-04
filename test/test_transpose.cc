#include <cassert>
#include <iostream>
#include "Tensor.h"

int main() {
    using namespace SingleNet;

    static_assert(std::is_same<
        TensorUtils::transpose_helper<Tensor<float, 2, 3, 4, 5>, Transpose<1, 2, 3, 0>>::type,
        Tensor<float, 3, 4, 5, 2>
    >::value, "Transpose test 1 failed");

    static_assert(std::is_same<
        TensorUtils::transpose_helper<Tensor<float, 1, 2, 3>, Transpose<0, 1, 2>>::type,
        Tensor<float, 1, 2, 3>
    >::value, "Transpose test 2 failed");

    static_assert(!std::is_same<
        TensorUtils::transpose_helper<Tensor<float, 1, 2, 3>, Transpose<0, 1, 2>>::type,
        Tensor<float, 3, 1, 2>
    >::value, "Transpose test 3 failed");

    Tensor<float, 1, 2, 3> tensor = {{{ 1, 2, 3 }, { 4, 5, 6 }}};

    Tensor<float, 2, 1, 3> transposed_1 = tensor.transpose<1, 0, 2>();
    Tensor<float, 2, 1, 3> transposed_1_test;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 1; ++j)
            for (int k = 0; k < 3; ++k)
                transposed_1_test[i][j][k] = tensor[j][i][k];
    assert(transposed_1 == transposed_1_test);

    Tensor<float, 3, 1, 2> transposed_2 = tensor.transpose<2, 0, 1>();
    Tensor<float, 3, 1, 2> transposed_2_test;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 1; ++j)
            for (int k = 0; k < 2; ++k)
                transposed_2_test[i][j][k] = tensor[j][k][i];
    assert(transposed_2 == transposed_2_test);

    Tensor<float, 1, 2, 3> transposed_3 = tensor.transpose<0, 1, 2>();
    assert(transposed_3 == tensor);

    Tensor<float, 1, 3, 2> transposed_4 = tensor.transpose<0, 2, 1>();
    Tensor<float, 1, 3, 2> transposed_4_test;
    for (int i = 0; i < 1; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 2; k++)
                transposed_4_test[i][j][k] = tensor[i][k][j];
    assert(transposed_4 == transposed_4_test);

    Tensor<float, 3, 2, 1> transposed_5 = tensor.transpose<2, 1, 0>();
    Tensor<float, 3, 2, 1> transposed_5_test;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 1; k++)
                transposed_5_test[i][j][k] = tensor[k][j][i];
    assert(transposed_5 == transposed_5_test);

    Tensor<float, 2, 3, 1> transposed_6 = tensor.transpose<1, 2, 0>();
    Tensor<float, 2, 3, 1> transposed_6_test;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 1; k++)
                transposed_6_test[i][j][k] = tensor[k][i][j];
    assert(transposed_6 == transposed_6_test);
}