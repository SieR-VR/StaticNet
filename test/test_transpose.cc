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
}