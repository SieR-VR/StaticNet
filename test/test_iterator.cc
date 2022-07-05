#include <iostream>
#include <assert.h>

#include "Tensor.h"

int main() {
    using namespace StaticNet;

    Tensor<float, 2, 2> float_test1 = {{0.1f, 0.2f}, {0.3f, 0.4f}};

    size_t idx = 0;
    for (auto it = float_test1.begin(); it != float_test1.end(); ++it, ++idx);
    assert(idx == 4);
}