#include <iostream>
#include <cassert>

#include "Tensor.h"

int main() {
    using namespace StaticNet;
    
    Tensor<int, 2, 2> test = {{1, 2}, {3, 4}};
    
    Tensor<int, 4, 4> test_padded = pad2d<1>(test);
    Tensor<int, 4, 4> test_padded_correct = {{0, 0, 0, 0}, {0, 1, 2, 0}, {0, 3, 4, 0}, {0, 0, 0, 0}};

    assert(test_padded == test_padded_correct);

    Tensor<int, 1, 1> test_pooled = pool<int, 2, 1>(test, [](const Tensor<int, 2, 2> &t) {
        int avg = 0;
        for (size_t i = 0; i < 2; i++)
            for (size_t j = 0; j < 2; j++)
                avg += t[i][j];
        return avg / 4;
    });

    Tensor<int, 1, 1> test_pooled_correct = {{2}};
    assert(test_pooled == test_pooled_correct);

    std::function<Tensor<int, 2, 2>(int)> unpool_func = [](int x) {
        return Tensor<int, 2, 2>(x);
    };

    Tensor<int, 4, 4> test_unpooled = unpool(test, unpool_func);
    Tensor<int, 4, 4> test_unpooled_correct = {{1, 1, 2, 2}, {1, 1, 2, 2}, {3, 3, 4, 4}, {3, 3, 4, 4}};
    
    assert(test_unpooled == test_unpooled_correct);

    auto test_reshaped = test.reshape<4>();
    size_t test_argmax = argmax(test_reshaped);
    assert(test_argmax == 3);
    assert(test_reshaped[test_argmax] == 4);

    Tensor<int, 1, 1, 3, 3> test_4d = {{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}};

    Tensor<int, 4, 4> test_im2col = im2col<2>(test_4d);
    Tensor<int, 4, 4> test_im2col_correct = {{ 1, 2, 4, 5 }, { 2, 3, 5, 6 }, { 4, 5, 7, 8 }, { 5, 6, 8, 9 }};
    assert(test_im2col == test_im2col_correct);
}