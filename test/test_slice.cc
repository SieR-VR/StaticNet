#include <iostream>
#include <cassert>

#include "Tensor.h"

int main() {
    using namespace StaticNet;

    Tensor<int, 4, 4> test = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    auto window_test1 = test.slice<3, 3>({ 0, 0 });
    Tensor<int, 3, 3> window_test1_correct = {
        {1, 2, 3},
        {5, 6, 7},
        {9, 10, 11}
    };
    assert(window_test1 == window_test1_correct);

    auto window_test2 = test.slice<3, 3>({ 1, 1 });
    Tensor<int, 3, 3> window_test2_correct = {
        {6, 7, 8},
        {10, 11, 12},
        {14, 15, 16}
    };
    assert(window_test2 == window_test2_correct);

    auto window_test_3 = window_test1.slice<2, 2>({ 1, 1 });
    Tensor<int, 2, 2> window_test_3_correct = {
        {6, 7},
        {10, 11}
    };
    assert(window_test_3 == window_test_3_correct);

    auto window_test_4 = window_test1.slice<2, 2>({ 0, 0 });
    Tensor<int, 2, 2> window_test_4_correct = {
        {1, 2},
        {5, 6}
    };
    assert(window_test_4 == window_test_4_correct);

    auto window_test_5 = window_test1.slice<2, 2>({ 0, 1 });
    Tensor<int, 2, 2> window_test_5_correct = {
        {2, 3},
        {6, 7}
    };
    assert(window_test_5 == window_test_5_correct);

    auto window_reshape_test = window_test_5.reshape<4>();
    Tensor<int, 4> window_reshape_test_correct = { 2, 3, 6, 7 };
    assert(window_reshape_test == window_reshape_test_correct);
}