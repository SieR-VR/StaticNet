#include <iostream>
#include "Tensor.h"

int main() {
    SingleNet::Tensor<float, 4, 4> t1 = {
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 },
        { 9, 10, 11, 12 },
        { 13, 14, 15, 16 },
    };

    auto t2 = t1;

    std::cout << SingleNet::Tensor<float, 4, 4>::random();
}