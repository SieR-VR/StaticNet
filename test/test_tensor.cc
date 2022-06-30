#include <iostream>
#include "Tensor.h"

int main() {
    auto t = SingleNet::Tensor<float, 4, 4>({
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 },
        { 9, 10, 11, 12 },
        { 13, 14, 15, 16 },
    });

    std::cout << t;
}