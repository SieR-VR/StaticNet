#include <iostream>

#include "Modules/Linear.h"
#include "Modules/Conv2D.h"

int main()
{
    using namespace SingleNet;

    Tensor<float, 1, 5, 5> test_tensor = {{
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25},
    }};

    Conv2D<Tensor<float, 5, 5>, Tensor<float, 3, 3>> conv2d;

    std::cout << conv2d.forward(test_tensor)[0];
}