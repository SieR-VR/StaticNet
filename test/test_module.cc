#include <iostream>

#include "Models/LeNet.h"

int main()
{
    using namespace StaticNet;

    Tensor<float, 1, 5, 5> test_tensor = {{
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25},
    }};
}