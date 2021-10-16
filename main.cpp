/* Copyright 2021- SieR-VR */

#include <iostream>

#include "Core/Datasets/Datasets.h"
#include "Core/Layers.h"

int main(int argc, char *argv[])
{
    SingleNet::Vector<float, 2> a = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    SingleNet::Vector<float, 1> b = {1, 2, 3};

    auto result = dot(a, b);
    std::cout << result << std::endl;
}
