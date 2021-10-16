/* Copyright 2021- SieR-VR */

#include <iostream>

#include "Core/Datasets/Datasets.h"
#include "Core/Layers.h"

int main(int argc, char *argv[])
{
    SingleNet::Vector<float, 2> a = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12} };
    SingleNet::Vector<float, 2> b = { {1, 2, 3, 4, 5}, {5, 6, 7, 8, 9}, {9, 10, 11, 12, 13} };

    auto result = dot(a, b);
    std::cout << result << std::endl; // [[38, 44, 50, 56, 62], [83, 98, 113, 128, 143], [128, 152, 176, 200, 224], [173, 206, 239, 272, 305]]
}
