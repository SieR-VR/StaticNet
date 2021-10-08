#include <iostream>

#include "Vector.h"

using namespace std;
using namespace SingleNet;

int main(int argc, char *argv[])
{
    Vector<float, 2> Tensor_(2, 2, 1.0f);
    Tensor_[1][0] = 2.0f;

    std::cout << Tensor_ << std::endl;
    std::cout << transpose(Tensor_) << std::endl;
}