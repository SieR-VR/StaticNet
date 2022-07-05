#include <iostream>
#include <cassert>
#include "Tensor.h"

int main() {
    using namespace StaticNet;

    Tensor<int, 1, 2, 3> tensor = {{{ 1, 2, 3 }, { 4, 5, 6 }}};
    
    assert(tensor[0][0][0] == 1);
    assert(tensor[0][0][1] == 2);
    assert(tensor[0][0][2] == 3);
    assert(tensor[0][1][0] == 4);
    assert(tensor[0][1][1] == 5);
    assert(tensor[0][1][2] == 6);
}