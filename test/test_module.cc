#include <iostream>

#include "Modules/Linear.h"
#include "Modules/Conv2D.h"

using namespace SingleNet;

class ReNet : public Module<float> {
public:
    ReNet() 
        : Module<float>("ReNet"),
          conv1(this),
          conv2(this),
          fc1(this)
    {}

private:
    Conv2D<Tensor<float, 28, 28>, Tensor<float, 14, 14>> conv1;
    Conv2D<Tensor<float, 14, 14>, Tensor<float, 5, 5>> conv2;
    Linear<Tensor<float, 25>, Tensor<float, 10>> fc1;
};

int main()
{
    Tensor<float, 1, 5, 5> test_tensor = {{
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25},
    }};

    ReNet renet;

    print(renet);
}