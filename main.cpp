#include <iostream>

#include "Float/Layers.h"

using namespace std;
using namespace SingleNet;

int main(int argc, char *argv[])
{
    std::function ReLU = [](float x) { return x > 0 ? x : 0; };
    std::function ReLU_derivative = [](float x) { return x > 0 ? 1 : 0; };

    Sequential net({
        Layer(new Net(2, 2), new Activation(ReLU, ReLU_derivative)),
        Layer(new Net(2, 2), new Sigmoid())
    });

    Vector<float, 2> XOR_Data = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    Vector<float, 2> XOR_Labels = {
        {0, 1},
        {1, 0},
        {1, 0},
        {0, 1}
    };

    for (int i = 0; i < 1000; i++)
    { 
        net.Train(XOR_Data, XOR_Labels, 1);
    }

    for (int i = 0; i < XOR_Data.size(); i++)
    {
        std::cout << "XOR_Data[" << i << "]: " << XOR_Data[i] << " -> " << net.Predict(XOR_Data[i]) << std::endl;
    }
}