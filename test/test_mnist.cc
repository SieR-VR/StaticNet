/* Copyright 2021- SieR-VR */

#include <iostream>
#include <time.h>

#include "Layers.h"
#include "Datasets.h"

constexpr size_t Input = 28 * 28;
constexpr size_t Output = 10;
constexpr size_t Batch = 200;

int main(int argc, char *argv[])
{   
    using namespace SingleNet;

    auto MNIST_Image = Image<200, Input>("../../../Datasets/MNIST/train-images-idx3-ubyte");
    auto MNIST_Label = Label<200, Output>("../../../Datasets/MNIST/train-labels-idx1-ubyte");

    Layer<Input, 100, 200> dense1 = { new Dense<Input, 100, Batch>(), new Activation<100, Batch>(Defines::ReLU, Defines::ReLUDerivative) };
    Layer<100, Output, 200> dense2 = { new Dense<100, Output, Batch>(), new Softmax<Output, Batch>() };

    for (int i = 0; i < 300; i++)
    {
    }

    auto testImage = Image<200, Input>("../../../Datasets/MNIST/t10k-images-idx3-ubyte");
    auto testLabel = Label<200, Output>("../../../Datasets/MNIST/t10k-labels-idx1-ubyte");
}
