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
    std::string path = argv[1];

    try {
        std::cout << "Loading MNIST dataset..." << std::endl;
        
        auto MNIST_Image = Image<Batch, Input>(path + "/train-images.idx3-ubyte");
        auto MNIST_Label = Label<Batch, Output>(path + "/train-labels.idx1-ubyte");

        Layer<Input, 100, Batch> dense1 = { new Dense<Input, 100, Batch>(), new Activation<100, Batch>(Defines::ReLU, Defines::ReLUDerivative) };
        Layer<100, Output, Batch> dense2 = { new Dense<100, Output, Batch>(), new Softmax<Output, Batch>() };

        for (int i = 0; i < 300; i++)
        {
            auto x = dense1(MNIST_Image[i]);
            auto result = dense2(x);

            auto y = MNIST_Label[i];
            Tensor<float, Batch> loss_list;
            for (int j = 0; j < Batch; j++)
                loss_list[j] = Defines::CrossEntropy<Output>(y[j].deref(), result[j].deref());

            auto loss = loss_list.sum() / Batch;
            std::cout << "Loss: " << loss << std::endl;

            auto grad = dense2.Backward(-(result - y), 0.1f);
            dense1.Backward(grad, 0.1f);
        }

        auto testImage = Image<Batch, Input>(path + "/t10k-images.idx3-ubyte");
        auto testLabel = Label<Batch, Output>(path + "/t10k-labels.idx1-ubyte");
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }
}
