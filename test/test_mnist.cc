/* Copyright 2021- SieR-VR */

#include <iostream>
#include <time.h>

#include "Layers.h"
#include "Datasets.h"

constexpr size_t Input = 28 * 28;
constexpr size_t Output = 10;
constexpr size_t Batch = 100;

int main(int argc, char *argv[])
{
    using namespace SingleNet;
    std::string path = argv[1];

    try
    {
        std::cout << "Loading MNIST dataset..." << std::endl;

        auto MNIST_Image = Image<Batch, Input>(path + "/train-images.idx3-ubyte");
        auto MNIST_Label = Label<Batch, Output>(path + "/train-labels.idx1-ubyte");

        Layer<Input, 100, Batch> dense1 = {new Dense<Input, 100, Batch>(), new Activation<100, Batch>(Defines::ReLU, Defines::ReLUDerivative)};
        Layer<100, 100, Batch> dense2 = {new Dense<100, 100, Batch>(), new Activation<100, Batch>(Defines::ReLU, Defines::ReLUDerivative)};
        Layer<100, Output, Batch> dense3 = {new Dense<100, Output, Batch>(), new Softmax<Output, Batch>()};

        for (size_t epoch = 0; epoch < 10; epoch++)
        {
            printf("Epoch %lu [", epoch);

            for (int i = 0; i < 600; i++)
            {
                auto x = dense1(MNIST_Image[i]);
                x = dense2(x);
                auto result = dense3(x);

                auto y = MNIST_Label[i];

                auto grad = dense3.Backward((result - y), 0.1f);
                grad = dense2.Backward(grad, 0.1f);
                dense1.Backward(grad, 0.1f);

                if (i % 30 == 29)
                    printf("=");
            }

            printf("]\n");
        }

        auto testImage = Image<Batch, Input>(path + "/t10k-images.idx3-ubyte");
        auto testLabel = Label<Batch, Output>(path + "/t10k-labels.idx1-ubyte");

        size_t correct = 0;

        for (int i = 0; i < 100; i++)
        {
            auto x = dense1(testImage[i]);
            x = dense2(x);
            auto result = dense3(x);

            auto y = testLabel[i];

            for (int j = 0; j < Batch; j++)
                if (argmax(result[j]) == argmax(y[j]))
                    correct++;
        }

        std::cout << "Accuracy: " << correct / 100.0f << "%" << std::endl;
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
    }
}
