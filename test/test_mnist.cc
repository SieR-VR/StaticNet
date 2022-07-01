/* Copyright 2021- SieR-VR */

#include <iostream>
#include <time.h>

#include "Models/ReNet.h"
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

        ReNet model;

        for (size_t epoch = 0; epoch < 10; epoch++)
        {
            printf("Epoch %lu [", epoch);

            for (int i = 0; i < 600; i++)
            {
                auto result = model.forward(MNIST_Image[i].reshape<Batch, 28, 28>().deref());
                auto y = MNIST_Label[i];
                model.backward(result - y, 0.1f); 

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
            auto result = model.forward(testImage[i].reshape<Batch, 28, 28>().deref());
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
