/* Copyright 2021- SieR-VR */

#include <iostream>
#include <time.h>

#include "Models/LeNet.h"
#include "Models/AffineNet.h"
#include "Datasets.h"

constexpr size_t Input = 28 * 28;
constexpr size_t Output = 10;
constexpr size_t Batch = 200;

int main(int argc, char *argv[])
{
    using namespace SingleNet;
    std::string path = argv[1];

    try
    {
        std::cout << "Loading MNIST dataset..." << std::endl;

        auto MNIST_Image = Image<Batch, Input>(path + "/train-images.idx3-ubyte");
        auto MNIST_Label = Label<Batch, Output>(path + "/train-labels.idx1-ubyte");

        LeNet model;
        // AffineNet model;
        print(model);

        for (size_t epoch = 0; epoch < 3; epoch++)
        {
            printf("Epoch %zu [", epoch);

            for (int i = 0; i < 60000 / Batch; i++)
            {
                auto x = model.forward(MNIST_Image[i].template reshape<Batch, 1, 28, 28>());
                // auto x = model.forward(MNIST_Image[i]);
                auto result = x.apply(Defines::Softmax<Output>);

                auto y = MNIST_Label[i];
                float loss = 0.0f;
                for (size_t j = 0; j < Batch; j++)
                    loss += Defines::CrossEntropy<Output>(y[j], result[j]) / (float)Batch;
                model.backward((result - y), 0.03f); 

                if (i % 10 == 9)
                    printf("=");   
            }

            printf("]\n");
        }

        auto testImage = Image<Batch, Input>(path + "/t10k-images.idx3-ubyte");
        auto testLabel = Label<Batch, Output>(path + "/t10k-labels.idx1-ubyte");

        size_t correct = 0;

        for (int i = 0; i < 10000 / Batch; i++)
        {
            auto result = model.forward(testImage[i].template reshape<Batch, 1, 28, 28>());
            // auto result = model.forward(testImage[i]);
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
