/* Copyright 2021- SieR-VR */

#include <iostream>
#include <time.h>

#include "Core/Datasets/Datasets.h"
#include "Core/Defines/Defines.cuh"
#include "Core/Layers.h"

int main(int argc, char *argv[])
{   
    using namespace SingleNet;

    Vector<float, 2> MNIST_Image = Datasets::MNIST::Image(
        "./Datasets/MNIST/train-images.idx3-ubyte");
    Vector<float, 2> MNIST_Label = Datasets::MNIST::Label(
        "./Datasets/MNIST/train-labels.idx1-ubyte");

    SequentialCUDA Model = SequentialCUDA({
        LayerCUDA(new DenseCUDA(784, 100), new ActivationCUDA(Defines::ReLUCUDA, Defines::ReLUDerivativeCUDA)),
        LayerCUDA(new DenseCUDA(100, 10), new SoftmaxCUDA())
    }, Defines::CrossEntropy);

    for (int i = 0; i < 1000; i++)
    {
        auto randomIndex = Datasets::RandomIndexes(60000, 200);
        auto Image = mask(MNIST_Image, randomIndex);
        auto Label = mask(MNIST_Label, randomIndex);

        float loss = Model.Train(Image, Label, 0.1);

        if (i % 10 == 0)
            std::cout << "Epoch: " << i << " Loss: " << loss << std::endl;
    }

    Vector<float, 2> testImage = Datasets::MNIST::Image(
        "./Datasets/MNIST/t10k-images.idx3-ubyte");
    Vector<float, 2> testLabel = Datasets::MNIST::Label(
        "./Datasets/MNIST/t10k-labels.idx1-ubyte");

    int correct = 0;

    for (size_t i = 0; i < shape(testImage)[0]; i++)
    {
        auto Image = testImage[i];
        auto Label = testLabel[i];

        auto predict = Model.Predict(Image);
        auto maxIndex_ = maxIndex(predict);

        if (Label[maxIndex_] == 1)
            correct++;
    }

    std::cout << "Accuracy: " << static_cast<float>(correct / 10000)
              << std::endl;
}
