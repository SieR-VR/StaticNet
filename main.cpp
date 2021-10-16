/* Copyright 2021- SieR-VR */

#include <iostream>

#include "Core/Datasets/Datasets.h"
#include "Core/Defines/Defines.h"
#include "Core/Layers.h"

int main(int argc, char *argv[])
{
    SingleNet::Vector<float, 2> MNIST_Image = SingleNet::Datasets::MNIST::Image(
        "./Datasets/MNIST/train-images.idx3-ubyte");
    SingleNet::Vector<float, 2> MNIST_Label = SingleNet::Datasets::MNIST::Label(
        "./Datasets/MNIST/train-labels.idx1-ubyte");

    std::function ReLU = [](float x)
    { return x > 0 ? x : 0; };
    std::function ReLU_derivative = [](float x)
    { return x > 0 ? 1 : 0; };

    SingleNet::Sequential Model(
        {SingleNet::Layer(new SingleNet::Dense(784, 100),
                          new SingleNet::Activation(ReLU, ReLU_derivative)),
         SingleNet::Layer(new SingleNet::Dense(100, 50),
                          new SingleNet::Activation(ReLU, ReLU_derivative)),
         SingleNet::Layer(new SingleNet::Dense(50, 10),
                          new SingleNet::Softmax())}, SingleNet::Defines::CrossEntropy);

    for (int i = 0; i < 1000; i++)
    {
        auto randomIndex = SingleNet::Datasets::RandomIndexes(60000, 200);
        auto Image = SingleNet::mask(MNIST_Image, randomIndex);
        auto Label = SingleNet::mask(MNIST_Label, randomIndex);

        float loss = Model.Train(Image, Label, 0.1);

        if (i % 10 == 0)
            std::cout << "Epoch: " << i << " Loss: " << loss << std::endl;
    }

    SingleNet::Vector<float, 2> testImage = SingleNet::Datasets::MNIST::Image(
        "./Datasets/MNIST/t10k-images.idx3-ubyte");
    SingleNet::Vector<float, 2> testLabel = SingleNet::Datasets::MNIST::Label(
        "./Datasets/MNIST/t10k-labels.idx1-ubyte");

    int correct = 0;

    for (size_t i = 0; i < shape(testImage)[0]; i++)
    {
        auto Image = testImage[i];
        auto Label = testLabel[i];

        auto predict = Model.Predict(Image);
        auto maxIndex_ = SingleNet::maxIndex(predict);

        if (Label[maxIndex_] == 1)
            correct++;
    }

    std::cout << "Accuracy: " << static_cast<float>(correct / 10000)
              << std::endl;
}
