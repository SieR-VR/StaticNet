#include <iostream>

#include "Vector.h"
#include "Datasets/Datasets.h"
#include "Tools/ProgressBar.h"
#include "Float/Aww.h"

using namespace std;
using namespace SingleNet;

int main(int argc, char *argv[])
{
    try
    {
        // Dataset
        auto Mnist_Image_Train = Datasets::MNIST::Image("./Datasets/MNIST/train-images.idx3-ubyte");
        auto Mnist_Label_Train = Datasets::MNIST::Label("./Datasets/MNIST/train-labels.idx1-ubyte");

        size_t InputSize = Mnist_Image_Train.shape()[0];
        size_t OutputSize = Mnist_Label_Train.shape()[0];
        size_t DatasetSize = Mnist_Image_Train.shape()[1];

        // Network
        auto Network = TwoLayerNet(InputSize, 100, OutputSize);
        auto Bar = Tools::ProgressBar(DatasetSize);

        // Training
        size_t Batch = 100;
        size_t Epoch = 5;
        size_t BatchData = DatasetSize / Batch;

        float LearningRate = 0.1;

        for (size_t Epoch_ = 0; Epoch_ < Epoch; ++Epoch_)
        {
            for (size_t Batch_ = 0; Batch_ < BatchData; ++Batch_)
            {
                // Get Data
                auto Data = Mnist_Image_Train.slice({ 0, BatchData * Epoch_}, { InputSize, BatchData * (Epoch_ + 1)});
                auto Label = Mnist_Label_Train.slice({ 0, BatchData * Epoch_}, { OutputSize, BatchData * (Epoch_ + 1)});

                // Train
                auto grad = Network.gradient(Data, Label);

                // Update
                Network.affine1.W -= grad.dW1 * LearningRate;
                Network.affine1.b -= grad.dB1 * LearningRate;

                Network.affine2.W -= grad.dW2 * LearningRate;
                Network.affine2.b -= grad.dB2 * LearningRate;

                // Show
                Bar.update(Batch_,  "Epoch: " + to_string(Epoch_) + " Batch: " + to_string(Batch_) + "Loss: " + to_string(Network.loss(Data, Label)));
                cout << endl;
            }
        }
    }
    catch (const Tools::StackTrace &e)
    {
        e.print();
    }
}