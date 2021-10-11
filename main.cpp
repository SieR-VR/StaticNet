#include <iostream>

#include "Float/Layers.h"
#include "Datasets/Datasets.h"

using namespace std;
using namespace SingleNet;

int main(int argc, char *argv[])
{
    Vector<float, 2> MNIST_Image = Datasets::MNIST::Image("./Datasets/MNIST/train-images.idx3-ubyte");
    Vector<float, 2> MNIST_Label = Datasets::MNIST::Label("./Datasets/MNIST/train-labels.idx1-ubyte");

    std::function ReLU = [](float x) { return x > 0 ? x : 0; };
    std::function ReLU_derivative = [](float x) { return x > 0 ? 1 : 0; };

    Sequential Model({
        Layer(new Dense(784, 100), new Activation(ReLU, ReLU_derivative)),
        Layer(new Dense(100, 50), new Activation(ReLU, ReLU_derivative)),
        Layer(new Dense(50, 10), new Softmax())
    });

    for (int i = 0; i < 1000; i++)
    {
        auto randomIndex = Datasets::RandomIndexes(60000, 200);
        auto Image = mask(MNIST_Image, randomIndex);
        auto Label = mask(MNIST_Label, randomIndex);

        float loss = Model.Train(Image, Label, 0.1);

        if (i % 10 == 0)
            cout << "Epoch: " << i << " Loss: " << loss << endl;
    }

    Vector<float, 2> testImage = Datasets::MNIST::Image("./Datasets/MNIST/t10k-images.idx3-ubyte");
    Vector<float, 2> testLabel = Datasets::MNIST::Label("./Datasets/MNIST/t10k-labels.idx1-ubyte");

    int correct = 0;

    for (int i = 0; i < shape(testImage)[0]; i++)
    {
        auto Image = testImage[i];
        auto Label = testLabel[i];

        auto predict = Model.Predict(Image);
        auto maxIndex_ = maxIndex(predict);

        if (Label[maxIndex_] == 1)
            correct++;
    }

    cout << "Accuracy: " << (float)correct / 10000 << endl;
}