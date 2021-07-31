#include <iostream>

#include "Tools/load_mnist.h"
#include "Tools/ProgressBar.h"
#include "Tools/Defines.h"
#include "Float/MultiLayerClassification.h"


using namespace std;

int main(int argc, char *argv[])
{
    try {
        Vector2D<float> mnist_images = get_mnist_image_float("/home/sier/MyFixNet/Datasets/train-images-idx3-ubyte");
        Vector2D<int> mnist_labels = get_mnist_label("/home/sier/MyFixNet/Datasets/train-labels-idx1-ubyte");

        cout << "Mnist Image Data Size: " << mnist_images.shape().x << endl;
        cout << "Mnist Image Dataset Size: " << mnist_images.shape().y << endl;

        cout << "Mnist Label Dataset Size: " << mnist_labels.shape().x << endl;
        cout << "Mnist Label Data Size: " << mnist_labels.shape().y << endl;

        size_t ImageSize = mnist_images.shape().x;
        size_t LabelSize = mnist_labels.shape().y;
        size_t DatasetNum = mnist_images.shape().y;

        MultiLayerClassification MyNet(Vector1D<size_t>({ImageSize, 100, 200, LabelSize}), Defines::CrossEntropyError);
        size_t BatchNum = 600;
        size_t EpochNum = 1;
        size_t BatchDataNum = DatasetNum / BatchNum;

        ProgressBar Bar(BatchNum);
        float LearningRate = 1.0f;

        for (size_t i = 0; i < EpochNum; i++) 
        {
            for (size_t j = 0; j < BatchNum; j++)
            {
                Vector2D<float> MiniBatchImages = mnist_images.slice({0, j * BatchDataNum}, {ImageSize, (j + 1) * BatchDataNum});
                Vector2D<int> MiniBatchLabels = mnist_labels.slice({j * BatchDataNum, 0}, {(j + 1) * BatchDataNum, LabelSize});

                float loss = MyNet.train(LearningRate, MiniBatchImages, MiniBatchLabels);
                Bar.update(j + 1, "Epoch: " + to_string(i + 1) + " Loss: " + to_string(loss));
                cout << endl;
            }
        }

        Vector2D<float> test_mnist_images = get_mnist_image_float("/home/sier/MyFixNet/Datasets/t10k-images-idx3-ubyte");
        Vector2D<int> test_mnist_labels = get_mnist_label("/home/sier/MyFixNet/Datasets/t10k-labels-idx1-ubyte");

        size_t TestImageSize = test_mnist_images.shape().x;
        size_t TestLabelSize = test_mnist_labels.shape().y;
        size_t TestDatasetNum = test_mnist_images.shape().y;

        size_t CorrectNum = 0;

        for(size_t i = 0; i < TestDatasetNum; i++)
        {
            Vector1D<float> estimate = MyNet.Classify(test_mnist_images[i]);
            if(i % 100 == 0) cout << "Estimate: " << estimate << endl;
            
            float max = -100.0f;
            size_t max_index = 0;

            for(size_t j = 0; j < estimate.shape().x; j++)
            {
                if(estimate[j] > max)
                {
                    max = estimate[j];
                    max_index = j;
                }
            }

            if(test_mnist_labels[max_index][i] == 1) CorrectNum++;
        }

        cout << "Accuracy: " << (float) CorrectNum / TestDatasetNum << endl;
    }
    catch (const exception &e) {
        cout << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}