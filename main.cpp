#include <iostream>
#include <vector>
#include <random>
#include <time.h>

#include "Float/LogisticClassification.h"
#include "Tools/load_mnist.h"
#include "Tools/vector_helper.h"

using namespace std;

int main()
{
    int image_num, label_num, image_size;
    auto data_ = read_mnist_images("C:/Users/nwh63/Desktop/MyFixNet/train-images.idx3-ubyte", image_num, image_size);
    auto labels_ = read_mnist_labels("C:/Users/nwh63/Desktop/MyFixNet/train-labels.idx1-ubyte", label_num);

    vector<vector<float>> data;

    for(int i = 0; i < image_num; i++) {
        vector<float> temp;
        for(int j = 0; j < image_size; j++) temp.push_back(((int) data_[i][j]) * 0.78125f);
        data.push_back(temp);
    }

    vector<vector<bool>> labels;
    for(int i = 0; i < 10; i++) {
        vector<bool> temp;
        for(int j = 0; j < label_num; j++) temp.push_back(labels_[j] == i);
        labels.push_back(temp);
    }

    LogisticClassification myNet(image_size, 10);

    int num_batch = 300;
    int num_batch_data = image_num / num_batch;
    int num_epoch = 10;
    float learning_rate = 0.3f;

    for(int i = 0; i < num_epoch; i++) {
        for(int j = 0; j < num_batch; j++) {
            if(j % 10 == 0) learning_rate /= 3;

            vector<vector<float>> miniBatch_data = vector_split(data, j * num_batch_data, (j+1) * num_batch_data);
            vector<vector<bool>> miniBatch_label;
            for(int i = 0; i < 10; i++)
                miniBatch_label.push_back(vector_split(labels[i], j * num_batch_data, (j+1) * num_batch_data));

            myNet.gradientDescent(learning_rate, miniBatch_data, miniBatch_label);
            cout << i << " epoch " << j << " batch end, loss: " << myNet.getCost(miniBatch_data, miniBatch_label) << endl;
        }
    }
}