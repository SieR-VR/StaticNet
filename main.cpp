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
    auto data_ = read_mnist_images("./train-images.idx3-ubyte", image_num, image_size);
    auto labels_ = read_mnist_labels("./train-labels.idx1-ubyte", label_num);

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

    int test_num = 1000;
    int train_num = image_num - test_num;

    int num_batch = 200;
    int num_batch_data = train_num / num_batch;
    int num_epoch = 1;
    float learning_rate = 0.3f;

    vector<vector<float>> train_data = vector_split(data, 0, train_num);
    vector<vector<bool>> train_label;
    for(int i = 0; i < 10; i++)
        train_label.push_back(vector_split(labels[i], 0, train_num));

    vector<vector<float>> test_data = vector_split(data, train_num, train_num + test_num);
    vector<vector<bool>> test_label;
    for(int i = 0; i < 10; i++)
        test_label.push_back(vector_split(labels[i], train_num, train_num + test_num));

    for(int i = 0; i < num_epoch; i++) {
        for(int j = 0; j < num_batch; j++) {
            if(j % 100 == 0) learning_rate /= 3;

            vector<vector<float>> miniBatch_data = vector_split(train_data, j * num_batch_data, (j+1) * num_batch_data);
            vector<vector<bool>> miniBatch_label;
            for(int i = 0; i < 10; i++)
                miniBatch_label.push_back(vector_split(train_label[i], j * num_batch_data, (j+1) * num_batch_data));
            
            float loss = myNet.gradientDescent(learning_rate, miniBatch_data, miniBatch_label);
            cout << i << " epoch " << j << " batch end, loss: " << loss << endl;
        }
    }

    cout << "Train End" << endl;

    int success_num = 0;
    for(int i = 0; i < test_num; i++) {
        if(test_label.at(myNet.logisticClassify(test_data[i]))[i] == 1) success_num++;
        if(i % 100 == 99) cout << success_num << endl; 
    }

    cout << "Success Rate: " << (float) success_num / test_num * 100 << "%" << endl;
}