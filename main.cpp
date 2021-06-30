#include <iostream>
#include <vector>
#include <random>
#include <time.h>

#include "Float/LogisticClassification.h"
#include "Tools/load_mnist.h"
#include "Tools/vector_helper.h"
#include "Tools/save_model_data.h"

using namespace std;

int main(int argc, char* argv[])
{
    vector<vector<float>> mnist_images;
    vector<vector<bool>> mnist_labels;

    try {
        mnist_images = get_mnist_image_float(argv[1]);
        mnist_labels = get_mnist_label(argv[2]);
    }
    catch(std::runtime_error err) {
        cout << err.what() << endl;
        return -1;
    };

    LogisticClassification myNet(0, 0);

    try {
        auto model_data = loadModelDataFromFile("./Models/Mnist.net");
        myNet.loadModelData(model_data);
    }
    catch(std::runtime_error err) {
        cout << err.what() << endl;
        return -2;
    }

    int success_num = 0;
    for(int i = 0; i < mnist_images.size(); i++)
    {
        if(mnist_labels.at(myNet.logisticClassify(mnist_images[i]))[i] == 1) success_num++;
    }

    cout << "Success Rate: " << (float) success_num / mnist_images.size() * 100 << "%" << endl;
}