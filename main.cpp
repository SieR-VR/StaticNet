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
    float x = 1, y = 0;
    vector<vector<float>> data = { {x, x, x * x, x * x, x * x}, {x, y, x * x, x * y, y * y}, {y, x, y * y, y * x, x * x}, {y, y, y * y, y * y, y * y} };
    vector<bool> labels = { 0, 1, 1, 0 };


    LogisticRegression myNet(5);

    for(int i = 0; i < 100000; i++) {
        myNet.gradientDescent(0.1f, data, labels);
    }

    for(int i = 0; i < data.size(); i++)
    {
        cout << myNet.logisticReg(data[i]) << endl; 
    }

    for(auto k : myNet.W) {
        cout << k << " ";
    }
    cout << myNet.b << endl;
    saveModelData(myNet.getModelData(), "./Models/XOR.net");
}