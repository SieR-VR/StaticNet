#include <iostream>
#include <vector>
#include <random>
#include <time.h>

#include "Float/LogisticClassification.h"

using namespace std;

int main()
{
    LogisticClassification myNet(784, 10);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dis(0, 1);

    vector<vector<float>> dataset(300);
    vector<vector<bool>> label(1);
    for(int i = 0; i < 300; i++) {
        vector<float> temp(2);
        temp[0] = dis(gen); temp[1] = dis(gen);

        dataset[i] = temp;
        label[0].push_back((temp[0] + temp[1]) > 0);
    }

    for(int i = 0; i < 10; i++) {
        myNet.gradientDescent(0.0001, dataset, label);
    }

    cout << "end!" << endl;
}