#include <iostream>
#include <vector>
#include <time.h>

#include "FixedLinearRegression.h"
#include "LinearRegression.h"

using namespace std;

int main() {
    clock_t now = clock();

    FixedLinearRegression mynet(2);

    vector<vector<fixed8bit>> xinput;
    vector<fixed8bit> yinput;

    for(int i = 1; i <= 10; i++) {
        std::vector<fixed8bit> temp = {fixed8bit(0x40 + i) * fixed8bit(0x40 + i), fixed8bit(0x40 + i)};
        xinput.push_back(temp); 
        yinput.push_back(fixed8bit(0x14) * fixed8bit(0x40 + i) * fixed8bit(0x40 + i) + fixed8bit(0x20) * fixed8bit(0x40 + i) + fixed8bit(0x04)); 
    }

    for(int i = 0; i < 100000; i++) {
        char learning_rate = 0x40;
        mynet.gradientDescent(fixed8bit(learning_rate), xinput, yinput);
        if(i % 10000 == 0) learning_rate >>= 1;
    }

    cout << mynet.W[0].mean() << " " << mynet.W[1].mean() << " " << mynet.b.mean() << " ";

    cout << clock() - now << endl;
    now = clock();

    LinearRegression mynet_2(2);

    vector<vector<float>> xinput_2;
    vector<float> yinput_2;

    for(int i = 1; i <= 10; i++) {
        std::vector<float> temp = {(float)i*i, (float)i};
        xinput_2.push_back(temp);
        yinput_2.push_back(5.0f * i * i + 2.5f * i + 1);
    }

    for(int i = 0; i < 100000; i++) {
        float learning_rate = 0.0003f;
        mynet_2.gradientDescent(learning_rate, xinput_2, yinput_2);
        if(i % 10000 == 0) {
            learning_rate /= 3;
        }
    }

    cout << mynet_2.W[0] << " " << mynet_2.W[1] << " " << mynet_2.b << " ";

    cout << clock() - now << endl;
}