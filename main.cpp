#include <iostream>
#include <vector>

#include "FixedLinearRegression.h"

using namespace std;

int main() {
    FixedLinearRegression mynet(1);
    cout << mynet.W[0].mean() << " " << mynet.b.mean() << endl;

    vector<vector<fixed8bit>> xinput;
    vector<fixed8bit> yinput;

    for(int i = 0; i < 10; i++) {
        std::vector<fixed8bit> temp = {fixed8bit(i + 0x70)};
        xinput.push_back(temp); 
        yinput.push_back(fixed8bit(i + 0x70)); 
    }

    for(int i = 0; i < 100; i++) {
        mynet.gradientDescent(fixed8bit(0b01000000), xinput, yinput);
    }

    cout << mynet.W[0].mean() << " " << mynet.b.mean() << endl;
}