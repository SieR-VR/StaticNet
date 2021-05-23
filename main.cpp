#include <iostream>
#include <vector>

#include "MyFixNet.h"

using namespace std;

int main() {
    MyFixNet mynet;
    cout << mynet.W.mean() << " " << mynet.b.mean() << endl;

    vector<fixed8bit> xinput;
    vector<fixed8bit> yinput;

    for(int i = 0; i < 10; i++) {
        xinput.push_back(fixed8bit(i + 0x70)); 
        yinput.push_back(fixed8bit(i + 0x70)); 
    }

    for(int i = 0; i < 100; i++) {
        mynet.gradientDescent(fixed8bit(0b01000000), xinput, yinput);
    }

    cout << mynet.W.mean() << " " << mynet.b.mean() << endl;
}