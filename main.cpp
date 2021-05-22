#include <iostream>
#include <vector>

#include "MyNet.h"

using namespace std;

int main() {
    MyNet mynet;
    cout << mynet.W.mean() << " " << mynet.b.mean() << endl;

    vector<fixed8bit> xinput;
    vector<fixed8bit> yinput;

    for(int i = 0; i < 10; i++) {
        xinput.push_back(fixed8bit(i+0b01000000)); 
        yinput.push_back(fixed8bit(i+0b01000000)); 
    }

    cout << mynet.getCost(xinput, yinput).mean() << endl;
    cout << mynet.getCostDiff(xinput, yinput).mean() << endl;

    for(int i = 0; i < 1000; i++) {
        mynet.gradientDescent(fixed8bit(0b01100000), xinput, yinput);
    }

    cout << mynet.W.mean() << " " << mynet.b.mean() << endl;
}