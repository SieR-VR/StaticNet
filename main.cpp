#include <iostream>
#include <vector>
#include <random>
#include <time.h>

#include "Structure/Vector2D.h"

using namespace std;

int main(int argc, char* argv[])
{
    float x = 1, y = 0;
    vector<vector<float>> data = { {x, x, x * x, x * x, x * x}, {x, y, x * x, x * y, y * y}, {y, x, y * y, y * x, x * x}, {y, y, y * y, y * y, y * y} };
    vector<bool> labels = { 0, 1, 1, 0 };

    Vector2D<float> vecData(data);
    Vector1D<bool> vecLabels(labels);

    cout << "Data: " << vecData.at({0, 0}) << endl;

}