#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>

class LinearRegression
{
public:
    std::vector<float> W;
    float b;

    LinearRegression(unsigned int argumentsNum)
    {
        W.assign(argumentsNum, 0.0f);
        b = 0.0f;
    }

    float getCost(std::vector<std::vector<float>> inputs, std::vector<float> results);
    float getCostDiff(std::vector<std::vector<float>> inputs, std::vector<float> results, int index);
    float getBiasDiff(std::vector<std::vector<float>> inputs, std::vector<float> results);
    float linearReg(std::vector<float> input);
    void gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<float> results);
};

#endif