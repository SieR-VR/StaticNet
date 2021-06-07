#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>

class LogisticRegression
{
public:
    std::vector<float> W;
    std::vector<float> logisticRegCal;
    float b;

    LogisticRegression(unsigned int argumentsNum)
    {
        W.assign(argumentsNum, 0.0f);
        b = 0.0f;
    }

    float getCost(std::vector<std::vector<float>> inputs, std::vector<bool> results);
    float getCostDiff(std::vector<std::vector<float>> inputs, std::vector<bool> results, int index);
    float getBiasDiff(std::vector<std::vector<float>> inputs, std::vector<bool> results);
    float logisticReg(std::vector<float> input);
    float sigmoid(float input);
    float gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<bool> results);
};

#endif