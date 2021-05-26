#include "LogisticRegression.h"
#include <math.h>

float LogisticRegression::getCost(std::vector<std::vector<float>> inputs, std::vector<bool> results)
{
    if (inputs.size() != results.size())
        return 0;
    float resultCost = 0;

    for (int i = 0; i < inputs.size(); i++)
    {
        float cost = logisticReg(inputs[i]) - results[i];
        resultCost += cost * cost;
    }

    return resultCost / inputs.size();
}

float LogisticRegression::getCostDiff(std::vector<std::vector<float>> inputs, std::vector<bool> results, int index)
{
    if (inputs.size() != results.size())
        return 0;
    float resultCostDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
        resultCostDiff += (logisticReg(inputs[i]) - results[i]) * inputs[i][index];

    return resultCostDiff / inputs.size();
}

float LogisticRegression::getBiasDiff(std::vector<std::vector<float>> inputs, std::vector<bool> results)
{
    if (inputs.size() != results.size())
        return 0;
    float resultBiasDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
        resultBiasDiff += (logisticReg(inputs[i]) - results[i]);

    return resultBiasDiff / inputs.size();
}

float LogisticRegression::logisticReg(std::vector<float> input)
{
    float result = 0;
    for (int i = 0; i < W.size(); i++)
        result += W[i] * input[i];
    result += b;
    return sigmoid(result);
}

float LogisticRegression::sigmoid(float input) {
    return (1 / (1 + exp(-1 * input)));
}

void LogisticRegression::gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<bool> results)
{
    std::vector<float> mW = W;
    float mB = b;
    
    for (int i = 0; i < W.size(); i++) {
        mW[i] -= getCostDiff(inputs, results, i) * alpha;
        mB -= getBiasDiff(inputs, results) * alpha;
    }
    
    W = mW;
    b = mB;
}