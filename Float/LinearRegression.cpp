#include "LinearRegression.h"

float LinearRegression::getCost(std::vector<std::vector<float>> inputs, std::vector<float> results)
{
    if (inputs.size() != results.size())
        return 0;
    float resultCost = 0;

    for (int i = 0; i < inputs.size(); i++)
    {
        float cost = linearReg(inputs[i]) - results[i];
        resultCost += cost * cost;
    }

    return resultCost / inputs.size();
}

float LinearRegression::getCostDiff(std::vector<std::vector<float>> inputs, std::vector<float> results, int index)
{
    if (inputs.size() != results.size())
        return 0;
    float resultCostDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
        resultCostDiff += (linearReg(inputs[i]) - results[i]) * inputs[i][index];

    return resultCostDiff / inputs.size();
}

float LinearRegression::getBiasDiff(std::vector<std::vector<float>> inputs, std::vector<float> results)
{
    if (inputs.size() != results.size())
        return 0;
    float resultBiasDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
        resultBiasDiff += (linearReg(inputs[i]) - results[i]);

    return resultBiasDiff / inputs.size();
}

float LinearRegression::linearReg(std::vector<float> input)
{
    float result = 0;
    for (int i = 0; i < W.size(); i++)
        result += W[i] * input[i];
    result += b;
    return result;
}

void LinearRegression::gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<float> results)
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