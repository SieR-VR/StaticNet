#include "LogisticRegression.h"
#include "../Tools/vector_helper.h"
#include <math.h>

float LogisticRegression::getCost(std::vector<std::vector<float>> inputs, std::vector<bool> results)
{
    if (inputs.size() != results.size())
        return 0;
    float resultCost = 0;

    for (int i = 0; i < inputs.size(); i++)
    {
        float cost = logisticRegCal[i] - results[i];
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
        resultCostDiff += (logisticRegCal[i] - results[i]) * inputs[i][index];

    return resultCostDiff / inputs.size();
}

float LogisticRegression::getBiasDiff(std::vector<std::vector<float>> inputs, std::vector<bool> results)
{
    if (inputs.size() != results.size())
        return 0;
    float resultBiasDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
        resultBiasDiff += logisticRegCal[i] - results[i];

    return resultBiasDiff / inputs.size();
}

float LogisticRegression::logisticReg(std::vector<float> input)
{
    float result = 0;
    for (int i = 0; i < W.size() / 8; i++)
        result += avx_dot_product(vector_split(input, i * 8, (i + 1) * 8), vector_split(W, i * 8, (i + 1) * 8));
    result += b;
    return sigmoid(result);
}

float LogisticRegression::sigmoid(float input) {
    return (1 / (1 + exp(-1 * input)));
}

float LogisticRegression::gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<bool> results)
{
    std::vector<float> mW = W;
    float mB = b;

    logisticRegCal.clear();
    for(int i = 0; i < inputs.size(); i++)
        logisticRegCal.push_back(logisticReg(inputs[i]));

    
    for (int i = 0; i < W.size(); i++)
        mW[i] -= getCostDiff(inputs, results, i) * alpha;
    mB -= getBiasDiff(inputs, results) * alpha;
    
    W = mW;
    b = mB;

    return getCost(inputs, results);
}