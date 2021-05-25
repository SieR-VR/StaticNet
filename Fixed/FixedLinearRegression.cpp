#include "FixedLinearRegression.h"
#include "sigmoid.h"

fixed8bit FixedLinearRegression::getCost(std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results)
{
    if (inputs.size() != results.size())
        return 0;
    fixed8bit resultCost = 0;

    for (int i = 0; i < inputs.size(); i++)
    {
        fixed8bit cost = linearReg(inputs[i]) - results[i];
        resultCost += cost * cost;
    }

    return resultCost / inputs.size();
}

fixed8bit FixedLinearRegression::getCostDiff(std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results, unsigned char index)
{
    if (inputs.size() != results.size())
        return 0;
    fixed8bit resultCostDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
        resultCostDiff += (linearReg(inputs[i]) - results[i]) * inputs[i][index];

    return resultCostDiff / inputs.size();
}

fixed8bit FixedLinearRegression::getBiasDiff(std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results) {
    if (inputs.size() != results.size())
        return 0;
    fixed8bit resultCostDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
        resultCostDiff += (linearReg(inputs[i]) - results[i]);

    return resultCostDiff / inputs.size();
}

fixed8bit FixedLinearRegression::linearReg(std::vector<fixed8bit> input)
{
    fixed8bit result = 0x00;
    for (unsigned char i = 0; i < W.size(); i++)
        result += W[i] * input[i];
    result += b;
    return result;
}

fixed8bit FixedLinearRegression::sigmoid(std::vector<fixed8bit> input)
{
    return SIGMOID[linearReg(input).value + 128];
}

fixed8bit FixedLinearRegression::gradientDescent(fixed8bit alpha, std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results)
{
    std::vector<fixed8bit> mW = W;
    fixed8bit mB = b;
    
    for (int i = 0; i < W.size(); i++) {
        mW[i] -= getCostDiff(inputs, results, i) * alpha;
        mB -= getBiasDiff(inputs, results) * alpha;
    }
    
    W = mW;
    b = mB;
}