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

fixed8bit FixedLinearRegression::getCostDiff(std::vector<fixed8bit> inputs, fixed8bit result, unsigned char index)
{
    fixed8bit resultCostDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
    {
        fixed8bit costDiff = (linearReg(inputs) - result) * inputs[index];
        resultCostDiff += costDiff;
    }

    return resultCostDiff / inputs.size();
}

fixed8bit FixedLinearRegression::linearReg(std::vector<fixed8bit> input)
{
    fixed8bit result = 0x00;
    for(unsigned char i = 0; i < W.size(); i++)
        result += W[i] * input[i];
    result += b;
    return result;
}

fixed8bit FixedLinearRegression::sigmoid(std::vector<fixed8bit> input) {
    return SIGMOID[linearReg(input).value + 128];
}

fixed8bit FixedLinearRegression::gradientDescent(fixed8bit alpha, std::vector<std::vector<fixed8bit>> inputs, std::vector<fixed8bit> results)
{
    for(int i = 0; i < W.size(); i++ ) {
        W[i] -= getCostDiff(inputs[i], results[i], i) * alpha;
    }
}