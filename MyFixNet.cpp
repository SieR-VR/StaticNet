#include "MyFixNet.h"

fixed8bit fixed8bit::operator+(fixed8bit operand)
{
    if(((short) operand.value + value) > CHAR_MAX) return CHAR_MAX;
    else if(((short) operand.value + value) < CHAR_MIN) return CHAR_MIN;
    else return operand.value + value;
}

fixed8bit fixed8bit::operator+=(fixed8bit operand)
{
    if(((short) operand.value + value) > CHAR_MAX) value = CHAR_MAX;
    else if(((short) operand.value + value) < CHAR_MIN) value = CHAR_MIN;
    else value = operand.value + value;
}

fixed8bit fixed8bit::operator-(fixed8bit operand)
{
    if(((short) value - operand.value) > CHAR_MAX) return CHAR_MAX;
    else if(((short) value - operand.value) < CHAR_MIN) return CHAR_MIN;
    else return value - operand.value;
}

fixed8bit fixed8bit::operator-=(fixed8bit operand)
{
    if(((short) value - operand.value) > CHAR_MAX) value = CHAR_MAX;
    else if(((short) value - operand.value) < CHAR_MIN) value = CHAR_MIN;
    else value = value - operand.value;
}

fixed8bit fixed8bit::operator*(fixed8bit operand)
{
    short operResRaw = (short)value * operand.value;
    short sign = operResRaw & 0x8000;

    return (operResRaw >> 7) | (sign >> 8);
}

fixed8bit fixed8bit::operator/(char operand)
{
    return value / operand;
}

float fixed8bit::mean()
{
    return value * 0.0078125f;
}

fixed8bit MyFixNet::getCost(std::vector<fixed8bit> inputs, std::vector<fixed8bit> results)
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

fixed8bit MyFixNet::getCostDiff(std::vector<fixed8bit> inputs, std::vector<fixed8bit> results)
{
    if (inputs.size() != results.size())
        return 0;
    fixed8bit resultCostDiff = 0;

    for (int i = 0; i < inputs.size(); i++)
    {
        fixed8bit costDiff = (linearReg(inputs[i]) - results[i]) * inputs[i];
        resultCostDiff += costDiff;
    }

    return resultCostDiff / inputs.size();
}

fixed8bit MyFixNet::linearReg(fixed8bit input)
{
    return W * input + b;
}

fixed8bit MyFixNet::gradientDescent(fixed8bit alpha, std::vector<fixed8bit> inputs, std::vector<fixed8bit> results)
{
    W -= getCostDiff(inputs, results) * alpha;
}