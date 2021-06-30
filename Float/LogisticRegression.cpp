#include "LogisticRegression.h"
#include "../Tools/vector_helper.h"
#include "../Tools/model_data_defines.h"

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
    for (int i = 0; i < W.size(); i++)
        result += input[i] * W[i];
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

std::vector<uint8_t> LogisticRegression::getModelData() 
{
    std::vector<uint8_t> modelData;
    modelData.push_back(static_cast<uint8_t>(modelNumberType::FLOAT));
    modelData.push_back(static_cast<uint8_t>(modelType::LOGISTIC_REGRESSION));

    uint32_t modelSize = W.size() + 1;
    modelData.push_back(((uint8_t*)&modelSize)[0]);
    modelData.push_back(((uint8_t*)&modelSize)[1]);
    modelData.push_back(((uint8_t*)&modelSize)[2]);
    modelData.push_back(((uint8_t*)&modelSize)[3]);

    for(const auto &weight : W)
    {
        modelData.push_back(((uint8_t*)&weight)[0]);
        modelData.push_back(((uint8_t*)&weight)[1]);
        modelData.push_back(((uint8_t*)&weight)[2]);
        modelData.push_back(((uint8_t*)&weight)[3]);
    }

    modelData.push_back(((uint8_t*)&b)[0]);
    modelData.push_back(((uint8_t*)&b)[1]);
    modelData.push_back(((uint8_t*)&b)[2]);
    modelData.push_back(((uint8_t*)&b)[3]);

    return modelData;
}

void LogisticRegression::loadModelData(std::vector<uint8_t> modelData)
{
    if(modelData[0] != static_cast<uint8_t>(modelNumberType::FLOAT))
        throw std::runtime_error("Model number type is invalid!");

    if(modelData[1] != static_cast<uint8_t>(modelType::LOGISTIC_REGRESSION))
        throw std::runtime_error("Model type is invalid!");

    W.clear();
    b = 0.0f;

    int modelSize = (modelData[5] << 24 | modelData[4] << 16 | modelData[3] << 8 | modelData[2]) - 1;

    for(int i = 0; i < modelSize; i++) 
    {
        int temp = (modelData[9 + 4*i] << 24 | modelData[8 + 4*i] | modelData[7 + 4*i] | modelData[6 + 4*i]);
        float weight = ((float *)&temp)[0];

        W.push_back(weight);
    }

    int tempB = modelData[9 + 4 * modelSize] << 24 | modelData[8 + 4 * modelSize] << 16 | modelData[7 + 4 * modelSize] << 8 | modelData[6 + 4 * modelSize];
    b = ((float *)&tempB)[0];
}