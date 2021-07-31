#include "LogisticRegression.h"
#include "../Tools/Defines.h"

using namespace Defines;

float LogisticRegression::getCost(Vector2D<float> inputs, Vector1D<int> results)
{
    if (inputs.shape().y != results.shape().x) {
        throw std::runtime_error("LogisticRegression::getCost: inputs and results must have the same number of rows");
        return 0;
    }

    float resultCost = 0;

    for (int i = 0; i < inputs.shape().y; i++)
        resultCost += -1 * (results[i] * log(preCalculatedHypothesys[i]) + (1 - results[i]) * log(1 - preCalculatedHypothesys[i]));

    return resultCost / inputs.shape().y;
}

float LogisticRegression::getCostDiff(Vector2D<float> inputs, Vector1D<int> results, int index)
{
    if (inputs.shape().y != results.shape().x) {
        throw std::runtime_error("LogisticRegression::getCost: inputs and results must have the same number of rows");
        return 0;
    }

    float resultCostDiff = 0;

    for (int i = 0; i < inputs.shape().y; i++)
        resultCostDiff += (preCalculatedHypothesys[i] - results[i]) * inputs[i][index];

    return resultCostDiff / inputs.shape().y;
}

float LogisticRegression::getBiasDiff(Vector2D<float> inputs, Vector1D<int> results)
{
    if (inputs.shape().y != results.shape().x) {
        throw std::runtime_error("LogisticRegression::getCost: inputs and results must have the same number of rows");
        return 0;
    }

    float resultBiasDiff = 0;

    for (int i = 0; i < inputs.shape().y; i++)
        resultBiasDiff += preCalculatedHypothesys[i] - results[i];

    return resultBiasDiff / inputs.shape().y;
}

float LogisticRegression::hypothesys(Vector1D<float> input)
{
    if(input.shape() != weights.shape()) {
        throw std::runtime_error("LogisticRegression::hypothesys: inputs and weights must have the same number of rows");
        return 0;
    }

    float result = 0;
    for (int i = 0; i < weights.shape().x; i++)
        result += input[i] * weights[i];
    result += bias;

    return sigmoid(result);
}

float LogisticRegression::sigmoid(float input) {
    return (1 / (1 + exp(-input)));
}

float LogisticRegression::train(float alpha, Vector2D<float> inputs, Vector1D<int> results)
{
    Vector1D<float> mWeights = weights;
    float mBias = bias;

    preCalculatedHypothesys.clear();
    for (size_t i = 0; i < inputs.shape().y; i++)
        preCalculatedHypothesys.push(hypothesys(inputs[i]));
    
    for (int i = 0; i < weights.shape().x; i++)
        mWeights.at(i) -= getCostDiff(inputs, results, i) * alpha;
    mBias -= getBiasDiff(inputs, results) * alpha;
    
    weights = mWeights;
    bias = mBias;

    return getCost(inputs, results);
}

void LogisticRegression::gradientDescent(float alpha, Vector1D<float> weightDiff, float biasDiff)
{
    weights -= (weightDiff * alpha);
    bias -= (biasDiff * alpha);
}

size_t LogisticRegression::getInputNumber()
{
    return weights.shape().x;
}

Vector1D<uint8_t> LogisticRegression::getModelData()
{
    Vector1D<uint8_t> modelData;
    modelData.push(static_cast<uint8_t>(modelNumberType::FLOAT));
    modelData.push(static_cast<uint8_t>(modelType::LOGISTIC_REGRESSION));

    uint32_t modelSize = weights.shape().x + 1;
    modelData.push(((uint8_t*)&modelSize)[0]);
    modelData.push(((uint8_t*)&modelSize)[1]);
    modelData.push(((uint8_t*)&modelSize)[2]);
    modelData.push(((uint8_t*)&modelSize)[3]);

    for(size_t i = 0; i < weights.shape().x; i++)
    {
        modelData.push(((uint8_t*)&weights)[0 + i]);
        modelData.push(((uint8_t*)&weights)[1 + i]);
        modelData.push(((uint8_t*)&weights)[2 + i]);
        modelData.push(((uint8_t*)&weights)[3 + i]);
    }

    modelData.push(((uint8_t*)&bias)[0]);
    modelData.push(((uint8_t*)&bias)[1]);
    modelData.push(((uint8_t*)&bias)[2]);
    modelData.push(((uint8_t*)&bias)[3]);

    return modelData;
}

void LogisticRegression::loadModelData(Vector1D<uint8_t> modelData)
{
    if(modelData[0] != static_cast<uint8_t>(modelNumberType::FLOAT))
        throw std::runtime_error("Model number type is invalid!");

    if(modelData[1] != static_cast<uint8_t>(modelType::LOGISTIC_REGRESSION))
        throw std::runtime_error("Model type is invalid!");

    weights.clear();
    bias = 0.0f;

    int modelSize = (modelData[5] << 24 | modelData[4] << 16 | modelData[3] << 8 | modelData[2]) - 1;

    for(int i = 0; i < modelSize; i++) 
    {
        int temp = (modelData[9 + 4*i] << 24 | modelData[8 + 4*i] << 16 | modelData[7 + 4*i] << 8 | modelData[6 + 4*i]);
        float weight = ((float *)&temp)[0];

        weights.push(weight);
    }

    int tempB = modelData[9 + 4 * modelSize] << 24 | modelData[8 + 4 * modelSize] << 16 | modelData[7 + 4 * modelSize] << 8 | modelData[6 + 4 * modelSize];
    bias = ((float *)&tempB)[0];
}