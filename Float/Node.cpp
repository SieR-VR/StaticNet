#include "Node.h"
#include "../Tools/Defines.h"

std::pair<float, float> Node::Hypothesys(Vector1D<float> input)
{
    if(input.shape() != weights.shape()) {
        throw std::runtime_error("Node::hypothesys: inputs and weights must have the same number of rows");
        return std::make_pair(0.0f, 0.0f);
    }

    float result = 0;
    for (int i = 0; i < weights.shape().x; i++)
        result += input[i] * weights[i];
    result += bias;

    return std::make_pair(activationFunction(result), result);
}

void Node::GradientDescent(float alpha, Vector1D<float> weightDiff, float biasDiff)
{
    weights -= (weightDiff * alpha);
    bias -= (biasDiff * alpha);
}

size_t Node::getInputNumber()
{
    return weights.shape().x;
}

Vector1D<uint8_t> Node::getModelData()
{
    Vector1D<uint8_t> modelData;
    modelData.push(static_cast<uint8_t>(Glow::Defines::ModelNumberType::FLOAT));
    modelData.push(static_cast<uint8_t>(Glow::Defines::ModelType::LOGISTIC_REGRESSION));

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

void Node::loadModelData(Vector1D<uint8_t> modelData)
{
    if(modelData[0] != static_cast<uint8_t>(Glow::Defines::ModelNumberType::FLOAT))
        throw std::runtime_error("Model number type is invalid!");

    if(modelData[1] != static_cast<uint8_t>(Glow::Defines::ModelType::LOGISTIC_REGRESSION))
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