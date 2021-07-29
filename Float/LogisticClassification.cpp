#include "LogisticClassification.h"
#include "../Tools/model_data_defines.h"

int LogisticClassification::logisticClassify(Vector1D<float> input)
{
    Vector1D<float> hypothesyses;
    hypothesyses.resize({ nodes.size() });

    for (int i = 0; i < nodes.size(); i++)
        hypothesyses.at(i) = nodes[i].hypothesys(input);

    float res_float = -100;
    int index = 0;
    for (int i = 0; i < nodes.size(); i++)
    {
        if (res_float < hypothesyses[i])
        {
            res_float = hypothesyses[i];
            index = i;
        }
    }

    return index;
}

Vector1D<float> LogisticClassification::softmax(Vector1D<float> input)
{
    Vector1D<float> inputExponent = input.map([](float i){ return exp(i); });
    return inputExponent / inputExponent.mean();
}

float LogisticClassification::getCost(Vector2D<float> inputs, Vector2D<bool> results)
{
    float res = 0;
    for (int i = 0; i < nodes.size(); i++)
        res += nodes[i].getCost(inputs, results[i]);

    return res;
}

float LogisticClassification::gradientDescent(float alpha, Vector2D<float> inputs, Vector2D<bool> results)
{
    float costSum = 0;

    for (int i = 0; i < nodes.size(); i++)
        costSum += nodes[i].gradientDescent(alpha, inputs, results[i]);

    return costSum;
}

Vector1D<uint8_t> LogisticClassification::getModelData()
{
    Vector1D<uint8_t> modelData;

    modelData.push(static_cast<uint8_t>(modelNumberType::FLOAT));
    modelData.push(static_cast<uint8_t>(modelType::LOGISTIC_CLASSIFICATON));

    uint32_t modelSize = nodes.size();
    modelData.push(((uint8_t *)&modelSize)[0]);
    modelData.push(((uint8_t *)&modelSize)[1]);
    modelData.push(((uint8_t *)&modelSize)[2]);
    modelData.push(((uint8_t *)&modelSize)[3]);

    for (auto &node : nodes)
    {
        auto nodeData = node.getModelData();
        modelData.push(nodeData);
    }

    return modelData;
}

void LogisticClassification::loadModelData(Vector1D<uint8_t> modelData)
{
    if(modelData[0] != static_cast<uint8_t>(modelNumberType::FLOAT))
        throw std::runtime_error("Model number type is invalid!");

    if(modelData[1] != static_cast<uint8_t>(modelType::LOGISTIC_CLASSIFICATON))
        throw std::runtime_error("Model type is invalid!");

    uint32_t modelSize = (modelData[5] << 24 | modelData[4] << 16 | modelData[3] << 8 | modelData[2]);
    uint32_t modelDataSize = (modelData.shape().x - 6) / modelSize;

    if((modelData.shape().x - 6) % modelSize != 0)
        throw std::runtime_error("Model is invalid!" + std::to_string(modelSize));
        
    nodes.clear();

    for(int i = 0; i < modelSize; i++) {
        LogisticRegression node(modelData.slice({ 6 + i * modelDataSize}, { 6 + (i + 1) * modelDataSize}));
        nodes.push_back(node);
    }
}