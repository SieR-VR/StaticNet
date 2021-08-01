#include "Layer.h"
#include "../Tools/Defines.h"

std::pair<Vector1D<float>, Vector1D<float>> Layer::Classify(Vector1D<float> input)
{
    std::pair<Vector1D<float>, Vector1D<float>> hypotheses;
    hypotheses.first.resize({outputNumber});
    hypotheses.second.resize({outputNumber});

    for (size_t i = 0; i < outputNumber; i++)
    {
        std::pair<float, float> hypothesys = nodes[i].Hypothesys(input);
        hypotheses.first[i] = hypothesys.first;
        hypotheses.second[i] = hypothesys.second;
    }

    return hypotheses;
}

Vector1D<float> Layer::GradientDescent(float alpha, Vector1D<float> deltaCurrentLayer, std::pair<Vector1D<float>, Vector1D<float>> resultPreviousLayer)
{
    Vector2D<float> weightDiff;
    Vector1D<float> biasDiff;
    weightDiff.resize({inputNumber, outputNumber}, 0.0f);
    biasDiff.resize({outputNumber}, 0.0f);

    for (size_t i = 0; i < outputNumber; i++)
    {
        for (size_t j = 0; j < inputNumber; j++)
            weightDiff.at({j, i}) = deltaCurrentLayer.at(i) * resultPreviousLayer.first.at(j);
        biasDiff.at(i) = deltaCurrentLayer.at(i);
    }

    for (size_t i = 0; i < outputNumber; i++)
        nodes[i].GradientDescent(alpha, weightDiff[i], biasDiff[i]);

    Vector1D<float> deltaPreviousLayer;
    deltaPreviousLayer.resize({inputNumber}, 0.0f);

    for (size_t i = 0; i < inputNumber; i++)
        deltaPreviousLayer.at(i) = (GetWeights(i) * deltaCurrentLayer).mean() * activationFunctionGradient(resultPreviousLayer.second.at(i));

    return deltaPreviousLayer;
}

Vector1D<float> Layer::GetWeights(size_t nodeIndex)
{
    Vector1D<float> weights;
    weights.resize({outputNumber});

    for (size_t i = 0; i < outputNumber; i++)
        weights.at(i) = nodes[i].weights[nodeIndex];

    return weights;
}

Vector1D<uint8_t> Layer::getModelData()
{
    Vector1D<uint8_t> modelData;

    modelData.push(static_cast<uint8_t>(Glow::Defines::ModelNumberType::FLOAT));
    modelData.push(static_cast<uint8_t>(Glow::Defines::ModelType::LOGISTIC_CLASSIFICATON));

    uint32_t modelSize = outputNumber;
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

void Layer::loadModelData(Vector1D<uint8_t> modelData)
{
    if (modelData[0] != static_cast<uint8_t>(Glow::Defines::ModelNumberType::FLOAT))
        throw std::runtime_error("Model number type is invalid!");

    if (modelData[1] != static_cast<uint8_t>(Glow::Defines::ModelType::LOGISTIC_CLASSIFICATON))
        throw std::runtime_error("Model type is invalid!");

    uint32_t modelSize = (modelData[5] << 24 | modelData[4] << 16 | modelData[3] << 8 | modelData[2]);
    uint32_t modelDataSize = (modelData.shape().x - 6) / modelSize;

    if ((modelData.shape().x - 6) % modelSize != 0)
        throw std::runtime_error("Model is invalid!" + std::to_string(modelSize));

    nodes.clear();

    for (int i = 0; i < modelSize; i++)
    {
        Node node(modelData.slice({6 + i * modelDataSize}, {6 + (i + 1) * modelDataSize}));
        nodes.push_back(node);
    }
}