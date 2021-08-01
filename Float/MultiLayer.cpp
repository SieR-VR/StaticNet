#include "MultiLayer.h"
#include <iostream>
#include "../Tools/Defines.h"

void MultiLayer::addLayer(
    size_t inputSize,
    size_t outputSize,
    std::function<float(float)> activationFunction,
    std::function<float(float)> activationFunctionGradient)
{
    layers.push_back(Layer(inputSize, outputSize, activationFunction, activationFunctionGradient));
}

void MultiLayer::addLayer(Layer layer)
{
    layers.push_back(layer);
}

Vector1D<float> MultiLayer::Classify(Vector1D<float> input)
{
    std::pair<Vector1D<float>, Vector1D<float>> outputLayerResult = layers[0].Classify(input);

    for (size_t i = 1; i < layers.size(); i++)
        outputLayerResult = layers[i].Classify(outputLayerResult.first);

    return outputLayerResult.first;
}

float MultiLayer::getCost(Vector2D<float> inputs, Vector2D<int> results)
{
    float costFinal = 0;

    for (size_t i = 0; i < inputs.shape().y; i++)
    {
        Vector1D<float> output = Classify(inputs[i]);
        Vector1D<int> result = results.pop(Vector2DAxis_t::X, 0);

        costFinal += costFunction(output, result);
    }

    return costFinal / inputs.shape().y;
}

float MultiLayer::train(float alpha, Vector2D<float> inputs, Vector2D<int> results)
{
    size_t inputDataLength = inputs.shape().y;
    for (size_t i = 0; i < inputDataLength; i++)
    {
        std::vector<std::pair<Vector1D<float>, Vector1D<float>>> forwardPropagation; // vector length = count of layers, Vector X = neuron count, Vector Y = input data size
        forwardPropagation.push_back(std::make_pair(inputs[i], inputs[i]));
        for (size_t j = 0; j < layers.size(); j++)
            forwardPropagation.push_back(layers[j].Classify(forwardPropagation[j].first));

        Vector1D<float> delta = costFunctionGradient(forwardPropagation.back().first, results.at(i, Vector2DAxis_t::X))
            * forwardPropagation.back().second.map(layers.back().activationFunctionGradient);
        for(int j = layers.size() - 1; j >= 0; j--)
            delta = layers[j].GradientDescent(alpha, delta, forwardPropagation[j]);
    }

    return getCost(inputs, results);
}