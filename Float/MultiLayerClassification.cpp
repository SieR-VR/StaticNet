#include "MultiLayerClassification.h"
#include <iostream>

Vector1D<float> MultiLayerClassification::Classify(Vector1D<float> input)
{
    Vector1D<float> outputLayerResult = layers[0].logisticClassify(input);

    for (size_t i = 1; i < layers.size(); i++)
        outputLayerResult = layers[i].logisticClassify(outputLayerResult);

    return outputLayerResult;
}

Vector1D<float> MultiLayerClassification::softmax(Vector1D<float> input)
{
    Vector1D<float> inputExponent = input.map([](float i)
                                              { return exp(i); });
    return inputExponent / inputExponent.mean();
}

float MultiLayerClassification::getCost(Vector2D<float> inputs, Vector2D<int> results)
{
    float costFinal = 0;

    for (size_t i = 0; i < inputs.shape().y; i++)
    {
        Vector1D<float> output = Classify(inputs[i]);
        Vector1D<int> result = results.pop(Vector2DAxis_t::X, 0);

        for (size_t j = 0; j < output.shape().x; j++)
            costFinal += costFunction(output[j], (float)result[j]);
    }

    return costFinal / inputs.shape().y;
}

float MultiLayerClassification::train(float alpha, Vector2D<float> inputs, Vector2D<int> results)
{
    size_t inputDataLength = inputs.shape().y;
    std::vector<Vector2D<float>> forwardPropagation; // vector length = count of layers, Vector X = neuron count, Vector Y = input data size

    forwardPropagation.push_back(inputs);
    for (size_t i = 0; i < layers.size(); i++)
    {
        Vector2D<float> layerOutputs;
        layerOutputs.resize({layers[i].getNodeNumber(), inputDataLength});

        for (size_t j = 0; j < inputDataLength; j++)
            layerOutputs[j] = layers[i].logisticClassify(forwardPropagation[i][j]);

        forwardPropagation.push_back(layerOutputs);
    }

    Vector2D<float> delta, deltaTemp; // Vector X = neuron count, Vector Y = input data size

    delta.resize({layers[layers.size() - 1].getNodeNumber(), inputDataLength}, 0.0f);
    for (size_t j = 0; j < inputDataLength; j++)
    {
        for (size_t k = 0; k < layers[layers.size() - 1].getNodeNumber(); k++)
        {
            float forwardPropagationResult = forwardPropagation[layers.size()][j][k];
            delta.at({k, j}) = (forwardPropagationResult - results[k][j]) * forwardPropagationResult * (1 - forwardPropagationResult);
        }
    }

    for (size_t i = layers.size() - 1; i > 0; i--)
    {
        Vector2D<float> weightDifference;
        Vector1D<float> biasDifference;

        weightDifference.resize({layers[i].getInputNumber(), layers[i].getNodeNumber()}, 0.0f);
        biasDifference.resize({layers[i].getNodeNumber()}, 0.0f);

        for (size_t j = 0; j < inputDataLength; j++)
        {
            for (size_t k = 0; k < layers[i].getNodeNumber(); k++)
            {
                for (size_t l = 0; l < layers[i].getInputNumber(); l++)
                    weightDifference.at({l, k}) += forwardPropagation[i][j][l] * delta.at({k, j});
                biasDifference.at(k) += delta.at({k, j});
            }
        }

        weightDifference /= (float)inputDataLength;
        biasDifference /= (float)inputDataLength;
        layers[i].gradientDescent(alpha, weightDifference, biasDifference);

        deltaTemp = delta;
        delta.resize({layers[i].getInputNumber(), inputDataLength}, 0.0f);

        for (size_t j = 0; j < inputDataLength; j++)
            for (size_t k = 0; k < layers[i].getInputNumber(); k++)
                delta.at({k, j}) = (deltaTemp[j] * layers[i].getWeightsFromNode(k)).mean() * forwardPropagation[i][j][k] * (1 - forwardPropagation[i][j][k]);
    }

    Vector2D<float> weightDifference;
    Vector1D<float> biasDifference;

    weightDifference.resize({layers[0].getInputNumber(), layers[0].getNodeNumber()}, 0.0f);
    biasDifference.resize({layers[0].getNodeNumber()}, 0.0f);

    for (size_t j = 0; j < inputDataLength; j++)
    {
        for (size_t k = 0; k < layers[0].getNodeNumber(); k++)
        {
            for (size_t l = 0; l < layers[0].getInputNumber(); l++)
                weightDifference.at({l, k}) += forwardPropagation[0][j][l] * delta.at({k, j});
            biasDifference.at(k) += delta.at({k, j});
        }
    }

    weightDifference /= (float)inputDataLength;
    biasDifference /= (float)inputDataLength;
    layers[0].gradientDescent(alpha, weightDifference, biasDifference);

    return getCost(inputs, results);
}