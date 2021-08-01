#ifndef LAYER_H
#define LAYER_H

#include "../Structure/Vector2D.h"
#include <stdint.h>

#include "Node.h"

class Layer
{
public:
    std::vector<Node> nodes;

    Layer(size_t inputNum, size_t outputNum, std::function<float(float)> activationFunction, std::function<float(float)> activationFunctionGradient)
    {
        for(size_t i = 0; i < outputNum; i++)
        {
            Node temp(inputNum, activationFunction, activationFunctionGradient);
            nodes.push_back(temp);
        }

        this->inputNumber = inputNum;
        this->outputNumber = outputNum;

        this->activationFunction = activationFunction;
        this->activationFunctionGradient = activationFunctionGradient;
    }

    Layer(Vector1D<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    std::pair<Vector1D<float>, Vector1D<float>> Classify(Vector1D<float> input);
    Vector1D<float> GradientDescent(float alpha, Vector1D<float> deltaCurrentLayer, std::pair<Vector1D<float>, Vector1D<float>> resultPreviousLayer);
    Vector1D<float> GetWeights(size_t nodeIndex);
    
    Vector1D<uint8_t> getModelData();
    void loadModelData(Vector1D<uint8_t> modelData);

    std::function<float(float)> activationFunction;
    std::function<float(float)> activationFunctionGradient;

    size_t inputNumber, outputNumber;
};

#endif