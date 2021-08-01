#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include "../Structure/Vector2D.h"
#include <stdint.h>
#include <time.h>
#include <stdexcept>
#include <utility>

class Node
{
public:
    Vector1D<float> weights;
    Vector1D<float> preCalculatedHypothesys;
    float bias;

    Node(size_t argumentsNum, std::function<float(float)> activationFunction, std::function<float(float)> activationFunctionGradient)
    {
        srand((uint32_t)time(NULL));

        weights.resize({ argumentsNum });
        for(size_t i = 0; i < argumentsNum; i++)
            weights.at(i) = (float)(rand() % 100) / 100 - 0.5; // -0.5 .. 0.5

        bias = (float)(rand() % 100) / 100 - 0.5; // -0.5 .. 0.5

        this->activationFunction = activationFunction;
        this->activationFunctionGradient = activationFunctionGradient;
    }

    Node(Vector1D<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    std::pair<float, float> Hypothesys(Vector1D<float> input);
    void GradientDescent(float alpha, Vector1D<float> weightDiff, float biasDiff);

    size_t getInputNumber();
    
    Vector1D<uint8_t> getModelData();
    void loadModelData(Vector1D<uint8_t> modelData);

    std::function<float(float)> activationFunction;
    std::function<float(float)> activationFunctionGradient;
};

#endif