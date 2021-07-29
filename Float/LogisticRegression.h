#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include "../Structure/Vector2D.h"
#include <stdint.h>
#include <math.h>
#include <stdexcept>

class LogisticRegression
{
public:
    Vector1D<float> weights;
    Vector1D<float> preCalculatedHypothesys;
    float bias;

    LogisticRegression(unsigned int argumentsNum)
    {
        weights.resize({ argumentsNum }, 0.0f);
        bias = 0.0f;
    }

    LogisticRegression(Vector1D<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    float getCost(Vector2D<float> inputs, Vector1D<bool> results);
    float getCostDiff(Vector2D<float> inputs, Vector1D<bool> results, int index);
    float getBiasDiff(Vector2D<float> inputs, Vector1D<bool> results);
    float hypothesys(Vector1D<float> input);
    float sigmoid(float input);
    float gradientDescent(float alpha, Vector2D<float> inputs, Vector1D<bool> results);
    Vector1D<uint8_t> getModelData();
    void loadModelData(Vector1D<uint8_t> modelData);
};

#endif