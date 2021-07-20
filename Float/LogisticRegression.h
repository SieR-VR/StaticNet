#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include "../Structure/Vector2D.h"
#include <stdint.h>
#include <math.h>
#include <stdexcept>

class LogisticRegression
{
public:
    Vector1D weights;
    Vector1D preCalculatedHypothesys;
    float bias;

    LogisticRegression(unsigned int argumentsNum)
    {
        weights.resize({ argumentsNum }, 0.0f);
        bias = 0.0f;
    }

    LogisticRegression(std::vector<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    float getCost(Vector2D inputs, Vector1D results);
    float getCostDiff(Vector2D inputs, Vector1D results, int index);
    float getBiasDiff(Vector2D inputs, Vector1D results);
    float hypothesys(Vector1D input);
    float sigmoid(float input);
    float gradientDescent(float alpha, Vector2D inputs, Vector1D results);
    std::vector<uint8_t> getModelData();
    void loadModelData(std::vector<uint8_t> modelData);
};

#endif