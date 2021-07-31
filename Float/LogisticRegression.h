#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include "../Structure/Vector2D.h"
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <stdexcept>

class LogisticRegression
{
public:
    Vector1D<float> weights;
    Vector1D<float> preCalculatedHypothesys;
    float bias;

    LogisticRegression(size_t argumentsNum)
    {
        srand((uint32_t)time(NULL));

        weights.resize({ argumentsNum });
        for(size_t i = 0; i < argumentsNum; i++)
            weights.at(i) = (float)(rand() % 100) / 100 - 0.5; // -0.5 .. 0.5

        bias = (float)(rand() % 100) / 100 - 0.5; // -0.5 .. 0.5
    }

    LogisticRegression(Vector1D<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    float getCost(Vector2D<float> inputs, Vector1D<int> results);
    float getCostDiff(Vector2D<float> inputs, Vector1D<int> results, int index);
    float getBiasDiff(Vector2D<float> inputs, Vector1D<int> results);
    float hypothesys(Vector1D<float> input);
    float sigmoid(float input);
    float train(float alpha, Vector2D<float> inputs, Vector1D<int> results);
    void gradientDescent(float alpha, Vector1D<float> weightDiff, float biasDiff);

    size_t getInputNumber();
    
    Vector1D<uint8_t> getModelData();
    void loadModelData(Vector1D<uint8_t> modelData);
};

#endif