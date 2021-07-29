#ifndef LOGISTICCLASSIFICATION_H
#define LOGISTICCLASSIFICATION_H

#include "../Structure/Vector2D.h"
#include <stdint.h>
#include <math.h>

#include "LogisticRegression.h"

class LogisticClassification
{
public:
    std::vector<LogisticRegression> nodes;

    LogisticClassification(unsigned int argumentsNum, unsigned int nodeNum)
    {
        LogisticRegression temp(argumentsNum);
        nodes.assign(nodeNum, temp);
    }

    LogisticClassification(Vector1D<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    int logisticClassify(Vector1D<float> input);
    Vector1D<float> softmax(Vector1D<float> input);
    float getCost(Vector2D<float> inputs, Vector2D<bool> results);
    float gradientDescent(float alpha, Vector2D<float> inputs, Vector2D<bool> results);
    
    Vector1D<uint8_t> getModelData();
    void loadModelData(Vector1D<uint8_t> modelData);
};

#endif