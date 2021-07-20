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

    LogisticClassification(std::vector<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    int logisticClassify(Vector1D input);
    Vector1D softmax(Vector1D input);
    float getCost(Vector2D inputs, Vector2D results);
    float gradientDescent(float alpha, Vector2D inputs, Vector2D results);
    std::vector<uint8_t> getModelData();
    void loadModelData(std::vector<uint8_t> modelData);
};

#endif