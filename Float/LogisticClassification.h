#ifndef LOGISTICCLASSIFICATION_H
#define LOGISTICCLASSIFICATION_H

#include <vector>
#include <stdint.h>
#include <math.h>

#include "LogisticRegression.h"
#include "../Tools/vector_helper.h"

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

    int logisticClassify(std::vector<float> input);
    std::vector<float> softmax(std::vector<float> input);
    float getCost(std::vector<std::vector<float>> inputs, std::vector<std::vector<bool>> results);
    float gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<std::vector<bool>> results);
    std::vector<uint8_t> getModelData();
    void loadModelData(std::vector<uint8_t> modelData);
};

#endif