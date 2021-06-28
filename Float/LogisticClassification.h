#ifndef LOGISTICCLASSIFICATION_H
#define LOGISTICCLASSIFICATION_H

#include "LogisticRegression.h"
#include <vector>
#include <string>

class LogisticClassification
{
public:
    std::vector<LogisticRegression> nodes;

    LogisticClassification(unsigned int argumentsNum, unsigned int nodeNum)
    {
        LogisticRegression temp(argumentsNum);
        nodes.assign(nodeNum, temp);
    }

    int logisticClassify(std::vector<float> input);
    std::vector<float> softmax(std::vector<float> input);
    float getCost(std::vector<std::vector<float>> inputs, std::vector<std::vector<bool>> results);
    float gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<std::vector<bool>> results);
    std::string getModelData();
};

#endif