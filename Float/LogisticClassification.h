#ifndef LOGISTICCLASSIFICATION_H
#define LOGISTICCLASSIFICATION_H

#include "LogisticRegression.h"
#include <vector>

class LogisticClassification
{
public:
    std::vector<LogisticRegression> nodes;

    LogisticClassification(unsigned int argumentsNum, unsigned int nodeNum)
    {
        LogisticRegression temp(argumentsNum);
        nodes.assign(nodeNum, temp);
    }

    std::vector<float> logisticClassify(std::vector<float> input);
    std::vector<float> softmax(std::vector<float> input);
    void gradientDescent(float alpha, std::vector<std::vector<float>> inputs, std::vector<std::vector<bool>> results);
};

#endif