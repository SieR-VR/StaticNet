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

    LogisticClassification(size_t inputNum, size_t outputNum)
    {
        for(size_t i = 0; i < outputNum; i++)
        {
            LogisticRegression temp(inputNum);
            nodes.push_back(temp);
        }
    }

    LogisticClassification(Vector1D<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    Vector1D<float> logisticClassify(Vector1D<float> input);
    Vector1D<float> softmax(Vector1D<float> input);
    float getCost(Vector2D<float> inputs, Vector2D<int> results);
    float train(float alpha, Vector2D<float> inputs, Vector2D<int> results);
    void gradientDescent(float alpha, Vector2D<float> weightDiff, Vector1D<float> biasDiff);

    size_t getNodeNumber();
    size_t getInputNumber();
    Vector1D<float> getWeightsFromNode(size_t nodeIndex);
    
    Vector1D<uint8_t> getModelData();
    void loadModelData(Vector1D<uint8_t> modelData);
};

#endif