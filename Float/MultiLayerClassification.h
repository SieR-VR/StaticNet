#ifndef MULTI_LAYER_CLASSIFICATION
#define MULTI_LAYER_CLASSIFICATION

#include "LogisticClassification.h"
#include "../Structure/Vector2D.h"

class MultiLayerClassification
{
public:
    MultiLayerClassification(Vector1D<size_t> layerSizes, std::function<float(float, float)> costFunction) // first element of layerSizes is the input layer size, last one is the output layer size
    {
        for (size_t i = 0; i < layerSizes.shape().x - 1; i++)
        {
            layers.push_back(LogisticClassification(layerSizes[i], layerSizes[i + 1]));
        }

        this->costFunction = costFunction;
    }

    MultiLayerClassification(Vector1D<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    Vector1D<float> Classify(Vector1D<float> input);
    Vector1D<float> softmax(Vector1D<float> input);
    float getCost(Vector2D<float> inputs, Vector2D<int> results);
    float train(float alpha, Vector2D<float> inputs, Vector2D<int> results);

    Vector1D<uint8_t> getModelData();
    void loadModelData(Vector1D<uint8_t> modelData);

    std::vector<LogisticClassification> layers;
    std::function<float(float, float)> costFunction;
};

#endif