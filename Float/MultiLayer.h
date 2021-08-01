#ifndef MULTI_LAYER_CLASSIFICATION
#define MULTI_LAYER_CLASSIFICATION

#include "Layer.h"
#include "../Structure/Vector2D.h"
#include "../Tools/Defines.h"

class MultiLayer
{
public:
    MultiLayer(std::function<float(Vector1D<float>, Vector1D<int>)> costFunction, std::function<Vector1D<float>(Vector1D<float>, Vector1D<int>)> costFunctionGradient)
    {
        this->costFunction = costFunction;
        this->costFunctionGradient = costFunctionGradient;
    }

    MultiLayer(Vector1D<uint8_t> modelData)
    {
        loadModelData(modelData);
    }

    void addLayer(
        size_t inputSize,
        size_t outputSize,
        std::function<float(float)> activationFunction, 
        std::function<float(float)> activationFunctionGradient
    );
    void addLayer(Layer layer);
    Vector1D<float> Classify(Vector1D<float> input);
    float getCost(Vector2D<float> inputs, Vector2D<int> results);
    float train(float alpha, Vector2D<float> inputs, Vector2D<int> results);

    Vector1D<uint8_t> getModelData();
    void loadModelData(Vector1D<uint8_t> modelData);

    std::vector<Layer> layers;
    std::function<float(Vector1D<float>, Vector1D<int>)> costFunction;
    std::function<Vector1D<float>(Vector1D<float>, Vector1D<int>)> costFunctionGradient;
};

#endif