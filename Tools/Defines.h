#ifndef MODEL_DATA_DEFINES
#define MODEL_DATA_DEFINES

#include <stdint.h>
#include <functional>
#include <cmath>
#include "../Structure/Vector3D.h"

namespace Defines
{

    enum class modelNumberType : uint8_t
    {
        FIXED = 0,
        FLOAT = 1
    };

    enum class modelType : uint8_t
    {
        LINEAR_REGRESSION = 0,
        LOGISTIC_REGRESSION = 1,
        LOGISTIC_CLASSIFICATON = 2
    };

    const std::function<float(float)> Sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    const std::function<float(float)> SigmoidGrad = [](float x) { return Sigmoid(x) * (1.0f - Sigmoid(x)); };
    const std::function<Vector1D<float>(Vector1D<float>)> Softmax = [](Vector1D<float> x) { return x.map([](float f){ return std::exp(f); }) / x.map([](float f){ return std::exp(f); }).mean(); };

    const std::function<float(float, float)> MeanSquaredError = [](float x, float y) { return (x - y) * (x - y); };
    const std::function<float(float, float)> MeanSquaredErrorGrad = [](float x, float y) { return 2.0f * (x - y); };
    const std::function<float(float, float)> CrossEntropyError = [](float x, float y) { return -y * std::log(x) - (1.0f - y) * std::log(1.0f - x); };

}

#endif