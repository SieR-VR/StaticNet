#ifndef MODEL_DATA_DEFINES
#define MODEL_DATA_DEFINES

#include <stdint.h>
#include <functional>
#include <cmath>
#include "../Structure/Vector1D.h"

namespace SingleNet
{
    namespace Defines
    {

        enum class ModelNumberType : uint8_t
        {
            FIXED = 0,
            FLOAT = 1
        };

        enum class ModelType : uint8_t
        {
            LINEAR_REGRESSION = 0,
            LOGISTIC_REGRESSION = 1,
            LOGISTIC_CLASSIFICATON = 2
        };

        const std::function<float(float)> Sigmoid = [](float x)
        { return 1.0f / (1.0f + std::exp(-x)); };
        const std::function<float(float)> SigmoidGradient = [](float x)
        { return Sigmoid(x) * (Sigmoid(x) * -1.0f + 1.0f); };
        const std::function<Vector1D<float>(Vector1D<float>)> Softmax = [](Vector1D<float> x)
        { return x.map([](float f) { return std::exp(f); }) / x.map([](float f) { return std::exp(f); }).mean(); };
        const std::function<Vector1D<float>(Vector1D<float>)> LogSoftmax = [](Vector1D<float> x)
        { return x - log(x.map([](float f) { return std::exp(f); }).mean()); };
        const std::function<Vector1D<float>(Vector1D<float>)> SoftmaxGradient = [](Vector1D<float> x)
        { return Softmax(x) * (Softmax(x) * -1.0f + 1.0f); };
        const std::function<float(float)> ReLU = [](float x) 
        { return x > 0.0f ? x : 0.0f; };
        const std::function<float(float)> ReLUGradient = [](float x)
        { return x > 0.0f ? 1.0f : 0.0f; };
        const std::function<Vector1D<int>(Vector1D<float>)> OneHot = [](Vector1D<float> x)
        { 
            Vector1D<int> result;
            result.resize({ x.shape().x }, 0);
            size_t index = x.max().second;

            for (size_t i = 0; i < result.shape().x; i++)
                if (i == index) result[i] = 1;

            return result;
        };

        const std::function<float(Vector1D<float>, Vector1D<int>)> MeanSquaredLoss = [](Vector1D<float> x, Vector1D<int> y)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < x.shape().x; i++)
                sum += (x[i] - y[i]) * (x[i] - y[i]);

            return sum / x.shape().x;
        };
        const std::function<Vector1D<float>(Vector1D<float>, Vector1D<int>)> MeanSquaredLossGradient = [](Vector1D<float> x, Vector1D<int> y)
        { 
            Vector1D<float> gradient;
            gradient.resize({ x.shape().x });
            
            for (size_t i = 0; i < x.shape().x; i++)
                gradient[i] = 2.0f * (x[i] - y[i]);

            return gradient;
        };
        const std::function<float(Vector1D<float>, Vector1D<int>)> CategoricalCrossEntropyLoss = [](Vector1D<float> x, Vector1D<int> y)
        { 
            float sum = 0.0f;

            for (size_t i = 0; i < x.shape().x; i++)
                sum += -y[i] * log(x[i]);

            return sum;
        };
        const std::function<Vector1D<float>(Vector1D<float>, Vector1D<int>)> CategoricalCrossEntropyLossGradient = [](Vector1D<float> x, Vector1D<int> y)
        {
            Vector1D<float> result;
            result.resize(x.shape());

            for (size_t i = 0; i < x.shape().x; i++)
                result[i] = x[i] - y[i];

            return result;
        };
    }
}

#endif
