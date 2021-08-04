#ifndef MODEL_DATA_DEFINES
#define MODEL_DATA_DEFINES

#include <cmath>
#include "../Structure/Vector.h"

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
        const std::function<Vector<float, 1>(Vector<float, 1>)> Softmax = [](Vector<float, 1> x)
        { return x.map([](float f) { return std::exp(f); }) / Utils::sum(x.map([](float f) { return std::exp(f); })); };
        const std::function<Vector<float, 1>(Vector<float, 1>)> SoftmaxGradient = [](Vector<float, 1> x)
        { return Softmax(x).map([](float f) { return f * (1.0f - f); }); };
        const std::function<float(float)> ReLU = [](float x) 
        { return x > 0.0f ? x : 0.0f; };
        const std::function<float(float)> ReLUGradient = [](float x)
        { return x > 0.0f ? 1.0f : 0.0f; };
        const std::function<Vector<bool, 1>(Vector<float, 1>)> OneHot = [](Vector<float, 1> x)
        { 
            Vector<bool, 1> result;
            result.resize(x.shape(), 0);
            size_t index = Utils::max(x).second;

            for (size_t i = 0; i < result.length; i++)
                if (i == index) result[i] = 1;

            return result;
        };

        const std::function<float(Vector<float, 1>, Vector<float, 1>)> MeanSquaredLoss = [](Vector<float, 1> x, Vector<float, 1> y)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < x.length; i++)
                sum += (x[i] - y[i]) * (x[i] - y[i]) / 2;

            return sum / x.shape().length;
        };

        const std::function<Vector<float, 1>(Vector<float, 1>, Vector<float, 1>)> MeanSquaredLossGradient = [](Vector<float, 1> x, Vector<float, 1> y)
        { 
            Vector<float, 1> gradient;
            gradient.resize(x.shape(), 0.0f);
            
            for (size_t i = 0; i < x.length; i++)
                gradient[i] = x[i] - y[i];

            return gradient;
        };

        const std::function<float(Vector<float, 1>, Vector<bool, 1>)> CategoricalCrossEntropyLoss = [](Vector<float, 1> x, Vector<bool, 1> y)
        { 
            float sum = 0.0f;

            for (size_t i = 0; i < x.length; i++)
                sum += -y[i] * log(x[i]);

            return sum;
        };

        const std::function<Vector<float, 1>(Vector<float, 1>, Vector<bool, 1>)> CategoricalCrossEntropyLossGradient = [](Vector<float, 1> x, Vector<bool, 1> y)
        {
            Vector<float, 1> result;
            result.resize(x.shape(), 0.0f);

            for (size_t i = 0; i < x.length; i++)
                result[i] = x[i] - y[i];

            return result;
        };
    }
}

#endif
