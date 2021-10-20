#include <cmath>
#include "Core/Defines/Defines.cuh"

namespace SingleNet
{
    namespace Defines
    {
        // ------------------------------------------------------------
        // Activation functions
        // ------------------------------------------------------------

        // Sigmoid function
        std::function<float(float)> Sigmoid = [](float x)
        { return 1.0f / (1.0f + std::exp(-x)); };
        std::function<float(float)> SigmoidDerivative = [](float x)
        { return Sigmoid(x) * (1.0f - Sigmoid(x)); };
        std::function<float(float)> SigmoidDerivative_ = [](float x)
        { return x * (1.0f - x); };

        // Tanh function
        std::function<float(float)> Tanh = [](float x)
        { return std::tanh(x); };
        std::function<float(float)> TanhDerivative = [](float x)
        { return 1.0f - std::pow(Tanh(x), 2); };
        std::function<float(float)> TanhDerivative_ = [](float x)
        { return 1.0f - x * x; };

        // ReLU function
        std::function<float(float)> ReLU = [](float x)
        { return x > 0.0f ? x : 0.0f; };
        std::function<float(float)> ReLUDerivative = [](float x)
        { return x > 0.0f ? 1.0f : 0.0f; };

        // Softmax function
        std::function<Vector<float, 1>(Vector<float, 1>)> Softmax = [](Vector<float, 1> x)
        {
            float max = x[maxIndex(x)];
            float sum = 0.0f;

            for (size_t i = 0; i < x.size(); i++)
            {
                x[i] = std::exp(x[i] - max);
                sum += x[i];
            }

            for (size_t i = 0; i < x.size(); i++)
                x[i] /= sum;

            return x;
        };

        // ------------------------------------------------------------
        // Loss functions
        // ------------------------------------------------------------

        // Mean squared error
        std::function<float(Vector<float, 1>, Vector<float, 1>)> MSE = [](Vector<float, 1> y, Vector<float, 1> y_)
        {
            float sum = 0.0f;

            for (size_t i = 0; i < y.size(); i++)
                sum += std::pow(y[i] - y_[i], 2);

            return sum / y.size();
        };

        // Cross-entropy
        std::function<float(Vector<float, 1>, Vector<float, 1>)> CrossEntropy = [](Vector<float, 1> y, Vector<float, 1> y_)
        {
            float sum = 0.0f;

            for (size_t i = 0; i < y.size(); i++)
                if(y[i]) sum -= std::log(y_[i]);

            return sum;
        };

        __device__ float SigmoidCUDA(float x)
        {
            return 1.0f / (1.0f + std::exp(-x));
        };

        __device__ float SigmoidDerivativeCUDA(float x)
        {
            return SigmoidCUDA(x) * (1.0f - SigmoidCUDA(x));
        };

        __device__ float SigmoidDerivativeCUDA_(float x)
        {
            return x * (1.0f - x);
        };

        __device__ float TanhCUDA(float x)
        {
            return std::tanh(x);
        };

        __device__ float TanhDerivativeCUDA(float x)
        {
            return 1.0f - std::pow(TanhCUDA(x), 2);
        };

        __device__ float TanhDerivativeCUDA_(float x)
        {
            return 1.0f - x * x;
        };

        __device__ float ReLUCUDA(float x)
        {
            return x > 0.0f ? x : 0.0f;
        };

        __device__ float ReLUDerivativeCUDA(float x)
        {
            return x > 0.0f ? 1.0f : 0.0f;
        };
    }
}