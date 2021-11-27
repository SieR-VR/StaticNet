#include <cmath>
#include "VectorCUDA.h"

namespace SingleNet
{
    namespace Defines
    {
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