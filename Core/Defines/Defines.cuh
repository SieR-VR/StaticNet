#ifndef DEFINES_CUH_
#define DEFINES_CUH_

#include "Core/Vector/VectorCUDA.h"

namespace SingleNet
{
    namespace Defines
    {
        // GPU Functions

        // ------------------------------------------------------------
        // Activation functions
        // ------------------------------------------------------------

        // Sigmoid function
        __device__ float SigmoidCUDA(float x);
        __device__ float SigmoidDerivativeCUDA(float x);
        __device__ float SigmoidDerivativeCUDA_(float x);

        // Tanh function
        __device__ float TanhCUDA(float x);
        __device__ float TanhDerivativeCUDA(float x);
        __device__ float TanhDerivativeCUDA_(float x);

        // ReLU function
        __device__ float ReLUCUDA(float x);
        __device__ float ReLUDerivativeCUDA(float x);
    }
}

#endif
