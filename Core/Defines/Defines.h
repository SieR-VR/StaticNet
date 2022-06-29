#ifndef DEFINES_H_
#define DEFINES_H_

#include <functional>
#include "Core/Vector/Vector.h"

namespace SingleNet
{
    namespace Defines
    {
        // CPU Fuctions

        // ------------------------------------------------------------
        // Activation functions
        // ------------------------------------------------------------

        // Sigmoid function
        extern std::function<float(float)> Sigmoid;
        extern std::function<float(float)> SigmoidDerivative;
        extern std::function<float(float)> SigmoidDerivative_;

        // Tanh function
        extern std::function<float(float)> Tanh;
        extern std::function<float(float)> TanhDerivative;
        extern std::function<float(float)> TanhDerivative_;

        // ReLU function
        extern std::function<float(float)> ReLU;
        extern std::function<float(float)> ReLUDerivative;

        // Lnh function
        extern std::function<float(float)> Lnh;
        extern std::function<float(float)> LnhDerivative;

        // Softmax function
        extern std::function<Vector<float, 1>(Vector<float, 1>)> Softmax;

        // ------------------------------------------------------------
        // Loss functions
        // ------------------------------------------------------------

        // Mean squared error
        extern std::function<float(Vector<float, 1>, Vector<float, 1>)> MSE;

        // Cross-entropy
        extern std::function<float(Vector<float, 1>, Vector<float, 1>)> CrossEntropy;
    }
}

#endif
