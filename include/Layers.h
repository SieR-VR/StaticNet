/* Copyright 2021- SieR-VR */

#ifndef FLOAT_LAYERS_H_
#define FLOAT_LAYERS_H_

#include <functional>

#include "Tensor.h"
#include "Defines.h"

namespace SingleNet
{
    // ------------------------------------------------------------------------
    // Base class for layers
    // ------------------------------------------------------------------------

    template <typename T, size_t Input, size_t Output, size_t Batch>
    class BaseNet
    {
    public:
        BaseNet() {}
        ~BaseNet() {}

        virtual Tensor<T, Batch, Output> Forward(const Tensor<T, Batch, Input> &input) = 0;
        virtual Tensor<T, Batch, Input> Backward(const Tensor<T, Batch, Output> &nextDelta,
                                                 const T &learningRate) = 0;

    private:
        Tensor<T, Input, Output> weights;
        Tensor<T, Output> biases;
    };

    template <typename T, size_t Input, size_t Batch>
    class BaseActivation
    {

    public:
        BaseActivation() {}
        virtual ~BaseActivation() {}
        virtual Tensor<T, Batch, Input> Forward(const Tensor<T, Batch, Input> &input) = 0;
        virtual Tensor<T, Batch, Input> Backward(const Tensor<T, Batch, Input> &nextDelta,
                                                 const T &learningRate) = 0;
    };

    // ------------------------------------------------------------------------
    // implementation
    // ------------------------------------------------------------------------

    template <size_t Input, size_t Output, size_t Batch>
    class Dense : public BaseNet<float, Input, Output, Batch>
    {
    public:
        Dense() {}
        Tensor<float, Batch, Output> Forward(const Tensor<float, Batch, Input> &input)
        {
            layerInput = input;
            Tensor<float, Batch, Output> result = dot(input, weights);
            for (size_t i = 0; i < Batch; i++)
                result[i] += biases;
        }

        Tensor<float, Batch, Input> Backward(const Tensor<float, Batch, Output> &nextDelta,
                                             const T &learningRate)
        {
            Tensor<float, Batch, Input> delta = dot(nextDelta, get_transposed(weights));
            weights -= dot(get_transposed(layerInput), nextDelta) / (float)Batch;
            biases -= delta.map([](Tensor<float, Batch, Input>) {
                Tensor<float, Input> result;
                for (size_t i = 0; i < Batch; i++)
                    result += Input;
                return (result / (float)Batch);
            });

            return delta;
        }

    private:
        Tensor<float, Batch, Input> layerInput;
    };

    template <size_t Input, size_t Batch>
    class Activation : public BaseActivation<float, Input, Batch>
    {
    public:
        Activation(const std::function<float(float)> &activation,
                   const std::function<float(float)> &activationDerivative)
            : m_activation(activation), m_activationDerivative(activationDerivative) {}

        Tensor<float, Batch, Input> Forward(const Tensor<float, Batch, Input> &input)
        {
            layerInput = input;
            return input.map(m_activation);
        }
        Tensor<float, Batch, Input> Backward(const Tensor<float, Batch, Input> &nextDelta,
                                             const float &learningRate)
        {
            return conv(nextDelta, input.map(m_activationDerivative));
        }

    private:
        std::function<float(float)> m_activation;
        std::function<float(float)> m_activationDerivative;

      1  Tensor<float, Batch, Input> layerInput;
    };

    template <size_t Input, size_t Batch>
    class Sigmoid : public BaseActivation<float, Input, Batch>
    {
    public:
        Sigmoid();
        Tensor<float, Batch, Input> Forward(const Tensor<float, Batch, Input> &input) override;
        Tensor<float, Batch, Input> Backward(const Tensor<float, Batch, Input> &nextDelta,
                                             const float &learningRate) override;

    private:
        std::function<float(float)> m_activation;
        std::function<float(float)> m_activationDerivative;

        Tensor<float, Batch, Input> layerOutput;
    };

    template <size_t Input, size_t Batch>
    class Softmax : public BaseActivation<float, Input, Batch>
    {
    public:
        Softmax();
        Tensor<float, Batch, Input> Forward(const Tensor<float, Batch, Input> &input) override;
        Tensor<float, Batch, Input> Backward(const Tensor<float, Batch, Input> &nextDelta,
                                             const float &learningRate) override;

    private:
        std::function<Tensor<float, Input>(Tensor<float, Input>)> m_activation;
    };
}; // namespace SingleNet

#endif // FLOAT_LAYERS_H_
