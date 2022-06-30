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

        virtual Tensor<T, Batch, Output> Forward(Tensor<T, Batch, Input> &input) = 0;
        virtual Tensor<T, Batch, Input> Backward(Tensor<T, Batch, Output> &nextDelta,
                                                 T &learningRate) = 0;

        Tensor<T, Batch, Output> operator()(Tensor<T, Batch, Input> &input)
        {
            return Forward(input);
        }

    protected:
        Tensor<T, Input, Output> weights;
        Tensor<T, Output> biases;
    };

    template <typename T, size_t Input, size_t Batch>
    class BaseActivation
    {

    public:
        BaseActivation() {}
        virtual ~BaseActivation() {}
        virtual Tensor<T, Batch, Input> Forward(Tensor<T, Batch, Input> &input) = 0;
        virtual Tensor<T, Batch, Input> Backward(Tensor<T, Batch, Input> &nextDelta,
                                                 T &learningRate) = 0;
    };

    // ------------------------------------------------------------------------
    // implementation
    // ------------------------------------------------------------------------

    template <size_t Input, size_t Output, size_t Batch>
    class Dense : public BaseNet<float, Input, Output, Batch>
    {
    public:
        Dense() {}
        Tensor<float, Batch, Output> Forward(Tensor<float, Batch, Input> &input)
        {
            layerInput = input;
            Tensor<float, Batch, Output> result = dot(input, this->weights);
            for (size_t i = 0; i < Batch; i++)
                result[i] += this->biases;

            return result;
        }

        Tensor<float, Batch, Input> Backward(Tensor<float, Batch, Output> &nextDelta,
                                             float &learningRate)
        {
            Tensor<float, Batch, Input> delta = dot(nextDelta, get_transposed(this->weights));
            this->weights -= dot(get_transposed(layerInput), nextDelta) / (float)Batch;
            this->biases -= ([](Tensor<float, Batch, Output> delta) {
                Tensor<float, Output> result(0.0f);
                for (size_t i = 0; i < Output; i++)
                    result[i] = 0;

                for (size_t i = 0; i < Batch; i++)
                    result += delta[i];
                return (result / (float)Batch); 
            })(nextDelta);

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

        Tensor<float, Batch, Input> Forward(Tensor<float, Batch, Input> &input)
        {
            layerInput = input;
            return input.map(m_activation);
        }
        Tensor<float, Batch, Input> Backward(Tensor<float, Batch, Input> &nextDelta,
                                             float &learningRate)
        {
            return conv(nextDelta, layerInput.map(m_activationDerivative));
        }

    private:
        std::function<float(float)> m_activation;
        std::function<float(float)> m_activationDerivative;

        Tensor<float, Batch, Input> layerInput;
    };

    template <size_t Input, size_t Batch>
    class Sigmoid : public BaseActivation<float, Input, Batch>
    {
    public:
        Sigmoid() : m_activation(Defines::Sigmoid), m_activationDerivative(Defines::SigmoidDerivative_) {}

        Tensor<float, Batch, Input> Forward(Tensor<float, Batch, Input> &input)
        {
            layerOutput = input.map(m_activation);
            return layerOutput;
        }
        Tensor<float, Batch, Input> Backward(Tensor<float, Batch, Input> &nextDelta,
                                             float &learningRate)
        {
            return conv(nextDelta, layerOutput.map(m_activationDerivative));
        }

    private:
        std::function<float(float)> m_activation;
        std::function<float(float)> m_activationDerivative;

        Tensor<float, Batch, Input> layerOutput;
    };

    template <size_t Input, size_t Batch>
    class Softmax : public BaseActivation<float, Input, Batch>
    {
    public:
        Softmax() : m_activation(Defines::Softmax<Input>) {}
        Tensor<float, Batch, Input> Forward(Tensor<float, Batch, Input> &input)
        {
            return input.apply(m_activation);
        }

        Tensor<float, Batch, Input> Backward(Tensor<float, Batch, Input> &nextDelta,
                                             float &learningRate)
        {
            return nextDelta;
        }

    private:
        std::function<Tensor<float, Input>(Tensor<float, Input>)> m_activation;
    };

    template <size_t Input, size_t Output, size_t Batch>
    class Layer
    {
    public:
        Layer(BaseNet<float, Input, Output, Batch> *net, BaseActivation<float, Output, Batch> *activation)
            : m_net(net), m_activation(activation) {}

        Tensor<float, Batch, Output> Forward(Tensor<float, Batch, Input> &input)
        {
            return (*m_activation)((*m_net)(input));
        }

        Tensor<float, Batch, Input> Backward(Tensor<float, Batch, Output> &nextDelta,
                                             float &learningRate)
        {
            return m_net->Backward(m_activation->Backward(nextDelta, learningRate), learningRate);
        }

        Tensor<float, Batch, Output> operator()(Tensor<float, Batch, Input> &input)
        {
            return Forward(input);
        }

    private:
        BaseNet<float, Input, Output, Batch> *m_net;
        BaseActivation<float, Output, Batch> *m_activation;
    };
}; // namespace SingleNet

#endif // FLOAT_LAYERS_H_
