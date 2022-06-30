#ifndef MODULE_H_
#define MODULE_H_

#include "Tensor.h"

namespace SingleNet
{
    template <typename... T>
    class Module
    {
    };

    template <class InputType, size_t... InputDim, class OutputType, size_t... OutputDim>
    class Module<Tensor<InputType, InputDim...>, Tensor<OutputType, OutputDim...>>
    {
        using InputTensor = Tensor<InputType, InputDim...>;
        using OutputTensor = Tensor<OutputType, OutputDim...>;

    public:
        Module() {}
        virtual ~Module() {}

        virtual OutputTensor forward(const InputTensor &input) = 0;
        virtual InputTensor backward(const OutputTensor &nextDelta, float learningRate) = 0;
    };

    template <typename... T>
    class Linear
    {
    };

    template <class T, size_t Input, size_t Output, size_t Batch>
    class Linear<Tensor<T, Batch, Input>, Tensor<T, Batch, Output>>
        : public Module<Tensor<T, Batch, Input>, Tensor<T, Batch, Output>>
    {
        using InputTensor = Tensor<T, Batch, Input>;
        using OutputTensor = Tensor<T, Batch, Output>;

    public:
        Linear() {}
        ~Linear() {}

        OutputTensor forward(const InputTensor &input)
        {
            layerInput = input;
            return dot(input, weights) + biases;
        }

        InputTensor backward(const OutputTensor &nextDelta, float learningRate)
        {
            weights -= dot(get_transposed(layerInput), nextDelta) / (float)Batch * learningRate;
            biases -= ([](OutputTensor delta)
                       {
                Tensor<T, Output> result(0.0f);
                for (size_t i = 0; i < Batch; i++)
                    result[i] = delta[i];
                return result; })(nextDelta) /
                      (float)Batch * learningRate;

            return dot(nextDelta, get_transposed(weights));
        }

    private:
        Tensor<T, Input, Output> weights;
        Tensor<T, Output> biases;

        Tensor<T, Batch, Input> layerInput;
    };
}

#endif