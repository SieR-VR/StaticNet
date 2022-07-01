#ifndef LINEAR_H_
#define LINEAR_H_

#include "Module.h"

namespace SingleNet
{
    template <typename... T>
    class Linear
    {
        Linear() = delete;
    };

    template <class T, size_t Input, size_t Output>
    class Linear<Tensor<T, Input>, Tensor<T, Output>> : public Module<T>
    {
    public:
        Linear() {}
        ~Linear() {}

        template <size_t Batch>
        Tensor<T, Batch, Output> forward(const Tensor<T, Batch, Input> &input)
        {
            memory(AccessType::Write, input);
            return dot(input, weights) + biases;
        }

        template <size_t Batch>
        Tensor<T, Batch, Input> backward(const Tensor<T, Batch, Output> &nextDelta, float learningRate)
        {
            Tensor<T, Batch, Input> input = memory(AccessType::Read, Tensor<T, Batch, Input>());

            weights -= dot(get_transposed(input), nextDelta) / (float)Batch * learningRate;
            biases -= ([](Tensor<T, Batch, Output> delta) {
                          Tensor<T, Output> result(0.0f);
                          for (size_t i = 0; i < Batch; i++)
                              result[i] = delta[i];
                          return result;
                      })(nextDelta) /
                      (float)Batch * learningRate;

            return dot(nextDelta, get_transposed(weights));
        }

    private:
        Tensor<T, Input, Output> weights;
        Tensor<T, Output> biases;
    };
}

#endif