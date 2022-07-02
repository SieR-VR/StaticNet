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
        Linear(Module<T> *parent) : Module<T>("Linear", parent, (Input) * (Output + 1)) {};
        ~Linear() {}

        template <size_t Batch>
        Tensor<T, Batch, Output> forward(const Tensor<T, Batch, Input> &input)
        {
            this->memory(AccessType::Write, input);
            auto result = dot(input, weights);
            for (size_t i = 0; i < Batch; i++)
                result[i] += biases;

            return result;
        }

        template <size_t Batch>
        Tensor<T, Batch, Input> backward(const Tensor<T, Batch, Output> &nextDelta, float learningRate)
        {
            Tensor<T, Batch, Input> input = this->memory(AccessType::Read, Tensor<T, Batch, Input>());

            weights -= (dot(get_transposed(input), nextDelta) / (float)Batch) * learningRate;
            biases -= (nextDelta.reduce() / (float)Batch) * learningRate;

            return dot(nextDelta, get_transposed(weights));
        }

    private:
        Tensor<T, Input, Output> weights = Tensor<T, Input, Output>::random();
        Tensor<T, Output> biases = Tensor<T, Output>::random();
    };
}

#endif