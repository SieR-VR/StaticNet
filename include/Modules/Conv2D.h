#ifndef CONV2D_H_
#define CONV2D_H_

#include "Module.h"

namespace SingleNet
{
    template <typename... T>
    class Conv2D
    {
        Conv2D() = delete;
    };

    template <class T, size_t IDim, size_t ODim, size_t I, size_t O>
    class Conv2D<Tensor<T, I, IDim, IDim>, Tensor<T, O, ODim, ODim>> 
        : public Module<T>
    {
        static_assert(O % I == 0, "Output channels must be a multiple of input channels");
        static constexpr size_t KDim = IDim-ODim+1;
        static constexpr size_t Channel = O/I;

    public:
        Conv2D(Module<T> *parent) : Module<T>("Conv2D", parent, KDim * KDim + ODim * ODim) {};

        template <size_t Batch>
        Tensor<T, Batch, O, ODim, ODim> forward(const Tensor<T, Batch, I, IDim, IDim> &input)
        {
            this->memory(AccessType::Write, input);
            Tensor<T, Batch, O, ODim, ODim> result;
            for (size_t i = 0; i < Batch; i++)
                for (size_t j = 0; j < O; j++)
                    result[i][j] = conv(input[i][j % I], kernel[j % Channel]) + biases[j % Channel];

            return result;
        }

        template <size_t Batch>
        Tensor<T, Batch, I, IDim, IDim> backward(const Tensor<T, Batch, O, ODim, ODim> &nextDelta, float learningRate)
        {   
            Tensor<T, Batch, I, IDim, IDim> delta;
            for (size_t i = 0; i < Batch; i++)
                for (size_t k = 0; k < O; k++) 
                    delta[i][k % I] += conv(pad<T, ODim, KDim-1>(nextDelta[i][k]), flip(kernel[k % Channel]));

            Tensor<T, Batch, I, IDim, IDim> input = this->memory(AccessType::Read, Tensor<T, Batch, I, IDim, IDim>());
            Tensor<T, Channel, KDim, KDim> dk = T();
            Tensor<T, Channel, ODim, ODim> db = T();

            for (size_t i = 0; i < Batch; i++)
                for (size_t j = 0; j < O; j++) {
                    dk[j % Channel] += conv(input[i][j % I], nextDelta[i][j]) / (float)Batch;
                    db[j % Channel] += nextDelta[i][j] / (float)Batch;
                }

            kernel -= dk * learningRate;
            biases -= db * learningRate;

            return delta;
        }

    private:
        Tensor<T, Channel, KDim, KDim> kernel = Tensor<T, Channel, KDim, KDim>::random();
        Tensor<T, Channel, ODim, ODim> biases = Tensor<T, Channel, ODim, ODim>::random();
    };
}

#endif