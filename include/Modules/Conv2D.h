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

    template <class T, size_t IDim, size_t ODim>
    class Conv2D<Tensor<T, IDim, IDim>, Tensor<T, ODim, ODim>> 
        : public Module<T>
    {
        static constexpr size_t KDim = IDim-ODim+1;

    public:
        Conv2D(Module<T> *parent) : Module<T>("Conv2D", parent, KDim * KDim + ODim * ODim) {};

        template <size_t Batch>
        Tensor<T, Batch, ODim, ODim> forward(const Tensor<T, Batch, IDim, IDim> &input)
        {
            this->memory(AccessType::Write, input);
            Tensor<T, Batch, ODim, ODim> result;
            for (size_t i = 0; i < Batch; i++)
                result[i] = conv<float, IDim, KDim>(input[i], kernel) + biases;

            return result;
        }

        template <size_t Batch>
        Tensor<T, Batch, IDim, IDim> backward(const Tensor<T, Batch, ODim, ODim> &nextDelta, float learningRate)
        {   
            auto delta = conv(pad<T, ODim, KDim-1>(nextDelta), flip(kernel));

            Tensor<T, Batch, IDim, IDim> input = memory(AccessType::Read, Tensor<T, Batch, IDim, IDim>());
            kernel -= conv(input, nextDelta).reduce() / (float)Batch * learningRate;
            biases -= nextDelta.reduce() / (float)Batch * learningRate;

            return delta;
        }

    private:
        Tensor<T, KDim, KDim> kernel = Tensor<T, KDim, KDim>::random();
        Tensor<T, ODim, ODim> biases = Tensor<T, ODim, ODim>::random();


    };
}

#endif