#ifndef CONV2D_H_
#define CONV2D_H_

#include "Module.h"

namespace StaticNet
{
    template <typename... T>
    class Conv2D
    {
        Conv2D() = delete;
    };

    template <class T, size_t IDim, size_t ODim, size_t C, size_t FN>
    class Conv2D<Tensor<T, C, IDim, IDim>, Tensor<T, FN, ODim, ODim>> 
        : public Module<T>
    {
        static constexpr size_t KDim = IDim-ODim+1;

    public:
        Conv2D(Module<T> *parent) : Module<T>("Conv2D", parent, KDim * KDim + ODim * ODim) {};

        template <size_t Batch>
        Tensor<T, Batch, FN, ODim, ODim> forward(const Tensor<T, Batch, C, IDim, IDim> &input)
        {
            this->memory(AccessType::Write, input);
            auto result = conv(input, kernel);

            return result;
        }

        template <size_t Batch>
        Tensor<T, Batch, C, IDim, IDim> backward(const Tensor<T, Batch, FN, ODim, ODim> &nextDelta, float learningRate)
        {   
            // Tensor<T, FN * C, KDim, KDim> kernel_filped = flip(kernel.template reshape_ref<FN * C, KDim, KDim>());

            Tensor<T, Batch, C, IDim, IDim> delta; // = conv(nextDelta, kernel_filped.template reshape_ref<C, FN, KDim, KDim>());
            // auto nextDelta_reshaped = transpose(nextDelta.template reshape_ref<Batch, FN * ODim * ODim>())
            //     .template reshape_ref<FN, Batch * ODim, ODim>();

            // Tensor<T, Batch, C, IDim, IDim> input = this->memory(AccessType::Read, Tensor<T, Batch, C, IDim, IDim>());
            // Tensor<T, FN, C, KDim, KDim> dk;
            // for (size_t i = 0; i < Batch; i++)
            //     for (size_t j = 0; j < FN; j++)
            //         for (size_t k = 0; k < C; k++)
            //             dk[j][k] += conv(input[i][k], nextDelta[i][j]);

            // Tensor<T, FN, 1, 1> db;

            // kernel -= dk * learningRate;
            // biases -= db * learningRate;

            return delta;
        }

    private:
        Tensor<T, FN, C, KDim, KDim> kernel = Tensor<T, FN, C, KDim, KDim>::random();
        Tensor<T, FN, 1, 1> biases = Tensor<T, FN, 1, 1>::random();
    };
}

#endif