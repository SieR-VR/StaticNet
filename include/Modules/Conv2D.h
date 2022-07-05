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
        static constexpr size_t KDim = IDim - ODim + 1;

    public:
        Conv2D(Module<T> *parent) : Module<T>("Conv2D", parent, KDim * KDim + ODim * ODim){};

        template <size_t Batch>
        Tensor<T, Batch, FN, ODim, ODim> forward(const Tensor<T, Batch, C, IDim, IDim> &input)
        {
            auto col = im2col<KDim>(input);
            this->memory(AccessType::Write, col);
            auto kernel_reshaped = kernel.template reshape_ref<FN, C * KDim * KDim>();
            Tensor<T, Batch, FN, ODim, ODim> result = dot(col, kernel_reshaped.template transpose<1, 0>())
                                                          .template reshape<Batch, FN, ODim, ODim>();
            for (size_t i = 0; i < Batch; i++)
                for (size_t j = 0; j < FN; j++)
                    for (size_t k = 0; k < ODim; k++)
                        for (size_t l = 0; l < ODim; l++)
                            result[i][j][k][l] += biases[j][0][0];

            return result;
        }

        template <size_t Batch>
        Tensor<T, Batch, C, IDim, IDim> backward(const Tensor<T, Batch, FN, ODim, ODim> &dout, float learningRate)
        {
            auto db_pre = dout.reduce();
            Tensor<T, FN, 1, 1> db;
            for (size_t i = 0; i < FN; i++)
                db[i][0][0] = db_pre[i].reduce().reduce();

            auto dout_reshaped = dout.template transpose<1, 2, 3, 0>().template reshape<FN, Batch * ODim * ODim>();
            auto input_transposed = this->memory<Batch * ODim * ODim, C * KDim * KDim>(AccessType::Read);

            auto dw_pre = dot(dout_reshaped, input_transposed);
            auto dw = dw_pre.template reshape<FN, C, KDim, KDim>();

            auto w_reshaped = kernel.template reshape_ref<FN, C * KDim * KDim>();
            auto dx_col = dot(w_reshaped.template transpose<1, 0>(), dout_reshaped);
            auto dx = col2im<float, KDim, Batch, C, IDim>(dx_col);

            kernel -= dw / (float)Batch * learningRate;
            biases -= db / (float)Batch * learningRate;

            return dx;
        }

    private:
        Tensor<T, FN, C, KDim, KDim> kernel = Tensor<T, FN, C, KDim, KDim>::random();
        Tensor<T, FN, 1, 1> biases = Tensor<T, FN, 1, 1>::random();
    };
}

#endif