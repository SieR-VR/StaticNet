#ifndef AVG_POOL_2D_H_
#define AVG_POOL_2D_H_

#include "Module.h"

namespace StaticNet
{
    template <typename... T>
    class AvgPool2D
    {
        AvgPool2D() = delete;
    };

    template <class T, size_t IDim, size_t ODim, size_t I>
    class AvgPool2D<Tensor<T, I, IDim, IDim>, Tensor<T, I, ODim, ODim>> : public Module<T>
    {
        static_assert(IDim % ODim == 0, "Output features must be a multiple of input features");
        static constexpr size_t KDim = IDim/ODim;

    public:
        AvgPool2D(Module<T> *parent) : Module<T>("AvgPool2D", parent) {};

        template <size_t Batch>
        Tensor<T, Batch, I, ODim, ODim> forward(const Tensor<T, Batch, I, IDim, IDim> &input)
        {
            this->memory(AccessType::Write, input);
            Tensor<T, Batch, I, ODim, ODim> result;
            for (size_t i = 0; i < Batch; i++)
                for (size_t j = 0; j < I; j++)
                    result[i][j] = pool<T, IDim, ODim>(input[i][j], pool_func);
            return result;
        }

        template <size_t Batch>
        Tensor<T, Batch, I, IDim, IDim> backward(const Tensor<T, Batch, I, ODim, ODim> &nextDelta, float learningRate)
        {
            Tensor<T, Batch, I, IDim, IDim> delta;
            // for (size_t i = 0; i < Batch; i++)
            //     for (size_t k = 0; k < I; k++) 
            //         delta[i][k] = unpool<T, IDim, ODim>(nextDelta[i][k], unpool_func); 
                    
            return delta;
        }

    private:
        std::function<T(Tensor<T, KDim, KDim>)> pool_func = [](Tensor<T, KDim, KDim> input) {
            return input.reduce().reduce() / (KDim * KDim);
        };
        std::function<Tensor<T, KDim, KDim>(T)> unpool_func = [](T input) {
            return Tensor<T, KDim, KDim>(input / (KDim * KDim));
        };
    };
}

#endif