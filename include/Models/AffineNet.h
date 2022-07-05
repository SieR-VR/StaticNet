#ifndef AFFINENET_H_
#define AFFINENET_H_

#include "Modules/Linear.h"
#include "Modules/ReLU.h"

namespace StaticNet
{
    class AffineNet : public Module<float>
    {
    public:
        AffineNet()
            : Module<float>("AffineNet"),
              conv1(this),
              relu1(this),
              conv2(this),
              relu2(this),
              fc1(this)
        {
        }

        template <size_t Batch>
        Tensor<float, Batch, 10> forward(const Tensor<float, Batch, 784> &input)
        {
            auto x1 = conv1.forward(input);
            auto x2 = relu1.forward(x1);
            auto x3 = conv2.forward(x2);
            auto x4 = relu2.forward(x3);
            return fc1.forward(x4);
        }

        template <size_t Batch>
        Tensor<float, Batch, 784> backward(const Tensor<float, Batch, 10> &delta, float learningRate)
        {
            auto d4 = fc1.backward(delta, learningRate);
            auto d3 = relu2.backward(d4, learningRate);
            auto d2 = conv2.backward(d3, learningRate);
            auto d1 = relu1.backward(d2, learningRate);
            return conv1.backward(d1, learningRate);
        }

    private:
        Linear<Tensor<float, 784>, Tensor<float, 300>> conv1;
        ReLU<Tensor<float, 300>> relu1;
        Linear<Tensor<float, 300>, Tensor<float, 200>> conv2;
        ReLU<Tensor<float, 200>> relu2;
        Linear<Tensor<float, 200>, Tensor<float, 10>> fc1;
    };
}
#endif