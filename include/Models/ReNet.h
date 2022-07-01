#ifndef RENET_H_
#define RENET_H_

#include "Modules/Linear.h"
#include "Modules/Conv2D.h"
#include "Modules/ReLU.h"

namespace SingleNet
{
    class ReNet : public Module<float>
    {
    public:
        ReNet()
            : Module<float>("ReNet"),
              conv1(this),
              relu1(this),
              conv2(this),
              relu2(this),
              fc1(this)
        {
        }

        template <size_t Batch>
        Tensor<float, Batch, 10> forward(const Tensor<float, Batch, 28, 28> &input)
        {
            auto x1 = conv1.forward(input);
            auto x2 = relu1.forward(x1);
            auto x3 = conv2.forward(x2);
            auto x4 = relu2.forward(x3);
            return fc1.forward(x4);
        }

        template <size_t Batch>
        Tensor<float, Batch, 28, 28> backward(const Tensor<float, Batch, 10> &delta, float learningRate)
        {
            auto d4 = fc1.backward(delta, learningRate);
            auto d3 = relu2.backward(d4);
            auto d2 = conv2.backward(d3, learningRate);
            auto d1 = relu1.backward(d2);
            return conv1.backward(d1, learningRate);
        }

    private:
        Conv2D<Tensor<float, 28, 28>, Tensor<float, 14, 14>> conv1;
        ReLU<Tensor<float, 14, 14>> relu1;
        Conv2D<Tensor<float, 14, 14>, Tensor<float, 5, 5>> conv2;
        ReLU<Tensor<float, 5, 5>> relu2;
        Linear<Tensor<float, 25>, Tensor<float, 10>> fc1;
    };
}
#endif