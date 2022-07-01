#ifndef RENET_H_
#define RENET_H_

#include "Modules/Linear.h"
#include "Modules/Conv2D.h"
#include "Modules/AvgPool2D.h"
#include "Modules/ReLU.h"

namespace SingleNet
{
    class LeNet : public Module<float>
    {
    public:
        LeNet()
            : Module<float>("LeNet"),
              conv1(this),
              avgpool1(this),
              relu1(this),
              conv2(this),
              avgpool2(this),
              relu2(this),
              fc1(this)
        {
        }

        template <size_t Batch>
        Tensor<float, Batch, 10> forward(const Tensor<float, Batch, 1, 28, 28> &input)
        {
            auto x1 = conv1.forward(input);
            auto x2 = avgpool1.forward(x1);
            auto x3 = relu1.forward(x2);
            auto x4 = conv2.forward(x3);
            auto x5 = avgpool2.forward(x4);
            auto x6 = relu2.forward(x5);
            auto x7 = x6.template reshape<Batch, 192>();
            return fc1.forward(x7);
        }

        template <size_t Batch>
        Tensor<float, Batch, 1, 28, 28> backward(const Tensor<float, Batch, 10> &delta, float learningRate)
        {
            auto d1 = fc1.backward(delta, learningRate);
            auto d2 = d1.template reshape<Batch, 12, 4, 4>();
            auto d3 = relu2.backward(d2, learningRate);
            auto d4 = avgpool2.backward(d3, learningRate);
            auto d5 = conv2.backward(d4, learningRate);
            auto d6 = relu1.backward(d5, learningRate);
            auto d7 = avgpool1.backward(d6, learningRate);
            return conv1.backward(d7, learningRate);
        }

    private:
        Conv2D<Tensor<float, 1, 28, 28>, Tensor<float, 4, 24, 24>> conv1;
        AvgPool2D<Tensor<float, 4, 24, 24>, Tensor<float, 4, 12, 12>> avgpool1;
        ReLU<Tensor<float, 4, 12, 12>> relu1;
        Conv2D<Tensor<float, 4, 12, 12>, Tensor<float, 12, 8, 8>> conv2;
        AvgPool2D<Tensor<float, 12, 8, 8>, Tensor<float, 12, 4, 4>> avgpool2;
        ReLU<Tensor<float, 12, 4, 4>> relu2;
        Linear<Tensor<float, 192>, Tensor<float, 10>> fc1;
    };
}
#endif