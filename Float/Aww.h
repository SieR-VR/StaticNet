#ifndef LAST_LAYER_H
#define LAST_LAYER_H

#include <stdint.h>
#include <time.h>
#include "../Structure/Vector3D.h"
#include "../Tools/Defines.h"

namespace SingleNet
{
    struct Aww
    {
        Vector2D<float> dW1, dW2;
        Vector1D<float> dB1, dB2;
    };

    class ReLU
    {
    public:
        ReLU(){};

        Vector2D<float> forward(Vector2D<float> x)
        {
            mask.resize(x.shape(), 0);
            for (int i = 0; i < x.shape().y; i++)
                for (int j = 0; j < x.shape().x; j++)
                    mask[i][j] = x[i][j] <= 0 ? 1 : 0;

            Vector2D<float> y = x;
            for (int i = 0; i < x.shape().y; i++)
                for (int j = 0; j < x.shape().x; j++)
                    if (mask[i][j] == 1)
                        y[i][j] = 0;

            return y;
        }

        Vector2D<float> backward(Vector2D<float> dout)
        {
            Vector2D<float> dx = dout;
            for (int i = 0; i < dx.shape().y; i++)
                for (int j = 0; j < dx.shape().x; j++)
                    if (mask[i][j] == 1)
                        dx[i][j] = 0;

            return dx;
        }

        Vector2D<int> mask;
    };

    class Sigmoid
    {
    public:
        Sigmoid(){};
        Vector2D<float> forward(Vector2D<float> x)
        {
            Vector2D<float> y = x.map(Defines::Sigmoid);
            out = y;
            return y;
        }
        Vector2D<float> backward(Vector2D<float> dout)
        {
            Vector2D<float> dx = dout * (out * -1 + 1) * out;
            return dx;
        }

        Vector2D<float> out;
    };

    class Affine
    {
    public:
        Affine(){};
        Affine(size_t input_size, size_t output_size)
        {
            this->input_size = input_size;
            this->output_size = output_size;

            this->W.resize({input_size, output_size}, 0.0f);
            this->b.resize({output_size}, 0.0f);

            for (size_t i = 0; i < output_size; i++)
            {
                for (size_t j = 0; j < input_size; j++)
                    W.at({j, i}) = (float)(rand() % 100) / 100 - 0.5; // -0.5 .. 0.5
                b[i] = (float)(rand() % 100) / 100 - 0.5;             // -0.5 .. 0.5
            }
        }

        Vector2D<float> forward(Vector2D<float> x)
        {
            this->x = x;
            Vector2D<float> out = W.dot(x.transpose());
            for (int i = 0; i < out.shape().y; i++)
                out[i] += b[i];

            return out;
        }

        Vector2D<float> backward(Vector2D<float> dout)
        {
            Vector2D<float> dx = dout.dot(W.transpose());

            dW = x.dot(dout);
            dB.resize({output_size}, 0.0f);
            for (int i = 0; i < output_size; i++)
                dB[i] = dout[i].mean();

            return dx;
        }

        Vector1D<float> b;
        Vector2D<float> W;

        Vector1D<float> dB;
        Vector2D<float> dW;

        Vector2D<float> x;

        size_t input_size;
        size_t output_size;
    };

    class SoftmaxWithLoss
    {
    public:
        SoftmaxWithLoss(){};

        float forward(Vector2D<float> x, Vector2D<int> t)
        {
            this->t = t;

            this->y.resize(x.shape(), 0.0f);
            for (int i = 0; i < x.shape().y; i++)
                y[i] = Defines::Softmax(x[i]);

            this->loss.resize({x.shape().y}, 0.0f);
            for (int i = 0; i < x.shape().y; i++)
                loss[i] = Defines::CategoricalCrossEntropyLoss(y[i], t[i]);

            return loss.mean() / x.shape().y;
        }

        Vector2D<float> backward(float dout = 1)
        {
            Vector2D<float> dx;
            dx.resize({y.shape().x, y.shape().y}, 0.0f);
            for (int i = 0; i < y.shape().y; i++)
                for (int j = 0; j < y.shape().x; j++)
                    dx[i][j] = y[i][j] - t[i][j];

            return dx;
        }

        Vector1D<float> loss;
        Vector2D<float> y;
        Vector2D<int> t;
    };

    class TwoLayerNet
    {
    public:
        TwoLayerNet(size_t input_size, size_t hidden_size, size_t output_size)
        {
            this->input_size = input_size;
            this->hidden_size = hidden_size;
            this->output_size = output_size;

            this->affine1 = Affine(input_size, hidden_size);
            this->relu1 = ReLU();
            this->affine2 = Affine(hidden_size, output_size);
            this->softmax = SoftmaxWithLoss();
        }

        float loss(Vector2D<float> x, Vector2D<int> t)
        {
            Vector2D<float> y = forward(x);
            return softmax.forward(y, t);
        }

        float accuracy(Vector2D<float> x, Vector2D<int> t)
        {
            Vector2D<int> y_t;
            y_t.resize({t.shape().x, output_size});

            auto res = forward(x);
            for (int i = 0; i < y_t.shape().y; i++)
                y_t[i] = Defines::OneHot(res[i]);

            size_t correct = 0;
            for (int i = 0; i < x.shape().y; i++)
                if (y_t[i] == t[i])
                    correct++;

            return correct / (float)x.shape().y;
        }

        Vector2D<float> forward(Vector2D<float> x)
        {
            Vector2D<float> h = affine1.forward(x);
            Vector2D<float> y = relu1.forward(h);
            Vector2D<float> out = affine2.forward(y);
            return out;
        }

        Vector2D<float> backward(Vector2D<float> dout)
        {
            Vector2D<float> dh = affine2.backward(dout);
            Vector2D<float> dy = relu1.backward(dh);
            Vector2D<float> dx = affine1.backward(dy);
            return dx;
        }

        Aww gradient(Vector2D<float> x, Vector2D<int> t)
        {
            loss(x, t);
            float dout = 1;
            Vector2D<float> dx = softmax.backward(dout).transpose();

            dx = affine2.backward(dx).transpose();
            dx = relu1.backward(dx).transpose();
            dx = affine1.backward(dx);

            return {affine1.dW, affine2.dW, affine1.dB, affine2.dB};
        }

        size_t input_size;
        size_t hidden_size;
        size_t output_size;

        Affine affine1;
        ReLU relu1;
        Affine affine2;
        SoftmaxWithLoss softmax;
    };
};

#endif