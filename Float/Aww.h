#ifndef LAST_LAYER_H
#define LAST_LAYER_H

#include <stdint.h>
#include <time.h>
#include "../Vector.h"
#include "../Defines/Defines.h"

namespace SingleNet
{
    struct Aww
    {
        Vector<float, 2> dW1, dW2;
        Vector<float, 1> dB1, dB2;
    };

    class ReLU
    {
    public:
        ReLU(){};

        Vector<float, 2> forward(Vector<float, 2> x)
        {
            mask.resize(x.shape(), 0);
            for (int i = 0; i < x.shape()[1]; i++)
                for (int j = 0; j < x.shape()[0]; j++)
                    mask[i][j] = x[i][j] <= 0 ? 1 : 0;

            Vector<float, 2> y = x;
            for (int i = 0; i < x.shape()[1]; i++)
                for (int j = 0; j < x.shape()[0]; j++)
                    if (mask[i][j] == 1)
                        y[i][j] = 0;

            return y;
        }

        Vector<float, 2> backward(Vector<float, 2> dout)
        {
            Vector<float, 2> dx = dout;
            for (int i = 0; i < dx.shape()[1]; i++)
                for (int j = 0; j < dx.shape()[0]; j++)
                    if (mask[i][j] == 1)
                        dx[i][j] = 0;

            return dx;
        }

        Vector<bool, 2> mask;
    };

    class Sigmoid
    {
    public:
        Sigmoid(){};
        Vector<float, 2> forward(Vector<float, 2> x)
        {
            Vector<float, 2> y = x.map(Defines::Sigmoid);
            out = y;
            return y;
        }
        Vector<float, 2> backward(Vector<float, 2> dout)
        {
            Vector<float, 2> dx;
            dx.resize(out.shape(), 0.0f);

            for(int i = 0; i < out.shape()[1]; i++)
                for(int j = 0; j < out.shape()[0]; j++)
                    dx[i][j] = dout[i][j] * (1 - out[i][j]) * out[i][j];
                     
            return dx;
        }

        Vector<float, 2> out;
    };

    class Affine
    {
    public:
        Affine(){};
        Affine(size_t input_size, size_t output_size)
        {
            this->input_size = input_size;
            this->output_size = output_size;

            this->W = Vector<float, 2>();
            this->b = Vector<float, 1>();

            this->W.resize({input_size, output_size}, 0.0f);
            this->b.resize({output_size}, 0.0f);

            for (size_t i = 0; i < output_size; i++)
            {
                for (size_t j = 0; j < input_size; j++)
                    W[i][j] = (float)(rand() % 100) / 100 - 0.5; // -0.5 .. 0.5
                b[i] = (float)(rand() % 100) / 100 - 0.5;             // -0.5 .. 0.5
            }
        }

        Vector<float, 2> forward(Vector<float, 2> x)
        {
            this->x = x;
            Vector<float, 2> out = Utils::dot(x, Utils::transpose(W));
            for (int i = 0; i < out.shape()[1]; i++)
                for(int j = 0; j < out.shape()[0]; j++)
                    out[i][j] += b[i];

            return out;
        }

        Vector<float, 2> backward(Vector<float, 2> dout)
        {
            Vector<float, 2> dx = Utils::dot(Utils::transpose(W), dout);

            dW = Utils::dot(dout, x);
            dB.resize({output_size}, 0.0f);
            for (int i = 0; i < output_size; i++)
                dB[i] = Utils::sum(dout[i]);

            return dx;
        }

        Vector<float, 1> b;
        Vector<float, 2> W;

        Vector<float, 1> dB;
        Vector<float, 2> dW;

        Vector<float, 2> x;

        size_t input_size;
        size_t output_size;
    };

    class SoftmaxWithLoss
    {
    public:
        SoftmaxWithLoss(){};

        float forward(Vector<float, 2> x, Vector<bool, 2> t)
        {
            this->t = t;

            this->y.resize(x.shape(), 0.0f);
            for (int i = 0; i < x.shape()[1]; i++)
                y[i] = Defines::Softmax(x[i]);

            this->loss.resize({x.shape()[1]}, 0.0f);
            for (int i = 0; i < x.shape()[1]; i++)
                loss[i] = Defines::CategoricalCrossEntropyLoss(y[i], t[i]);

            return Utils::sum(loss) / x.shape()[1];
        }

        Vector<float, 2> backward(float dout = 1)
        {
            Vector<float, 2> dx;
            dx.resize({y.shape()[0], y.shape()[1]}, 0.0f);
            for (int i = 0; i < y.shape()[1]; i++)
                for (int j = 0; j < y.shape()[0]; j++)
                    dx[i][j] = y[i][j] - t[i][j];

            return dx;
        }

        Vector<float, 1> loss;
        Vector<float, 2> y;
        Vector<bool, 2> t;
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

        float loss(Vector<float, 2> x, Vector<bool, 2> t)
        {
            Vector<float, 2> y = forward(x);
            return softmax.forward(y, t);
        }

        float accuracy(Vector<float, 2> x, Vector<bool, 2> t)
        {
            Vector<bool, 2> y_t;
            y_t.resize({t.shape()[0], output_size}, false);

            auto res = forward(x);
            for (int i = 0; i < y_t.shape()[1]; i++)
                y_t[i] = Defines::OneHot(res[i]);

            size_t correct = 0;
            for (int i = 0; i < x.shape()[1]; i++)
                if (y_t[i] == t[i])
                    correct++;

            return correct / (float)x.shape()[1];
        }

        Vector<float, 2> forward(Vector<float, 2> x)
        {
            Vector<float, 2> h = affine1.forward(x);
            Vector<float, 2> y = relu1.forward(h);
            Vector<float, 2> out = affine2.forward(y);
            return out;
        }

        Vector<float, 2> backward(Vector<float, 2> dout)
        {
            Vector<float, 2> dh = affine2.backward(dout);
            Vector<float, 2> dy = relu1.backward(dh);
            Vector<float, 2> dx = affine1.backward(dy);
            return dx;
        }

        Aww gradient(Vector<float, 2> x, Vector<bool, 2> t)
        {
            loss(x, t);
            float dout = 1;
            Vector<float, 2> dx = softmax.backward(dout);

            dx = affine2.backward(dx);
            dx = relu1.backward(dx);
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