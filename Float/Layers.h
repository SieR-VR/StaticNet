#ifndef LAYERS_H
#define LAYERS_H

#include "../Vector.h"
#include <cmath>
#include <functional>
#include <type_traits>
#include <random>

namespace SingleNet
{
    template <typename T>
    class BaseNet
    {
        static_assert(std::is_floating_point<T>::value, "T must be a floating point type");

    public:
        BaseNet(size_t inputSize, size_t outputSize)
            : m_inputSize(inputSize), m_outputSize(outputSize), weights(outputSize, inputSize, T()), biases(outputSize, T())
        {
        }

        virtual ~BaseNet()
        {
        }

        virtual Vector<T, 2> Forward(const Vector<T, 2> &input) = 0;
        virtual Vector<T, 2> Backward(const Vector<T, 2> &nextDelta, const T &learningRate) = 0;

    protected:
        size_t m_inputSize;
        size_t m_outputSize;

        Vector<T, 2> weights;
        Vector<T, 1> biases;
    };

    template <typename T>
    class BaseActivation
    {
        static_assert(std::is_floating_point<T>::value, "T must be a floating point type");

    public:
        BaseActivation(const std::function<T(T)> &activation, const std::function<T(T)> &activationDerivative)
            : m_activation(activation), m_activationDerivative(activationDerivative)
        {
        }

        BaseActivation(const std::function<Vector<T, 1>(Vector<T, 1>)> &activation, const std::function<Vector<T, 1>(Vector<T, 1>)> &activationDerivative)
            : m_activationVector(activation), m_activationDerivativeVector(activationDerivative)
        {
        }

        virtual ~BaseActivation()
        {
        }

        virtual Vector<T, 2> Forward(const Vector<T, 2> &input) = 0;
        virtual Vector<T, 2> Backward(const Vector<T, 2> &nextDelta, const T &learningRate) = 0;

    protected:
        std::function<T(T)> m_activation;
        std::function<T(T)> m_activationDerivative;

        std::function<Vector<T, 1>(Vector<T, 1>)> m_activationVector;
        std::function<Vector<T, 1>(Vector<T, 1>)> m_activationDerivativeVector;
    };

    class Dense : public BaseNet<float>
    {
    public:
        Dense(size_t inputSize, size_t outputSize)
            : BaseNet(inputSize, outputSize)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

            for (size_t i = 0; i < outputSize; i++)
            {
                for (size_t j = 0; j < inputSize; j++)
                    weights[i][j] = distribution(gen);
                biases[i] = distribution(gen);
            }
        }

        Vector<float, 2> Forward(const Vector<float, 2> &input) override
        {
            try
            {
                layerInput = input;
                Vector<float, 2> unBiased = dot(input, transpose(weights));
                
                for (size_t i = 0; i < shape(unBiased)[0]; i++)
                    unBiased[i] += biases;

                return unBiased;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Dense::Forward:\n\t" + std::string(e.what()));
            }
        }

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta, const float &learningRate) override
        {
            try
            {
                Vector<float, 2> delta = dot(nextDelta, weights);
                Vector<float, 2> nextDeltaTranspose = transpose(nextDelta);

                weights -= dot(nextDeltaTranspose, layerInput) * learningRate / (float)shape(layerInput)[0];
                for (size_t i = 0; i < m_outputSize; i++)
                    biases[i] -= learningRate * mean(nextDeltaTranspose[i]);

                return delta;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Dense::Backward:\n\t" + std::string(e.what()));
            }
        }

    private:
        Vector<float, 2> layerInput;
    };

    class Activation : public BaseActivation<float>
    {
    public:
        Activation(const std::function<float(float)> &activation, const std::function<float(float)> &activationDerivative)
            : BaseActivation(activation, activationDerivative)
        {
        }

        Vector<float, 2> Forward(const Vector<float, 2> &input) override
        {
            try
            {
                layerInput = input;

                Vector<float, 2> output(shape(input)[0], shape(input)[1]);
                for (size_t i = 0; i < shape(input)[0]; i++)
                    for (size_t j = 0; j < shape(input)[1]; j++)
                        output[i][j] = m_activation(input[i][j]);

                return output;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Activation::Forward:\n\t" + std::string(e.what()));
            }
        }

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta, const float &learningRate) override
        {
            try
            {
                Vector<float, 2> result(shape(nextDelta)[0], shape(nextDelta)[1]);
                for (size_t i = 0; i < shape(nextDelta)[0]; i++)
                    for (size_t j = 0; j < shape(nextDelta)[1]; j++)
                        result[i][j] = m_activationDerivative(layerInput[i][j]) * nextDelta[i][j];

                return result;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Activation::Backward:\n\t" + std::string(e.what()));
            }
        }

    private:
        Vector<float, 2> layerInput;
    };

    class Sigmoid : public BaseActivation<float>
    {
    public:
        Sigmoid()
            : BaseActivation(
                  [](float x)
                  { return 1.0f / (1.0f + std::exp(-x)); },
                  [](float x)
                  { return x * (1.0f - x); })
        {
        }

        Vector<float, 2> Forward(const Vector<float, 2> &input) override
        {
            try {
                Vector<float, 2> output(shape(input)[0], shape(input)[1]);
                for (size_t i = 0; i < shape(input)[0]; i++)
                    for (size_t j = 0; j < shape(input)[1]; j++)
                        output[i][j] = m_activation(input[i][j]);

                layerOutput = output;
                return output;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Sigmoid::Forward:\n\t" + std::string(e.what()));
            }
        }

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta, const float &learningRate) override
        {
            try {
                Vector<float, 2> result(shape(nextDelta)[0], shape(nextDelta)[1]);
                for (size_t i = 0; i < shape(nextDelta)[0]; i++)
                    for (size_t j = 0; j < shape(nextDelta)[1]; j++)
                        result[i][j] = m_activationDerivative(layerOutput[i][j]) * nextDelta[i][j];

                return result;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Sigmoid::Backward:\n\t" + std::string(e.what()));
            }
        }

    private:
        Vector<float, 2> layerOutput;
    };

    class Softmax : public BaseActivation<float>
    {
    public:
        Softmax()
            : BaseActivation(
                [](Vector<float, 1> x) { 
                    float max = x[0];
                    for (size_t i = 1; i < shape(x)[0]; i++)
                        if (x[i] > max)
                            max = x[i];

                    float sum = 0.0f;
                    for (size_t i = 0; i < shape(x)[0]; i++)
                        sum += std::exp(x[i] - max);

                    Vector<float, 1> result(shape(x)[0]);
                    for (size_t i = 0; i < shape(x)[0]; i++)
                        result[i] = std::exp(x[i] - max) / sum;

                    return result;
                },
                [](Vector<float, 1> x) {
                    return x;
                }) {}

        Vector<float, 2> Forward(const Vector<float, 2> &input) override
        {
            try {
                Vector<float, 2> output(shape(input)[0], shape(input)[1]);
                for (size_t i = 0; i < shape(input)[0]; i++)
                    output[i] = m_activationVector(input[i]);

                return output;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Softmax::Forward:\n\t" + std::string(e.what()));
            }
        }

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta, const float &learningRate) override
        {
            return nextDelta;
        }
    };

    class SimpleRNN : public BaseActivation<float>
    {
    public:
        SimpleRNN(size_t size)
            : BaseActivation(
                [](const float &f){ return (2.0f / (1.0f + std::exp(-2.0f * f))) - 1.0f; }, 
                [](const float &f){ return 1 - f * f; }), m_size(size), m_dense(size, size)
        {
        }

        Vector<float, 2> Forward(const Vector<float, 2> &input) override
        {
            try {
                if (layerOutput.size() == 0)
                    layerOutput = Vector<float, 2>(shape(input)[0], shape(input)[1]);

                Vector<float, 2> input_ = input + m_dense.Forward(layerOutput);
                Vector<float, 2> output = map(input_, m_activation);

                layerOutput = output;
                return output;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("SimpleRNN::Forward:\n\t" + std::string(e.what()));
            }
        }

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta, const float &learningRate)
        {
            try {
                m_dense.Backward(nextDelta, learningRate);
                Vector<float, 2> delta(shape(nextDelta)[0], shape(nextDelta)[1]);

                for (size_t i = 0; i < shape(nextDelta)[0]; i++)
                    for (size_t j = 0; j < shape(nextDelta)[1]; j++)
                        delta[i][j] = m_activationDerivative(layerOutput[i][j]) * nextDelta[i][j];

                return delta;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("SimpleRNN::Backward:\n\t" + std::string(e.what()));
            }
        }

    private:
        Dense m_dense;
        size_t m_size;

        Vector<float, 2> layerOutput;
    };  

    class Layer
    {
    public:
        Layer(BaseNet<float> *net, BaseActivation<float> *activation)
            : m_net(net), m_activation(activation)
        {
        }

        Vector<float, 2> Forward(const Vector<float, 2> &input)
        {
            return m_activation->Forward(m_net->Forward(input));
        }

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta, const float &learningRate)
        {
            return m_net->Backward(m_activation->Backward(nextDelta, learningRate), learningRate);
        }

    private:
        BaseNet<float> *m_net;
        BaseActivation<float> *m_activation;
    };

    class Sequential
    {
    public:
        Sequential(const std::initializer_list<Layer> &layers)
        {
            for (auto layer : layers)
                m_layers.push_back(layer);
        }

        Vector<float, 2> Forward(const Vector<float, 2> &input)
        {
            Vector<float, 2> output = input;
            for (auto layer : m_layers)
                output = layer.Forward(output);

            return output;
        }

        void Backward(const Vector<float, 2> &output, const float &learningRate)
        {
            Vector<float, 2> nextDelta = output;
            for (int layerIndex = m_layers.size() - 1; layerIndex >= 0; layerIndex--)
                nextDelta = m_layers[layerIndex].Backward(nextDelta, learningRate);
        }

        float Train(const Vector<float, 2> &input, const Vector<float, 2> &output, const float &learningRate)
        {
            Vector<float, 2> outputVector = Forward(input);
            Vector<float, 2> outDelta = outputVector - output;
            
            Backward(outDelta, learningRate);

            float error = 0.0f;
            for (size_t i = 0; i < shape(outputVector)[0]; i++)
                for (size_t j = 0; j < shape(outputVector)[1]; j++)
                    error += -(output[i][j] * std::log(outputVector[i][j]) + (1.0f - output[i][j]) * std::log(1.0f - outputVector[i][j]));

            return error;
        }

        Vector<float, 1> Predict(const Vector<float, 1> &input)
        {
            Vector<float, 2> input_ = {input};
            Vector<float, 2> output = Forward(input_);

            return output[0];
        }

    private:
        std::vector<Layer> m_layers;
    };
};

#endif