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

        virtual ~BaseActivation()
        {
        }

        virtual Vector<T, 2> Forward(const Vector<T, 2> &input) = 0;
        virtual Vector<T, 2> Backward(const Vector<T, 2> &nextDelta) = 0;

    protected:
        std::function<T(T)> m_activation;
        std::function<T(T)> m_activationDerivative;
    };

    class Net : public BaseNet<float>
    {
    public:
        Net(size_t inputSize, size_t outputSize)
            : BaseNet(inputSize, outputSize)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

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
                    for (size_t j = 0; j < shape(unBiased)[1]; j++)
                        unBiased[i][j] += biases[i];

                return unBiased;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Net::Forward:\n\t" + std::string(e.what()));
            }
        }

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta, const float &learningRate) override
        {
            try
            {
                Vector<float, 2> delta = dot(nextDelta, weights);
                Vector<float, 2> nextDeltaTranspose = transpose(nextDelta);

                weights -= dot(nextDeltaTranspose, layerInput) * learningRate;
                for (size_t i = 0; i < m_outputSize; i++)
                    biases[i] -= learningRate * mean(nextDeltaTranspose[i]);

                return delta;
            }
            catch (const std::exception &e)
            {
                throw std::runtime_error("Net::Backward:\n\t" + std::string(e.what()));
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

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta) override
        {
            try
            {
                Vector<float, 2> result(shape(nextDelta)[0], shape(nextDelta)[1]);
                for (size_t i = 0; i < shape(nextDelta)[0]; i++)
                    for (size_t j = 0; j < shape(nextDelta)[1]; j++)
                        result[i][j] += m_activationDerivative(layerInput[i][j]) * nextDelta[i][j];

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

        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta) override
        {
            try {
                Vector<float, 2> result(shape(nextDelta)[0], shape(nextDelta)[1]);
                for (size_t i = 0; i < shape(nextDelta)[0]; i++)
                    for (size_t j = 0; j < shape(nextDelta)[1]; j++)
                        result[i][j] += m_activationDerivative(layerOutput[i][j]) * nextDelta[i][j];

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
            return m_net->Backward(m_activation->Backward(nextDelta), learningRate);
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