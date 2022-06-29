/* Copyright 2021- SieR-VR */

#ifndef FLOAT_LAYERS_H_
#define FLOAT_LAYERS_H_

#include <functional>
#include <vector>

#include "Core/Vector/Vector.h"
#include "Core/Defines/Defines.h"

namespace SingleNet
{
    // ------------------------------------------------------------------------
    // Base class for layers
    // ------------------------------------------------------------------------

    template <typename T>
    class BaseNet
    {

    public:
        BaseNet(int inputSize, int outputSize)
            : m_inputSize(inputSize),
              m_outputSize(outputSize),
              weights(outputSize, inputSize, T()),
              biases(outputSize, T()) {}

        virtual ~BaseNet() {}

        virtual Vector<T, 2> Forward(const Vector<T, 2> &input) = 0;
        virtual Vector<T, 2> Backward(const Vector<T, 2> &nextDelta,
                                      const T &learningRate) = 0;
        
        std::vector<char> Serialize() {
            std::vector<char> serialized;
            for (int i = 0; i < sizeof(int); i++)
                serialized.push_back(((char*)(&m_inputSize))[i]);
            for (int i = 0; i < sizeof(int); i++)
                serialized.push_back(((char*)(&m_outputSize))[i]);

            for (int i = 0; i < m_outputSize; i++)
                for (int j = 0; j < m_inputSize; j++)
                    for (int k = 0; k < sizeof(T); k++)
                        serialized.push_back(((char*)(&weights[i][j]))[k]);

            for (int i = 0; i < m_outputSize; i++)
                for (int j = 0; j < sizeof(T); j++)
                    serialized.push_back(((char*)(&biases[i]))[j]);

            return serialized;
        }

        void Deserialize(const std::vector<char> &serialized) {
            m_inputSize = serialized[0] << 24 | serialized[1] << 16 | serialized[2] << 8 | serialized[3];
            m_outputSize = serialized[4] << 24 | serialized[5] << 16 | serialized[6] << 8 | serialized[7];

            int index = 8;

            weights = Vector<T, 2>(m_outputSize, m_inputSize, T());
            for (int i = 0; i < m_outputSize; i++)
                for (int j = 0; j < m_inputSize; j++) {
                    weights[i][j] = serialized[index] << 24 |
                                    serialized[index + 1] << 16 |
                                    serialized[index + 2] << 8 |
                                    serialized[index + 3];
                    index += 4;
                }

            biases = Vector<T, 1>(m_outputSize, T());
            for (int i = 0; i < m_outputSize; i++) { 
                biases[i] = serialized[index] << 24 |
                            serialized[index + 1] << 16 |
                            serialized[index + 2] << 8 |
                            serialized[index + 3];
                index += 4;
            }
        }

    protected:
        int m_inputSize;
        int m_outputSize;

        Vector<T, 2> weights;
        Vector<T, 1> biases;
    };

    template <typename T>
    class BaseActivation
    {

    public:
        BaseActivation() {}
        virtual ~BaseActivation() {}
        virtual Vector<T, 2> Forward(const Vector<T, 2> &input) = 0;
        virtual Vector<T, 2> Backward(const Vector<T, 2> &nextDelta,
                                      const T &learningRate) = 0;
    };

    // ------------------------------------------------------------------------
    // implementation
    // ------------------------------------------------------------------------

    class Dense : public BaseNet<float>
    {
    public:
        Dense(size_t inputSize, size_t outputSize);
        Vector<float, 2> Forward(const Vector<float, 2> &input) override;
        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta,
                                  const float &learningRate) override;

    private:
        Vector<float, 2> layerInput;
    };

    class Activation : public BaseActivation<float>
    {
    public:
        Activation(const std::function<float(float)> &activation,
                   const std::function<float(float)> &activationDerivative);

        Vector<float, 2> Forward(const Vector<float, 2> &input) override;
        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta,
                                  const float &learningRate) override;

    private:
        std::function<float(float)> m_activation;
        std::function<float(float)> m_activationDerivative;

        Vector<float, 2> layerInput;
    };

    class Sigmoid : public BaseActivation<float>
    {
    public:
        Sigmoid();
        Vector<float, 2> Forward(const Vector<float, 2> &input) override;
        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta,
                                  const float &learningRate) override;

    private:
        std::function<float(float)> m_activation;
        std::function<float(float)> m_activationDerivative;

        Vector<float, 2> layerOutput;
    };

    class Softmax : public BaseActivation<float>
    {
    public:
        Softmax();
        Vector<float, 2> Forward(const Vector<float, 2> &input) override;
        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta,
                                  const float &learningRate) override;

    private:
        std::function<Vector<float, 1>(Vector<float, 1>)> m_activation;
    };

    class SimpleRNN : public BaseActivation<float>
    {
    public:
        explicit SimpleRNN(size_t size);
        Vector<float, 2> Forward(const Vector<float, 2> &input) override;
        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta,
                                  const float &learningRate);

    private:
        std::function<float(float)> m_activation;
        std::function<float(float)> m_activationDerivative;

        size_t m_size;
        Dense m_dense;

        Vector<float, 2> layerOutput;
    };

    class Layer
    {
    public:
        Layer(BaseNet<float> *net, BaseActivation<float> *activation);
        Vector<float, 2> Forward(const Vector<float, 2> &input);
        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta,
                                  const float &learningRate);

    public:
        BaseNet<float> *m_net;
        BaseActivation<float> *m_activation;
    };

    class Sequential
    {
    public:
        Sequential(const std::initializer_list<Layer> &layers,
                   const std::function<float(Vector<float, 1>, Vector<float, 1>)> &lossFunction);
        float Train(const Vector<float, 2> &input, const Vector<float, 2> &output,
                    const float &learningRate);
        Vector<float, 1> Predict(const Vector<float, 1> &input);

    private:
        Vector<float, 2> Forward(const Vector<float, 2> &input);
        void Backward(const Vector<float, 2> &output, const float &learningRate);
        float Loss(const Vector<float, 2> &output, const Vector<float, 2> &expected);

    public:
        std::vector<Layer> m_layers;
        std::function<float(Vector<float, 1>, Vector<float, 1>)> m_lossFunction;
    };
}; // namespace SingleNet

#endif // FLOAT_LAYERS_H_
