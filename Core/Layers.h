/* Copyright 2021- SieR-VR */

#ifndef FLOAT_LAYERS_H_
#define FLOAT_LAYERS_H_

#include <functional>
#include <vector>

#include "Core/Vector/Vector.h"
#include "Core/Vector/VectorCUDA.h"
#include "Core/Defines/Defines.cuh"

namespace SingleNet
{
    // ------------------------------------------------------------------------
    // Base class for layers
    // ------------------------------------------------------------------------

    template <typename T>
    class BaseNet
    {

    public:
        BaseNet(size_t inputSize, size_t outputSize)
            : m_inputSize(inputSize),
              m_outputSize(outputSize),
              weights(outputSize, inputSize, T()),
              biases(outputSize, T()) {}

        virtual ~BaseNet() {}

        virtual Vector<T, 2> Forward(const Vector<T, 2> &input) = 0;
        virtual Vector<T, 2> Backward(const Vector<T, 2> &nextDelta,
                                      const T &learningRate) = 0;

    protected:
        size_t m_inputSize;
        size_t m_outputSize;

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

    // CUDA version

    class BaseNetCUDA
    {
    public:
        BaseNetCUDA(size_t inputSize, size_t outputSize)
            : m_inputSize(inputSize),
              m_outputSize(outputSize),
              weights(Vector<float, 2>(outputSize, inputSize, 0.0f)),
              biases(Vector<float, 1>(outputSize, 0.0f)) {}

        virtual VectorCUDA<2> Forward(const VectorCUDA<2> &input) = 0;
        virtual VectorCUDA<2> Backward(const VectorCUDA<2> &nextDelta,
                                       const float &learningRate) = 0;

    protected:
        size_t m_inputSize;
        size_t m_outputSize;

        VectorCUDA<2> weights;
        VectorCUDA<1> biases;
    };

    class BaseActivationCUDA
    {
    public:
        BaseActivationCUDA() {}
        virtual ~BaseActivationCUDA() {}
        virtual VectorCUDA<2> Forward(const VectorCUDA<2> &input) = 0;
        virtual VectorCUDA<2> Backward(const VectorCUDA<2> &nextDelta,
                                       const float &learningRate) = 0;
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

    class DenseCUDA : public BaseNetCUDA
    {
    public:
        DenseCUDA(size_t inputSize, size_t outputSize);

        VectorCUDA<2> Forward(const VectorCUDA<2> &input) override;
        VectorCUDA<2> Backward(const VectorCUDA<2> &nextDelta,
                               const float &learningRate) override;
    private:
        VectorCUDA<2> layerInput;
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

    class ActivationCUDA : public BaseActivationCUDA
    {
    public:
        ActivationCUDA(float (*activation)(float),
                       float (*activationDerivative)(float));

        VectorCUDA<2> Forward(const VectorCUDA<2> &input) override;
        VectorCUDA<2> Backward(const VectorCUDA<2> &nextDelta,
                               const float &learningRate) override;

    private:
        float (*m_activation)(float);
        float (*m_activationDerivative)(float);

        VectorCUDA<2> layerInput;
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

    class SigmoidCUDA : public BaseActivationCUDA
    {
    public:
        SigmoidCUDA();

        VectorCUDA<2> Forward(const VectorCUDA<2> &input) override;
        VectorCUDA<2> Backward(const VectorCUDA<2> &nextDelta,
                               const float &learningRate) override;

    private:
        float (*m_activation)(float);
        float (*m_activationDerivative)(float);

        VectorCUDA<2> layerOutput;
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

    class SoftmaxCUDA : public BaseActivationCUDA
    {
    public:
        SoftmaxCUDA();

        VectorCUDA<2> Forward(const VectorCUDA<2> &input) override;
        VectorCUDA<2> Backward(const VectorCUDA<2> &nextDelta,
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

    class SimpleRNNCUDA : public BaseActivationCUDA
    {
    public:
        explicit SimpleRNNCUDA(size_t size);

        VectorCUDA<2> Forward(const VectorCUDA<2> &input) override;
        VectorCUDA<2> Backward(const VectorCUDA<2> &nextDelta,
                               const float &learningRate) override;
        
    private:
        float (*m_activation)(float);
        float (*m_activationDerivative)(float);

        size_t m_size;
        DenseCUDA m_dense;

        VectorCUDA<2> layerOutput;
    };

    class Layer
    {
    public:
        Layer(BaseNet<float> *net, BaseActivation<float> *activation);
        Vector<float, 2> Forward(const Vector<float, 2> &input);
        Vector<float, 2> Backward(const Vector<float, 2> &nextDelta,
                                  const float &learningRate);

    private:
        BaseNet<float> *m_net;
        BaseActivation<float> *m_activation;
    };

    class LayerCUDA
    {
    public:
        LayerCUDA(BaseNetCUDA *net, BaseActivationCUDA *activation);
        VectorCUDA<2> Forward(const VectorCUDA<2> &input);
        VectorCUDA<2> Backward(const VectorCUDA<2> &nextDelta,
                               const float &learningRate);
    
    private:
        BaseNetCUDA *m_net;
        BaseActivationCUDA *m_activation;
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

        std::vector<Layer> m_layers;
        std::function<float(Vector<float, 1>, Vector<float, 1>)> m_lossFunction;
    };

    class SequentialCUDA
    {
    public:
        SequentialCUDA(const std::initializer_list<LayerCUDA> &layers,
                       const std::function<float(Vector<float, 1>, Vector<float, 1>)> &lossFunction);
        float Train(const VectorCUDA<2> &input, const VectorCUDA<2> &output,
                    const float &learningRate);
        Vector<float, 1> Predict(const Vector<float, 1> &input);

    private:
        VectorCUDA<2> Forward(const VectorCUDA<2> &input);
        void Backward(const VectorCUDA<2> &output, const float &learningRate);
        float Loss(const Vector<float, 2> &output, const Vector<float, 2> &expected);

        std::vector<LayerCUDA> m_layers;
        std::function<float(Vector<float, 1>, Vector<float, 1>)> m_lossFunction;
    };
}; // namespace SingleNet

#endif // FLOAT_LAYERS_H_
