#include <random>

#include "Core/Defines/Defines.cuh"
#include "Core/Layers.h"

namespace SingleNet
{
    // ------------------------------------------------------------------------
    //  CUDA
    // ------------------------------------------------------------------------

    DenseCUDA::DenseCUDA(size_t inputSize, size_t outputSize)
        : BaseNetCUDA(inputSize, outputSize)
    {
        Vector<float, 2> weights_cpu(outputSize, inputSize);
        Vector<float, 1> biases_cpu(outputSize);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

        for (size_t i = 0; i < outputSize; i++)
        {
            for (size_t j = 0; j < inputSize; j++)
                weights_cpu[i][j] = distribution(gen);
            biases_cpu[i] = distribution(gen);
        }

        weights = weights_cpu;
        biases = biases_cpu;
    }

    VectorCUDA<2> DenseCUDA::Forward(const VectorCUDA<2> &input)
    {
        try
        {
            layerInput = input;
            VectorCUDA<2> output = dot(layerInput, transpose(weights)) + biases;

            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("DenseCUDA::Forward:\n\t" + std::string(e.what()));
        }
    }

    VectorCUDA<2> DenseCUDA::Backward(const VectorCUDA<2> &nextDelta,
                                      const float &learningRate)
    {
        try
        {
            VectorCUDA<2> delta = dot(nextDelta, weights);
            VectorCUDA<2> nextDeltaTranspose = transpose(nextDelta);

            weights -= dot(nextDeltaTranspose, layerInput) * learningRate /
                       static_cast<float>(layerInput.m_shape[0]);
            biases -= mean(nextDeltaTranspose) * learningRate;

            return delta;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("DenseCUDA::Backward:\n\t" + std::string(e.what()));
        }
    }

    ActivationCUDA::ActivationCUDA(float (*activation)(float),
                                   float (*activationDerivative)(float))
        : BaseActivationCUDA(), m_activation(activation), m_activationDerivative(activationDerivative) {}

    VectorCUDA<2> ActivationCUDA::Forward(const VectorCUDA<2> &input)
    {
        try
        {
            layerInput = input;
            VectorCUDA<2> output = layerInput.map(m_activation);

            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("ActivationCUDA::Forward:\n\t" + std::string(e.what()));
        }
    }

    VectorCUDA<2> ActivationCUDA::Backward(const VectorCUDA<2> &nextDelta,
                                           const float &learningRate)
    {
        try
        {
            VectorCUDA<2> delta = times(layerInput.map(m_activationDerivative), nextDelta);
            return delta;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("ActivationCUDA::Backward:\n\t" + std::string(e.what()));
        }
    }

    SigmoidCUDA::SigmoidCUDA()
        : m_activation(Defines::SigmoidCUDA), m_activationDerivative(Defines::SigmoidDerivativeCUDA_) {}

    VectorCUDA<2> SigmoidCUDA::Forward(const VectorCUDA<2> &input)
    {
        try
        {
            VectorCUDA<2> output = input.map(m_activation);
            VectorCUDA<2> outputCopy = output.copy();
            layerOutput = outputCopy;
            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SigmoidCUDA::Forward:\n\t" + std::string(e.what()));
        }
    }

    VectorCUDA<2> SigmoidCUDA::Backward(const VectorCUDA<2> &nextDelta,
                                        const float &learningRate)
    {
        try
        {
            VectorCUDA<2> delta = times(layerOutput.map(m_activationDerivative), nextDelta);
            return delta;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SigmoidCUDA::Backward:\n\t" + std::string(e.what()));
        }
    }

    SoftmaxCUDA::SoftmaxCUDA()
        : m_activation(Defines::Softmax) {}

    VectorCUDA<2> SoftmaxCUDA::Forward(const VectorCUDA<2> &input)
    {
        try
        {
            Vector<float, 2> output = to_cpu(input);
            for (size_t i = 0; i < shape(output)[0]; i++)
                output[i] = m_activation(output[i]);

            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SoftmaxCUDA::Forward:\n\t" + std::string(e.what()));
        }
    }

    VectorCUDA<2> SoftmaxCUDA::Backward(const VectorCUDA<2> &nextDelta,
                                        const float &learningRate)
    {
        try
        {
            return nextDelta;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SoftmaxCUDA::Backward:\n\t" + std::string(e.what()));
        }
    }

    SimpleRNNCUDA::SimpleRNNCUDA(size_t size)
        : m_activation(Defines::TanhCUDA), 
          m_activationDerivative(Defines::TanhDerivativeCUDA_),
          m_size(size),
          m_dense(size, size) {}

    VectorCUDA<2> SimpleRNNCUDA::Forward(const VectorCUDA<2> &input)
    {
        try
        {
            if (layerOutput.m_shape[0] == 0)
                layerOutput = Vector<float, 2>(input.m_shape[0], input.m_shape[1]);

            VectorCUDA<2> input_ = input + m_dense.Forward(layerOutput);
            VectorCUDA<2> output = input_.map(m_activation);

            VectorCUDA<2> outputCopy = output.copy();
            layerOutput = outputCopy;
            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SimpleRNNCUDA::Forward:\n\t" + std::string(e.what()));
        }
    }

    VectorCUDA<2> SimpleRNNCUDA::Backward(const VectorCUDA<2> &nextDelta,
                                          const float &learningRate)
    {
        try
        {
            VectorCUDA<2> delta = times(layerOutput.map(m_activationDerivative), nextDelta);
            m_dense.Backward(delta, learningRate);
            return delta;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SimpleRNNCUDA::Backward:\n\t" + std::string(e.what()));
        }
    }

    LayerCUDA::LayerCUDA(BaseNetCUDA* net, BaseActivationCUDA *activation)
        : m_net(net), m_activation(activation) {}
    
    VectorCUDA<2> LayerCUDA::Forward(const VectorCUDA<2> &input)
    {
        try
        {
            return m_activation->Forward(m_net->Forward(input));
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("LayerCUDA::Forward:\n\t" + std::string(e.what()));
        }
    }

    VectorCUDA<2> LayerCUDA::Backward(const VectorCUDA<2> &nextDelta,
                                      const float &learningRate)
    {
        try
        {
            return m_net->Backward(m_activation->Backward(nextDelta, learningRate), learningRate);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("LayerCUDA::Backward:\n\t" + std::string(e.what()));
        }
    }

    SequentialCUDA::SequentialCUDA(const std::initializer_list<LayerCUDA> &layers,
                       const std::function<float(Vector<float, 1>, Vector<float, 1>)> &lossFunction)
        : m_layers(layers), m_lossFunction(lossFunction) {}

    VectorCUDA<2> SequentialCUDA::Forward(const VectorCUDA<2> &input)
    {
        try
        {
            VectorCUDA<2> output = input;
            for (auto &layer : m_layers)
                output = layer.Forward(output);
            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SequentialCUDA::Forward:\n\t" + std::string(e.what()));
        }
    }

    void SequentialCUDA::Backward(const VectorCUDA<2> &nextDelta,
                                           const float &learningRate)
    {
        try
        {
            VectorCUDA<2> delta = nextDelta;
            for (int layerIndex = m_layers.size() - 1; layerIndex >= 0; layerIndex--)
                delta = m_layers[layerIndex].Backward(delta, learningRate);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SequentialCUDA::Backward:\n\t" + std::string(e.what()));
        }
    }

    float SequentialCUDA::Loss(const Vector<float, 2> &output,
                               const Vector<float, 2> &expected)
    {
        try
        {
            float loss = 0.0f;
            for (size_t i = 0; i < shape(output)[0]; i++)
                loss += m_lossFunction(output[i], expected[i]);
        
            return loss / static_cast<float>(shape(output)[0]);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SequentialCUDA::Loss:\n\t" + std::string(e.what()));
        }
    }

    float SequentialCUDA::Train(const VectorCUDA<2> &input,
                                const VectorCUDA<2> &expected,
                                const float &learningRate)
    {
        try
        {
            VectorCUDA<2> outputVector = Forward(input);
            VectorCUDA<2> delta = outputVector - expected;

            Backward(delta, learningRate);
            return Loss(to_cpu(expected), to_cpu(outputVector));
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SequentialCUDA::Train:\n\t" + std::string(e.what()));
        }
    }
    
    Vector<float, 1> SequentialCUDA::Predict(const Vector<float, 1> &input)
    {
        try
        {
            VectorCUDA<2> output = Forward(Vector<float, 2>({ input }));
            return to_cpu(output)[0];
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SequentialCUDA::Predict:\n\t" + std::string(e.what()));
        }
    }
}