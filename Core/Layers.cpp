#include <random>

#include "Core/Layers.h"

namespace SingleNet
{
    // ------------------------------------------------------------------------
    //  Dense
    // ------------------------------------------------------------------------

    Dense::Dense(size_t inputSize, size_t outputSize)
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

    Vector<float, 2> Dense::Forward(const Vector<float, 2> &input)
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

    Vector<float, 2> Dense::Backward(const Vector<float, 2> &nextDelta,
                                     const float &learningRate)
    {
        try
        {
            Vector<float, 2> delta = dot(nextDelta, weights);
            Vector<float, 2> nextDeltaTranspose = transpose(nextDelta);

            weights -= dot(nextDeltaTranspose, layerInput) * learningRate /
                       static_cast<float>(shape(layerInput)[0]);
            for (size_t i = 0; i < m_outputSize; i++)
                biases[i] -= learningRate * mean(nextDeltaTranspose[i]);

            return delta;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Dense::Backward:\n\t" + std::string(e.what()));
        }
    }

    // ------------------------------------------------------------------------
    //  Activation
    // ------------------------------------------------------------------------

    Activation::Activation(const std::function<float(float)> &activation,
                           const std::function<float(float)> &activationDerivative)
        : BaseActivation(), m_activation(activation), m_activationDerivative(activationDerivative) {}

    Vector<float, 2> Activation::Forward(const Vector<float, 2> &input)
    {
        try
        {
            layerInput = input;
            Vector<float, 2> output = map(input, m_activation);
            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Activation::Forward:\n\t" +
                                     std::string(e.what()));
        }
    }

    Vector<float, 2> Activation::Backward(const Vector<float, 2> &nextDelta,
                                          const float &learningRate)
    {
        try
        {
            Vector<float, 2> result = map(layerInput, m_activationDerivative);
            for (size_t i = 0; i < shape(nextDelta)[0]; i++)
                for (size_t j = 0; j < shape(nextDelta)[1]; j++)
                    result[i][j] *= nextDelta[i][j];

            return result;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Activation::Backward:\n\t" +
                                     std::string(e.what()));
        }
    }

    // ------------------------------------------------------------------------
    //  Sigmoid
    // ------------------------------------------------------------------------

    Sigmoid::Sigmoid() : BaseActivation(), m_activation(Defines::Sigmoid),
                         m_activationDerivative(Defines::SigmoidDerivative_) {}

    Vector<float, 2> Sigmoid::Forward(const Vector<float, 2> &input)
    {
        try
        {
            Vector<float, 2> output = map(input, m_activation);
            layerOutput = output;
            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Sigmoid::Forward:\n\t" + std::string(e.what()));
        }
    }

    Vector<float, 2> Sigmoid::Backward(const Vector<float, 2> &nextDelta,
                                       const float &learningRate)
    {
        try
        {
            Vector<float, 2> result = map(layerOutput, m_activationDerivative);
            for (size_t i = 0; i < shape(nextDelta)[0]; i++)
                for (size_t j = 0; j < shape(nextDelta)[1]; j++)
                    result[i][j] *= nextDelta[i][j];

            return result;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Sigmoid::Backward:\n\t" + std::string(e.what()));
        }
    }

    // ------------------------------------------------------------------------
    //  Softmax
    // ------------------------------------------------------------------------

    Softmax::Softmax() : BaseActivation(), m_activation(Defines::Softmax) {}

    Vector<float, 2> Softmax::Forward(const Vector<float, 2> &input)
    {
        try
        {
            Vector<float, 2> output(shape(input)[0], shape(input)[1]);
            for (size_t i = 0; i < shape(input)[0]; i++)
                output[i] = m_activation(input[i]);

            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Softmax::Forward:\n\t" + std::string(e.what()));
        }
    }

    Vector<float, 2> Softmax::Backward(const Vector<float, 2> &nextDelta,
                                       const float &learningRate)
    {
        return nextDelta;
    }

    // ------------------------------------------------------------------------
    //  SimpleRNN
    // ------------------------------------------------------------------------

    SimpleRNN::SimpleRNN(size_t size)
        : m_activation(Defines::Tanh),
          m_activationDerivative(Defines::TanhDerivative_),
          m_size(size),
          m_dense(size, size) {}

    Vector<float, 2> SimpleRNN::Forward(const Vector<float, 2> &input)
    {
        try
        {
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

    Vector<float, 2> SimpleRNN::Backward(const Vector<float, 2> &nextDelta,
                                         const float &learningRate)
    {
        try
        {
            m_dense.Backward(nextDelta, learningRate);
            Vector<float, 2> delta = map(layerOutput, m_activationDerivative);

            for (size_t i = 0; i < shape(nextDelta)[0]; i++)
                for (size_t j = 0; j < shape(nextDelta)[1]; j++)
                    delta[i][j] *= nextDelta[i][j];

            return delta;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("SimpleRNN::Backward:\n\t" + std::string(e.what()));
        }
    }

    // ------------------------------------------------------------------------
    //  Layer
    // ------------------------------------------------------------------------

    Layer::Layer(BaseNet<float> *net, BaseActivation<float> *activation)
        : m_net(net), m_activation(activation) {}

    Vector<float, 2> Layer::Forward(const Vector<float, 2> &input)
    {
        try
        {
            return m_activation->Forward(m_net->Forward(input));
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Layer::Forward:\n\t" + std::string(e.what()));
        }
    }

    Vector<float, 2> Layer::Backward(const Vector<float, 2> &nextDelta,
                                     const float &learningRate)
    {
        try
        {
            return m_net->Backward(m_activation->Backward(nextDelta, learningRate), learningRate);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Layer::Backward:\n\t" + std::string(e.what()));
        }
    }

    // ------------------------------------------------------------------------
    //  Sequential
    // ------------------------------------------------------------------------

    Sequential::Sequential(const std::initializer_list<Layer> &layers,
                           const std::function<float(Vector<float, 1>, Vector<float, 1>)> &lossFunction)
        : m_layers(layers), m_lossFunction(lossFunction) {}

    Vector<float, 2> Sequential::Forward(const Vector<float, 2> &input)
    {
        try
        {
            Vector<float, 2> output = input;
            for (auto &layer : m_layers)
                output = layer.Forward(output);

            return output;
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Sequential::Forward:\n\t" + std::string(e.what()));
        }
    }

    void Sequential::Backward(const Vector<float, 2> &nextDelta,
                              const float &learningRate)
    {
        try
        {
            Vector<float, 2> delta = nextDelta;
            for (int layerIndex = m_layers.size() - 1; layerIndex >= 0; layerIndex--)
                delta = m_layers[layerIndex].Backward(nextDelta, learningRate);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Sequential::Backward:\n\t" + std::string(e.what()));
        }
    }

    float Sequential::Loss(const Vector<float, 2> &output,
                           const Vector<float, 2> &expected)
    {
        try
        {
            float loss = 0;
            for (size_t i = 0; i < shape(output)[0]; i++)
                loss += m_lossFunction(output[i], expected[i]);

            return loss / static_cast<float>(shape(output)[0]);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Sequential::Loss:\n\t" + std::string(e.what()));
        }
    }

    float Sequential::Train(const Vector<float, 2> &input,
                            const Vector<float, 2> &expected,
                            const float &learningRate)
    {
        try
        {
            Vector<float, 2> outputVector = Forward(input);
            Vector<float, 2> outDelta = outputVector - expected;

            Backward(outDelta, learningRate);
            return Loss(expected, outputVector);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Sequential::Train:\n\t" + std::string(e.what()));
        }
    }

    Vector<float, 1> Sequential::Predict(const Vector<float, 1> &input)
    {
        try
        {
            return Forward({ input })[0];
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Sequential::Predict:\n\t" + std::string(e.what()));
        }
    }
}