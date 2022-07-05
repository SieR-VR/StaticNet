#ifndef RELU_H_
#define RELU_H_

#include "Module.h"
#include "Defines.h"

namespace StaticNet
{
    template <typename... T>
    class ReLU
    {
        ReLU() = delete;
    };

    template <class T, size_t... Input>
    class ReLU<Tensor<T, Input...>> : public Module<T>
    {
    public:
        ReLU(Module<T> *parent) : Module<T>("ReLU", parent) {}
        ~ReLU() {}

        template <size_t Batch>
        Tensor<T, Batch, Input...> forward(const Tensor<T, Batch, Input...> &input) {
            this->memory(AccessType::Write, input);
            return input.map(relu);
        }

        template <size_t Batch>
        Tensor<T, Batch, Input...> backward(const Tensor<T, Batch, Input...> &delta, float learningRate) {
            Tensor<T, Batch, Input...> input = this->memory(AccessType::Read, Tensor<T, Batch, Input...>());
            return hadamard(input.map(relu_grad), delta);
        }

    private:
        std::function<T(T)> relu = [](T value) {
            return value > 0 ? value : T();
        };
        
        std::function<T(T)> relu_grad = [](T value) {
            return value > 0 ? 1 : 0;
        };
    };
}

#endif