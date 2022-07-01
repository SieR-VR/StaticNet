#ifndef RELU_H_
#define RELU_H_

#include "Module.h"
#include "Defines.h"

namespace SingleNet
{
    template <typename... T>
    class ReLU
    {
        ReLU() = delete;
    };

    template <class T, size_t... Input>
    class ReLU<Tensor<T, Input...>> : public Module<T>
    {
        ReLU() : Module<T>("ReLU") {}
        ~ReLU() {}

        template <size_t Batch>
        Tensor<T, Input...> forward(const Tensor<T, Input...> &input) {
            memory(AccessType::Write, input);
            return input.map([](T value) {
                return value > 0 ? value : 0;
            });
        }

        template <size_t Batch>
        Tensor<T, Input...> backward(const Tensor<T, Input...> &delta) {
            return conv(memory(AccessType::Read, Tensor<T, Input...>()).map([](T value) {
                return value > 0 ? 1 : 0;
            }), delta);
        }
    };
}

#endif