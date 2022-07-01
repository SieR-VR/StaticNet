#ifndef MODULE_H_
#define MODULE_H_

#include "Tensor.h"

namespace SingleNet
{
    enum class AccessType
    {
        Read = 0,
        Write = 1
    };

    template <class T>
    class Module
    {
    public:
        Module() {}
        Module(Module<T> *parent) { parent->children.push_back(this); }
        virtual ~Module() {}

        template <size_t Batch, size_t Input, size_t Output>
        Tensor<T, Batch, Output> forward(const Tensor<T, Batch, Input> &input)
        {
            return Tensor<T, Batch, Output>();
        };
        template <size_t Batch, size_t Input, size_t Output>
        Tensor<T, Batch, Input> backward(const Tensor<T, Batch, Output> &nextDelta, float learningRate)
        {
            return Tensor<T, Batch, Input>();
        };

        template <size_t Batch, size_t ...InputSize>
        Tensor<T, Batch, InputSize...> memory(AccessType access, const Tensor<T, Batch, InputSize...> &input = Tensor<T, Batch, InputSize...>())
        {
            static Tensor<T, Batch, InputSize...> layer_input;

            if (access == AccessType::Read)
            {
                return layer_input;
            }
            else if (access == AccessType::Write)
            {
                layer_input = input;
                return input;
            }

            return input;
        }

        std::vector<Module<T> *> children;
    };
}

#endif