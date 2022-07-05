#ifndef MODULE_H_
#define MODULE_H_

#include <string>

#include "Tensor.h"

namespace StaticNet
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
        Module(std::string name, Module<T> *parent = nullptr, size_t parameters = 0): name(name) {
            if (parent) {
                parent->children.push_back(this);
                parent->parameters += parameters;
                this->depth = parent->depth + 1;
                this->parameters = parameters;
            } 
        }
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
            static Tensor<T, Batch, InputSize...> mem;

            if (access == AccessType::Read)
            {
                return mem;
            }
            else if (access == AccessType::Write)
            {
                mem = input;
                return input;
            }

            return input;
        }

        std::vector<Module<T> *> children;
        std::string name = "Module";
        size_t parameters = 0;
        size_t depth = 0;
    };
}

template <typename T>
void print(const StaticNet::Module<T>& mod, size_t level = 0)
{
    std::string tab(level * 4, ' ');

    if (mod.children.size()) {
        std::cout << tab << mod.name << " [" << mod.parameters << "] [\n";
        for (auto child : mod.children)
            print(*child, level+1);
        std::cout << tab << "]\n";
    }
    else {
        std::cout << tab << mod.name << " [" << mod.parameters << "]\n";
    }
}

#endif