#ifndef RANDOM_H_
#define RANDOM_H_

#include <random>

namespace StaticNet {
    namespace Random {
        static std::random_device rd;
        static std::mt19937 mt(rd());

        template <class T>
        T rand();
    }
}

#endif