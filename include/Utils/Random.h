#ifndef RANDOM_H_
#define RANDOM_H_

#include <random>

namespace SingleNet {
    namespace Random {
        static std::random_device rd;
        static std::mt19937 mt(rd());

        static std::uniform_real_distribution<float> float_dist;
        static std::uniform_real_distribution<double> double_dist;
        static std::bernoulli_distribution bool_dist;

        template <class T>
        T rand();
    }
}

#endif