#include "Utils/Random.h"

namespace SingleNet
{
    namespace Random
    {
        template <>
        float rand() {
            return float_dist(mt);
        }

        template <>
        double rand() {
            return double_dist(mt);
        }

        template <>
        bool rand() {
            return bool_dist(mt);
        }
    }
}