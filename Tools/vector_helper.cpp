#include "vector_helper.h"

float avx_dot_product(std::vector<float> a, std::vector<float> b) {
    if((a.size()) != 8 || (b.size() != 8)) return 0;

    __m256 m1 = { 0 }; memcpy(&m1, a.data(), 8);
    __m256 m2 = { 0 }; memcpy(&m2, b.data(), 8);

    __m256 res = _mm256_dp_ps(m1, m2, 0xff);
    return res[0] + res[4];
}