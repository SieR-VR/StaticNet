#ifndef VECTORCUDA_H_
#define VECTORCUDA_H_

#include "Vector.h"

namespace SingleNet {
    template<size_t N>
    class VectorCUDA {
        static_assert(N > 0, "VectorCUDA: N must be greater than 0");

    public:
        VectorCUDA();
        VectorCUDA(const Vector<float, N> &v);
        VectorCUDA(const VectorCUDA<N> &v);
        VectorCUDA(VectorCUDA<N> &v);
        ~VectorCUDA();

        VectorCUDA<N>& operator=(const Vector<float, N> &v);
        VectorCUDA<N>& operator=(const VectorCUDA<N> &v);
        VectorCUDA<N>& operator=(VectorCUDA<N> &v);

        VectorCUDA<N> operator+(const VectorCUDA<N> &v) const;
        VectorCUDA<N> operator-(const VectorCUDA<N> &v) const;
        VectorCUDA<N>& operator+=(const VectorCUDA<N> &v);
        VectorCUDA<N>& operator-=(const VectorCUDA<N> &v);

        VectorCUDA<N> operator*(const float &s) const;
        VectorCUDA<N> operator/(const float &s) const;
        VectorCUDA<N>& operator*=(const float &s);
        VectorCUDA<N>& operator/=(const float &s);

        VectorCUDA<N> map(float (*func)(float)) const;
        VectorCUDA<N> copy() const;

        void *m_pDeviceData = nullptr;
        Vector<size_t, 1> m_shape = Vector<size_t, 1>();
    };

    VectorCUDA<2> operator+(const VectorCUDA<2> &v1, const VectorCUDA<1> &v2);
    VectorCUDA<2> operator-(const VectorCUDA<2> &v1, const VectorCUDA<1> &v2);
    VectorCUDA<2> &operator+=(VectorCUDA<2> &v1, const VectorCUDA<1> &v2);
    VectorCUDA<2> &operator-=(VectorCUDA<2> &v1, const VectorCUDA<1> &v2);

    VectorCUDA<1> mean(const VectorCUDA<2> &v);

    VectorCUDA<2> transpose(const VectorCUDA<2> &v);
    VectorCUDA<2> dot(const VectorCUDA<2> &v1, const VectorCUDA<2> &v2);

    template <size_t N>
    VectorCUDA<N> times(const VectorCUDA<N> &v1, const VectorCUDA<N> &v2);

    template <size_t N>
    Vector<float, N> to_cpu(const VectorCUDA<N> &v);
}

#endif