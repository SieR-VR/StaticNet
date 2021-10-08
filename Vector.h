#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <ostream>
#include <stdexcept>

namespace SingleNet
{
    template <typename T, size_t N>
    class Vector : public std::vector<Vector<T, N - 1>>
    {
        static_assert(N > 0, "Vector must have at least one dimension");
        typedef Vector<T, N - 1> SubVector;

    public:
        template <typename... Args>
        Vector(size_t n = 0, Args... args) : std::vector<SubVector>(n, SubVector(args...)) {}
    };

    template <typename T>
    class Vector<T, 1> : public std::vector<T>
    {
    public:
        Vector(size_t n = 0, const T &value = T()) : std::vector<T>(n, value) {}
    };

    // Vector operations

    template <typename T, size_t N>
    Vector<size_t, 1> shape(const Vector<T, N> &v)
    {
        if constexpr (N == 1)
            return Vector<size_t, 1>(1, v.size());
        else
        {
            Vector<size_t, 1> _shape = shape(v[0]);
            _shape.push_back(v.size());
            return _shape;
        }
    }

    template <typename T>
    bool operator==(const Vector<T, 1> &v1, const Vector<T, 1> &v2)
    {
        if (v1.size() != v2.size())
            return false;
        for (size_t i = 0; i < v1.size(); i++)
            if (v1[i] != v2[i])
                return false;
        return true;
    }

    template <typename T, size_t N>
    bool operator==(const Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        if (v1.shape() != v2.shape())
            return false;
        for (size_t i = 0; i < v1.size(); i++)
            if (!(v1[i] == v2[i]))
                return false;
        return true;
    }

    template <typename T, size_t N>
    bool operator!=(const Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        return !(v1 == v2);
    }

    template <typename T, size_t N>
    Vector<T, N> operator+(const Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::invalid_argument("[Core] SingleNet::operator+(): vectors must have the same shape");

        Vector<T, N> v = v1;
        for (size_t i = 0; i < v1.size(); ++i)
            v[i] = v1[i] + v2[i];
        return v;
    }

    template <typename T, size_t N>
    Vector<T, N> operator-(const Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::invalid_argument("[Core] SingleNet::operator-(): vectors must have the same shape");

        Vector<T, N> v = v1;
        for (size_t i = 0; i < v1.size(); ++i)
            v[i] = v1[i] - v2[i];
        return v;
    }

    template <typename T, size_t N>
    Vector<T, N> &operator+=(Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::invalid_argument("[Core] SingleNet::operator+=(): vectors must have the same shape");

        for (size_t i = 0; i < v1.size(); ++i)
            v1[i] += v2[i];
        return v1;
    }

    template <typename T, size_t N>
    Vector<T, N> &operator-=(Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::invalid_argument("[Core] SingleNet::operator-=(): vectors must have the same shape");

        for (size_t i = 0; i < v1.size(); ++i)
            v1[i] -= v2[i];
        return v1;
    }

    template <typename T>
    Vector<T, 2> transpose(const Vector<T, 2> &v)
    {
        Vector<T, 2> res(shape(v)[1], shape(v)[0]);

        for (size_t i = 0; i < shape(v)[0]; ++i)
            for (size_t j = 0; j < shape(v)[1]; ++j)
                res[j][i] = v[i][j];

        return res;
    }

    template <typename T>
    T dot(const Vector<T, 1> &v1, const Vector<T, 1> &v2)
    {
        if (v1.size() != v2.size())
            throw std::invalid_argument("[Core] SingleNet::dot(): vectors must have the same size");

        T sum = T();
        for (size_t i = 0; i < v1.size(); ++i)
            sum += v1[i] * v2[i];

        return sum;
    }

    template <typename T>
    Vector<T, 2> dot(const Vector<T, 2> &v1, const Vector<T, 2> &v2)
    {
        Vector<T, 2> v(shape(v1)[0], shape(v2)[1]);
        Vector<T, 2> v2_transposed = transpose(v2);

        for (size_t i = 0; i < shape(v1)[0]; ++i)
            for (size_t j = 0; j < shape(v2)[1]; ++j)
                v[i][j] = dot(v1[i], v2_transposed[j]);

        return v;
    }
}

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const SingleNet::Vector<T, N> &v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i)
        os << v[i] << (i < v.size() - 1 ? ", " : "");
    os << "]";
    return os;
}

#endif