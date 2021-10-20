/* Copyright 2021- SieR-VR */

#ifndef VECTOR_H_
#define VECTOR_H_

#include <functional>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace SingleNet
{
    template <typename T, size_t N>
    class Vector : public std::vector<Vector<T, N - 1>>
    {
        static_assert(N > 0, "Vector must have at least one dimension");
        typedef Vector<T, N - 1> SubVector;

    public:
        template <typename... Args>
        explicit Vector(size_t n = 0, Args... args)
            : std::vector<SubVector>(n, SubVector(args...)) {}
        Vector(std::initializer_list<SubVector> list)
            : std::vector<SubVector>(list) {}
    };

    template <typename T>
    class Vector<T, 1> : public std::vector<T>
    {
    public:
        explicit Vector(size_t n = 0, const T &value = T())
            : std::vector<T>(n, value) {}
        Vector(std::initializer_list<T> list) : std::vector<T>(list) {}
    };

    // Vector operations

    template <typename T, size_t N>
    Vector<size_t, 1> shape(const Vector<T, N> &v)
    {
        if constexpr (N == 1)
        {
            return Vector<size_t, 1>(1, v.size());
        }
        else
        {
            Vector<size_t, 1> _shape = shape(v[0]);
            _shape.insert(_shape.begin(), v.size());
            return _shape;
        }

        return Vector<size_t, 1>();
    }

    template <typename T, size_t N>
    Vector<T, N> mask(const Vector<T, N> &v, const Vector<size_t, 1> &index)
    {
        Vector<T, N> _v;
        for (size_t i = 0; i < index.size(); i++)
            _v.push_back(v[index[i]]);

        return _v;
    } // move to datasets.cpp

    template <typename T, size_t N>
    Vector<T, N> map(const Vector<T, N> &v, const std::function<T(T)> &f)
    {
        if constexpr (N == 1)
        {
            Vector<T, N> _v;
            for (size_t i = 0; i < v.size(); i++)
                _v.push_back(f(v[i]));

            return _v;
        }
        else
        {
            Vector<T, N> _v;
            for (size_t i = 0; i < v.size(); i++)
                _v.push_back(map(v[i], f));

            return _v;
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
            throw std::invalid_argument(
                "SingleNet::operator+(): vectors must have the same shape");

        Vector<T, N> v = v1;
        for (size_t i = 0; i < v1.size(); ++i)
            v[i] = v1[i] + v2[i];
        return v;
    }

    template <typename T, size_t N>
    Vector<T, N> operator-(const Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::invalid_argument(
                "SingleNet::operator-(): vectors must have the same shape");

        Vector<T, N> v = v1;
        for (size_t i = 0; i < v1.size(); ++i)
            v[i] = v1[i] - v2[i];
        return v;
    }

    template <typename T, size_t N>
    Vector<T, N> &operator+=(Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::invalid_argument(
                "SingleNet::operator+=(): vectors must have the same shape");

        for (size_t i = 0; i < v1.size(); ++i)
            v1[i] += v2[i];
        return v1;
    }

    template <typename T, size_t N>
    Vector<T, N> &operator-=(Vector<T, N> &v1, const Vector<T, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::invalid_argument(
                "SingleNet::operator-=(): vectors must have the same shape");

        for (size_t i = 0; i < v1.size(); ++i)
            v1[i] -= v2[i];
        return v1;
    }

    template <typename T, size_t N>
    Vector<T, N> operator*(const Vector<T, N> &v1, const T &s)
    {
        Vector<T, N> v = v1;
        for (size_t i = 0; i < v1.size(); ++i)
            v[i] *= s;

        return v;
    }

    template <typename T, size_t N>
    Vector<T, N> operator*(const T &s, const Vector<T, N> &v1)
    {
        return v1 * s;
    }

    template <typename T, size_t N>
    Vector<T, N> &operator*=(Vector<T, N> &v1, const T &s)
    {
        for (size_t i = 0; i < v1.size(); ++i)
            v1[i] *= s;

        return v1;
    }

    template <typename T, size_t N>
    Vector<T, N> operator/(const Vector<T, N> &v1, const T &s)
    {
        Vector<T, N> v = v1;
        for (size_t i = 0; i < v1.size(); ++i)
            v[i] /= s;

        return v;
    }

    template <typename T, size_t N>
    Vector<T, N> &operator/=(Vector<T, N> &v1, const T &s)
    {
        for (size_t i = 0; i < v1.size(); ++i)
            v1[i] /= s;

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
    Vector<T, 2> transpose(const Vector<T, 1> &v)
    {
        Vector<T, 2> res(1, v.size());
        res[0] = v;

        return res;
    }

    template <typename T>
    T dot(const Vector<T, 1> &v1, const Vector<T, 1> &v2)
    {
        if (v1.size() != v2.size())
            throw std::invalid_argument(
                "SingleNet::dot(<T, 1>, <T, 1>): vectors must have the same size");

        T sum = T();
        for (size_t i = 0; i < v1.size(); ++i)
            sum += v1[i] * v2[i];

        return sum;
    }

    template <typename T>
    Vector<T, 1> dot(const Vector<T, 2> &v1, const Vector<T, 1> &v2)
    {
        if (shape(v1)[1] != v2.size())
            throw std::invalid_argument(
                "SingleNet::dot(<T, 2>, <T, 1>): vectors must have the same size");

        Vector<T, 1> res(shape(v1)[0]);

        for (size_t i = 0; i < shape(v1)[0]; ++i)
            res[i] = dot(v1[i], v2);

        return res;
    }

    template <typename T>
    Vector<T, 2> dot(const Vector<T, 2> &v1, const Vector<T, 2> &v2)
    {
        if (shape(v1)[1] != shape(v2)[0])
            throw std::invalid_argument(
                "SingleNet::dot(<T, 2>, <T, 2>): vectors must have compatible shapes");

        Vector<T, 2> v(shape(v1)[0], shape(v2)[1]);
        Vector<T, 2> v2_transposed = transpose(v2);

        for (size_t i = 0; i < shape(v1)[0]; ++i)
            for (size_t j = 0; j < shape(v2)[1]; ++j)
                v[i][j] = dot(v1[i], v2_transposed[j]);

        return v;
    }

    template <typename T>
    T mean(const Vector<T, 1> &v)
    {
        T sum = T();

        for (size_t i = 0; i < v.size(); ++i)
            sum += v[i];

        return sum / v.size();
    }

    template <typename T>
    size_t maxIndex(const Vector<T, 1> &v)
    {
        size_t max_index = 0;
        T max_value = v[0];

        for (size_t i = 1; i < v.size(); ++i)
        {
            if (v[i] > max_value)
            {
                max_value = v[i];
                max_index = i;
            }
        }

        return max_index;
    }

    template <typename T>
    Vector<T, 1> reverse(const Vector<T, 1> &v)
    {
        Vector<T, 1> res(v.size());

        for (size_t i = 0; i < v.size(); ++i)
            res[i] = v[v.size() - i - 1];

        return res;
    }

    template <typename T, size_t N>
    void *to_pointer(const Vector<T, N> &v)
    {
        if constexpr (N == 1)
        {
            void * ptr = malloc(sizeof(T) * v.size());
            memcpy(ptr, v.data(), sizeof(T) * v.size());
            return ptr;
        }
        else {
            void * ptr = malloc(sizeof(void *) * v.size());
            for (size_t i = 0; i < v.size(); ++i)
                (static_cast<void **>(ptr))[i] = to_pointer(v[i]);

            return ptr;
        }

        return nullptr;
    }   

    template <typename T, size_t N>
    Vector<T, N> from_pointer(void *ptr, const Vector<size_t, 1> &shape_reversed)
    {
        if constexpr (N == 1)
        {
            Vector<T, 1> v;
            for (size_t i = 0; i < shape_reversed[0]; ++i)
                v.push_back((static_cast<T *>(ptr))[i]);
            
            if(ptr) free(ptr);
            return v;
        }
        else {
            Vector<T, N> v;

            for (size_t i = 0; i < shape_reversed[N-1]; ++i)
                v.push_back(from_pointer<T, N-1>(static_cast<void **>(ptr)[i], shape_reversed));

            if(ptr) free(ptr);
            return v;
        }

        return Vector<T, N>();
    }

} // namespace SingleNet

void CUDA_Init();

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const SingleNet::Vector<T, N> &v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i)
        os << v[i] << (i < v.size() - 1 ? ", " : "");
    os << "]";
    return os;
}

#endif // VECTOR_H_
