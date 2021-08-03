#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <cstdlib>
#include <memory.h>
#include <stdexcept>
#include <functional>
#include <condition_variable>

template <class T, size_t N>
class Vector
{
public:
    typedef typename std::conditional<N == 1, T, Vector<T, N - 1>>::type VectorT;
    //--------------------------------------------------
    // Constructors
    //--------------------------------------------------

    Vector(void)
    {
        data = (VectorT *)malloc(sizeof(VectorT) * 0);
        length = 0;
    }

    Vector(const Vector<T, N> &v)
    {
        data = (VectorT *)malloc(sizeof(VectorT) * v.length);
        length = v.length;

        for (size_t i = 0; i < v.length; i++)
            data[i] = v.at(i);
    }

    ~Vector()
    {
        clear();
        data = nullptr;
    }

    //--------------------------------------------------
    // Member access function
    //--------------------------------------------------

    VectorT &operator[](const size_t &i)
    {
        if (!length)
            throw std::runtime_error("Vector::at: vector doesn't have any value!");

        if (i >= length || i < 0)
            throw std::out_of_range("Vector::operator[]: can't access out of range index!");

        return data[i];
    }

    VectorT at(const size_t &i) const
    {
        if (!length)
            throw std::runtime_error("Vector::at: vector doesn't have any value!");

        if (i >= length || i < 0)
            throw std::out_of_range("Vector::at: can't access out of range index!");

        return data[i];
    }

    VectorT front() const
    {
        if (!length)
            throw std::runtime_error("Vector::at: vector doesn't have any value!");

        return data[0];
    }

    VectorT back() const
    {
        if (!length)
            throw std::runtime_error("Vector::at: vector doesn't have any value!");

        return data[length - 1];
    }

    //--------------------------------------------------
    // Assignment and compound assignment operator
    //--------------------------------------------------

    Vector<T, N> &operator=(const Vector<T, N> &v)
    {
        if (data)
            free(data);

        data = (VectorT *)malloc(sizeof(VectorT) * v.length);
        length = v.length;

        for (size_t i = 0; i < length; i++)
            data[i] = v.at(i);

        return *this;
    }

    Vector<T, N> &operator+=(const Vector<T, N> &v)
    {
        if (shape() != v.shape())
            throw std::runtime_error("Vector::operator+=: shape mismatch");

        for (size_t i = 0; i < length; i++)
            data[i] += v.at(i);

        return *this;
    }

    Vector<T, N> &operator-=(const Vector<T, N> &v)
    {
        if (shape() != v.shape())
            throw std::runtime_error("Vector::operator-=: shape mismatch");

        for (size_t i = 0; i < length; i++)
            data[i] -= v.at(i);

        return *this;
    }

    Vector<T, N> &operator*=(const T &s)
    {
        for (size_t i = 0; i < length; i++)
            data[i] *= s;

        return *this;
    }

    Vector<T, N> &operator/=(const T &s)
    {
        for (size_t i = 0; i < length; i++)
            data[i] /= s;

        return *this;
    }

    //--------------------------------------------------
    // Calculation operator
    //--------------------------------------------------

    Vector<T, N> operator+(const Vector<T, N> &v) const
    {
        if (shape() != v.shape())
            throw std::runtime_error("Vector::operator+: shape mismatch");

        Vector<T, N> result = *this;
        result += v;
        return result;
    }

    Vector<T, N> operator-(const Vector<T, N> &v) const
    {
        if (shape() != v.shape())
            throw std::runtime_error("Vector::operator-: shape mismatch");

        Vector<T, N> result = *this;
        result -= v;
        return result;
    }

    Vector<T, N> operator*(const T &s) const
    {
        Vector<T, N> result = *this;
        result *= s;
        return result;
    }

    Vector<T, N> operator/(const T &s) const
    {
        Vector<T, N> result = *this;
        result /= s;
        return result;
    }

    //--------------------------------------------------
    // Comparison operator
    //--------------------------------------------------

    bool operator==(const Vector<T, N> &v) const
    {
        if(length != v.length)
            throw std::runtime_error("Vector::operator==: shape mismatch");

        bool result = true;

        for (size_t i = 0; i < v.length; i++)
            if (at(i) != v.at(i))
                result = false;

        return result;
    }

    bool operator!=(const Vector<T, N> &v) const
    {
        if(length != v.length)
            throw std::runtime_error("Vector::operator!=: shape mismatch");
            
        return !(*this == v);
    }

    //--------------------------------------------------
    // Map function
    //--------------------------------------------------

    Vector<T, N> map(std::function<T(T)> f) const
    {
        Vector<T, N> result = *this;

        for (int i = 0; i < length; i++)
        {
            if (std::is_same<VectorT, T>::value)
                result[i] = f(result[i]);
            else
                result[i] = result[i].map(f);
        }

        return result;
    }

    //--------------------------------------------------
    // Push and pop function
    //--------------------------------------------------

    void push_back(const VectorT &v)
    {
        if constexpr (!std::is_same<VectorT, T>::value)
            if (length && data[0].shape() != v.shape())
                throw std::length_error("Vector::push_back: shape mismatch");

        data = (VectorT *)realloc(data, sizeof(VectorT) * (length + 1));
        length++;

        data[length - 1] = v;
    }

    VectorT pop_back(void)
    {
        if (!length)
            throw std::length_error("Vector::pop_back: vector is empty");

        VectorT result = data[length - 1];
        if constexpr (!std::is_same<VectorT, T>::value)
            data[length - 1].~Vector();

        data = (VectorT *)realloc(data, sizeof(VectorT) * (length - 1));
        length--;

        return result;
    }

    void push_front(const VectorT &v)
    {
        if constexpr (!std::is_same<VectorT, T>::value && length && data[0].shape() != v.shape())
            throw std::length_error("Vector::push_front: shape mismatch");

        data = (VectorT *)realloc(data, sizeof(VectorT) * (length + 1));
        length++;

        memmove(data[1], data[0], sizeof(VectorT *) * (length - 1));

        data[0] = v;
    }

    VectorT pop_front(void)
    {
        if (!length)
            throw std::length_error("Vector::pop_front: vector is empty");

        VectorT result = data[0];
        if constexpr (!std::is_same<VectorT, T>::value)
            data[0].~Vector();

        memmove(data[0], data[1], sizeof(VectorT *) * (length - 1));

        data = (VectorT *)realloc(data, sizeof(VectorT) * (length - 1));
        length--;

        return result;
    }

    //--------------------------------------------------
    // Slice
    //--------------------------------------------------

    Vector<T, N> slice(const Vector<size_t, 1> &start, const Vector<size_t, 1> &end) const
    {
        if (N != start.length || N != end.length)
            throw std::length_error("Vector::slice: Dimension size mismatch");

        bool flag = true;
        for (size_t i = 0; i < start.length; i++)
            if (start.at(i) > end.at(i))
                flag = false;

        if (!flag)
            throw std::length_error("Vector::slice: Some member of start is bigger than end's one");

        Vector<T, N> result;
        Vector<size_t, 1> start_copy = start, end_copy = end;

        for (size_t i = start_copy.back(); i < end_copy.back(); i++)
        {
            if (std::is_same<VectorT, T>::value)
                result.push_back(data[i]);
            else
                result.push_back(at(i).slice(start_copy.pop_back(), end_copy.pop_back()));
        }

        return result;
    }

    //--------------------------------------------------
    // Clear and resize
    //--------------------------------------------------

    void clear(void)
    {
        for (size_t i = 0; i < length; i++)
            if constexpr (!std::is_same<VectorT, T>::value)
                data[i].clear();

        if (data && length) free(data);
        data = (VectorT *)malloc(sizeof(VectorT) * 0);
        length = 0;
    }

    void resize(const Vector<size_t, 1> &n, const T &s)
    {
        if (N != n.length)
            throw std::length_error("Vector::resize: Dimension size mismatch");

        Vector<size_t, 1> n_copy = n;

        clear();
        data = (VectorT *)realloc(data, sizeof(VectorT *) * n_copy.back());
        length = n_copy.pop_back();

        if constexpr (!std::is_same<VectorT, T>::value)
            for (size_t i = 0; i < length; i++)
                data[i].resize(n_copy, s);
    }

    Vector<T, 1> from(const std::vector<T> &v)
    {
        Vector<T, 1> result;

        Vector<size_t, 1> n;
        n.push_back(v.size());

        result.resize(n);
        for (size_t i = 0; i < v.size(); i++)
            result[i] = v.at(i);

        return result;
    }

    //--------------------------------------------------
    // Shape
    //--------------------------------------------------

    Vector<size_t, 1> shape(void) const
    {
        Vector<size_t, 1> result;
        if constexpr (std::is_same<VectorT, T>::value)
        {
            result.push_back(length);
            return result;
        }
        else
        {
            result = at(0).shape();
            result.push_back(length);
            return result;
        }
    }

    size_t length;

private:
    VectorT *data;
};

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &m_os, const Vector<T, N> &m_value)
{
    m_os << std::string("[");
    for (size_t i = 0; i < m_value.length; i++)
    {
        if (i == m_value.length - 1)
            m_os << m_value.at(i);
        else
            m_os << m_value.at(i) << std::string(", ");
    }
    m_os << std::string("]");
    return m_os;
}

template <class T>
Vector<T, 2> transpose(const Vector<T, 2> &v)
{
    Vector<T, 2> result;
    result.resize(v.shape().reverse(), 0);

    for (int i = 0; i < v.shape()[0]; i++)
        for (int j = 0; j < v.shape()[1]; j++)
            result[j][i] = v[i][j];

    return result;
}

template <class T>
Vector<T, 2> dot(const Vector<T, 2> &v1, const Vector<T, 2> &v2)
{
    if (v1.shape()[1] != v2.shape()[0])
        throw std::length_error("Vector::dot: size mismatch");

    Vector<size_t, 1> shape = v1.shape();
    shape[1] = v2.shape()[1];

    Vector<T, 2> result;
    result.resize(shape, 0);

    for (int i = 0; i < v1.shape()[0]; i++)
        for (int j = 0; j < v2.shape()[1]; j++)
            for (int k = 0; k < v1.shape()[1]; k++)
                result[i][j] += v1.at(i).at(k) * v2.at(i).at(j);

    return result;
}

#endif