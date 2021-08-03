#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <cstdlib>
#include <memory.h>
#include <stdexcept>
#include <functional>

template <class T, size_t N>
class Vector
{
public:
    Vector<T, N>(void);
    Vector<T, N>(const Vector<T, N> &v);

    ~Vector<T, N>();

    Vector<T, N - 1> &operator[](const size_t &i);
    Vector<T, N - 1> at(const size_t &i) const;
    Vector<T, N - 1> front() const;
    Vector<T, N - 1> back() const;

    Vector<T, N> &operator=(const Vector<T, N> &v);
    Vector<T, N> &operator+=(const Vector<T, N> &v);
    Vector<T, N> &operator-=(const Vector<T, N> &v);
    Vector<T, N> &operator*=(const T &s);
    Vector<T, N> &operator/=(const T &s);

    Vector<T, N> operator+(const Vector<T, N> &v) const;
    Vector<T, N> operator-(const Vector<T, N> &v) const;
    Vector<T, N> operator*(const T &s) const;
    Vector<T, N> operator/(const T &s) const;

    bool operator==(const Vector<T, N> &v) const;
    bool operator!=(const Vector<T, N> &v) const;

    Vector<T, N> map(std::function<T(T)> f) const;

    void push_back(const Vector<T, N - 1> &v);
    Vector<T, N - 1> pop_back(void);
    void push_front(const Vector<T, N - 1> &v);
    Vector<T, N - 1> pop_front(void);

    Vector<T, N> slice(const Vector<size_t, 1> &start, const Vector<size_t, 1> &end) const;

    void clear(void);
    void resize(const Vector<size_t, 1> &n, const T &s);

    Vector<size_t, 1> shape(void) const;
    size_t length;

private:
    Vector<T, N - 1> *data;
};

template <class T>
class Vector<T, 1>
{
public:
    Vector<T, 1>(void);
    Vector<T, 1>(const std::vector<T> &v);
    Vector<T, 1>(const Vector<T, 1> &v);
    ~Vector<T, 1>(void);

    T &operator[](const size_t &i);
    T at(const size_t &i) const;
    T front(void) const;
    T back(void) const;

    Vector<T, 1> &operator=(const Vector<T, 1> &v);
    Vector<T, 1> &operator+=(const Vector<T, 1> &v);
    Vector<T, 1> &operator-=(const Vector<T, 1> &v);
    Vector<T, 1> &operator*=(const T &s);
    Vector<T, 1> &operator/=(const T &s);

    Vector<T, 1> operator+(const Vector<T, 1> &v) const;
    Vector<T, 1> operator-(const Vector<T, 1> &v) const;
    Vector<T, 1> operator*(const T &s) const;
    Vector<T, 1> operator/(const T &s) const;

    bool operator==(const Vector<T, 1> &v) const;
    bool operator!=(const Vector<T, 1> &v) const;

    Vector<T, 1> map(std::function<T(T)> f) const;

    void push_back(const T &v);
    T pop_back(void);
    void push_front(const T &v);
    T pop_front(void);

    Vector<T, 1> slice(const Vector<size_t, 1> &start, const Vector<size_t, 1> &end) const;
    Vector<T, 1> slice(const size_t &start, const size_t &end) const;

    T dot(const Vector<T, 1> &v) const;
    T sum(void) const;
    Vector<T, 1> reverse(void) const;

    void clear(void);
    void resize(const Vector<size_t, 1> &n, const T &s);
    void resize(const size_t &n, const T &s);
    
    Vector<size_t, 1> shape() const;

    size_t length;
private:
    T *data;
};

template <typename T>
std::ostream &operator<<(std::ostream &m_os, Vector<T, 1> m_value)
{
    m_os << std::string("[");
    for (size_t i = 0; i < m_value.length; i++)
    {
        if (i == m_value.length - 1) m_os << m_value[i];
        else m_os << m_value[i] << std::string(", ");
    }
    m_os << std::string("]");

    return m_os;
}

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &m_os, Vector<T, N> m_value)
{
    m_os << std::string("[");
    for (size_t i = 0; i < m_value.length; i++)
    {
        if (i == m_value.length - 1) m_os << m_value[i];
        else m_os << m_value[i] << std::string(", ");
    }
    m_os << std::string("]");
    return m_os;
}

template <class T>
Vector<T, 2> transpose(const Vector<T, 2> &v)
{
    Vector<T, 2> result;
    result.resize(v.shape().reverse(), 0);

    for(int i = 0; i < v.shape()[0]; i++)
        for(int j = 0; j < v.shape()[1]; j++)
            result[j][i] = v[i][j];

    return result;
}

template <class T>
Vector<T, 2> dot(const Vector<T, 2> &v1, const Vector<T, 2> &v2)
{
    if(v1.shape()[1] != v2.shape()[0])
        throw std::length_error("Vector::dot: size mismatch");

    Vector<size_t, 1> shape = {v1.shape()[0], v2.shape()[1]};
    Vector<T, 2> result;
    result.resize(shape, 0);

    for(int i = 0; i < shape[0]; i++) {
        for(int j = 0; j < shape[1]; j++) {
            Vector<T, 1> x = v1.at(i);
            Vector<T, 1> y = transpose(v2.slice({0, j}, {v2.shape()[0], j + 1})).at(0);
            result[i][j] = x.dot(y);
        }
    }

    return result;
}

#endif