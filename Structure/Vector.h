#ifndef VECTOR_H
#define VECTOR_H

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

// template <class T, size_t N>
// class Vector<T, 2> : public Vector<T, N>
// {
// public:
//     Vector<T, 2> transpose(void) const;
//     Vector<T, 2> dot(const Vector<T, 2> &v) const;
// };

template <class T>
class Vector<T, 1>
{
public:
    Vector<T, 1>(void);
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

    void clear(void);
    void resize(const Vector<size_t, 1> &n, const T &s);
    void resize(const size_t &n, const T &s);
    
    Vector<size_t, 1> shape() const;

    size_t length;
private:
    T *data;
};

#endif