#ifndef VECTOR1D_H
#define VECTOR1D_H

#include <vector> 
#include <functional>
#include <stdexcept>

struct Vector1DSize_t {
    size_t x;

    bool operator==(const Vector1DSize_t& other) const {
        return x == other.x;
    }

    bool operator!=(const Vector1DSize_t& other) const {
        return !(*this == other);
    }
};

enum Vector1DAxis_t {
    X = 0,
};

template <class T>
class Vector1D {
public:
    Vector1D(const std::vector<T> &m_value);
    Vector1D(const Vector1D<T> &m_value);
    ~Vector1D();

    T &operator[](const size_t &i);

    Vector1D<T> &operator=(const Vector1D<T> &m_value);

    Vector1D<T> &operator+=(const Vector1D<T> &m_value);
    Vector1D<T> &operator-=(const Vector1D<T> &m_value);
    Vector1D<T> &operator+=(const T &m_value);
    Vector1D<T> &operator-=(const T &m_value);
    Vector1D<T> &operator*=(const Vector1D<T> &m_value);
    Vector1D<T> &operator/=(const Vector1D<T> &m_value);
    Vector1D<T> &operator*=(const T &m_value);
    Vector1D<T> &operator/=(const T &m_value);

    Vector1D<T> operator+(const Vector1D<T> &m_value) const;
    Vector1D<T> operator-(const Vector1D<T> &m_value) const;
    Vector1D<T> operator+(const T &m_value) const;
    Vector1D<T> operator-(const T &m_value) const;
    Vector1D<T> operator*(const Vector1D<T> &m_value) const;
    Vector1D<T> operator/(const Vector1D<T> &m_value) const;
    Vector1D<T> operator*(const T &m_value) const;
    Vector1D<T> operator/(const T &m_value) const;

    bool operator==(const Vector1D<T> &m_value) const;
    bool operator!=(const Vector1D<T> &m_value) const;

    T pop(const size_t &i = -1);
    void push(const T &m_value);
    void push(const Vector1D<T> &m_value);
    void clear();
    void resize(const Vector1DSize_t &m_size);
    void resize(const Vector1DSize_t &m_size, const T &m_value);

    Vector1D<T> slice(const Vector1DSize_t &begin = 0, const Vector1DSize_t &end = value.size()) const;
    Vector1D<T> map(const std::function<T(T)> &m_function) const;
    
    Vector1DSize_t shape() const;
private:
    static std::vector<T> value;
};

#endif

