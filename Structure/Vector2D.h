#ifndef VECTOR2D_H
#define VECTOR2D_H

#include "Vector1D.h"

struct Vector2DSize_t {
    size_t x, y;

    bool operator==(const Vector2DSize_t& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Vector2DSize_t& other) const {
        return !(*this == other);
    }
};

enum class Vector2DAxis_t {
    X = 0,
    Y = 1
};

template <typename T>
class Vector2D {
public:
    Vector2D(const std::vector<std::vector<T>>& m_value);
    Vector2D(const Vector1D<Vector1D<T>>& m_value);
    Vector2D(const Vector2D<T>& m_value);
    ~Vector2D();

    Vector1D<T> operator[](const size_t &index) const;
    T at(const Vector2DSize_t &index) const;

    Vector2D<T> &operator=(const Vector2D<T>& m_value);

    Vector2D<T> &operator+=(const Vector2D<T>& m_value);
    Vector2D<T> &operator-=(const Vector2D<T>& m_value);
    Vector2D<T> &operator+=(const T& m_value);
    Vector2D<T> &operator-=(const T& m_value);
    Vector2D<T> &operator*=(const Vector2D<T>& m_value);
    Vector2D<T> &operator/=(const Vector2D<T>& m_value);
    Vector2D<T> &operator*=(const T& m_value);
    Vector2D<T> &operator/=(const T& m_value);

    Vector2D<T> operator+(const Vector2D<T>& m_value) const;
    Vector2D<T> operator-(const Vector2D<T>& m_value) const;
    Vector2D<T> operator+(const T& m_value) const;
    Vector2D<T> operator-(const T& m_value) const;
    Vector2D<T> operator*(const Vector2D<T>& m_value) const;
    Vector2D<T> operator/(const Vector2D<T>& m_value) const;
    Vector2D<T> operator*(const T& m_value) const;
    Vector2D<T> operator/(const T& m_value) const;

    bool operator==(const Vector2D<T>& m_value) const;
    bool operator!=(const Vector2D<T>& m_value) const;

    Vector1D<T> pop(const Vector2DAxis_t &axis, const size_t &index = -1);
    void push(const Vector1D<T>& m_value, const Vector2DAxis_t &m_axis);
    void push(const Vector2D<T>& m_value, const Vector2DAxis_t &m_axis);
    void clear();
    void resize(const Vector2DSize_t& m_size);
    void resize(const Vector2DSize_t& m_size, const T& m_value);

    Vector2D<T> slice(const Vector2DSize_t& begin = {0, 0}, const Vector2DSize_t& end = shape()) const;
    Vector2D<T> map(const std::function<T(T)>& m_function) const;

    bool checkIsValid() const;
    Vector1D<T> toVector1D() const;
    Vector1D<T> getVector1DByAxis(const Vector2DAxis_t &m_axis, const size_t &m_index) const;
    Vector2D<T> transpose() const;

    Vector2DSize_t shape() const;
private:
    Vector1D<Vector1D<T>> value;
};

#endif