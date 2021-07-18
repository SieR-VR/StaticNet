#ifndef VECTOR3D_H
#define VECTOR3D_H

#include "Vector2D.h"

struct Vector3DSize_t
{
    size_t x, y, z;

    bool operator==(const Vector3DSize_t& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Vector3DSize_t& other) const
    {
        return !(*this == other);
    }
};

enum class Vector3DAxis_t
{
    X = 0,
    Y = 1,
    Z = 2
};

template <typename T>
class Vector3D {
public:
    Vector3D(const std::vector<std::vector<std::vector<T>>> &m_value);
    Vector3D(const Vector1D<Vector1D<Vector1D<T>>> &m_value);
    Vector3D(const Vector3D<T> &m_value);

    T at(const Vector3DSize_t& m_index) const;

    Vector3D<T> &operator= (const Vector3D<T> &m_value);

    Vector3D<T> &operator+=(const Vector3D<T> &m_value);
    Vector3D<T> &operator-=(const Vector3D<T> &m_value);
    Vector3D<T> &operator+=(const T &m_value);
    Vector3D<T> &operator-=(const T &m_value);
    Vector3D<T> &operator*=(const Vector3D<T> &m_value);
    Vector3D<T> &operator/=(const Vector3D<T> &m_value);
    Vector3D<T> &operator*=(const T &m_value);
    Vector3D<T> &operator/=(const T &m_value);

    Vector3D<T> operator+(const Vector3D<T> &m_value) const;
    Vector3D<T> operator-(const Vector3D<T> &m_value) const;
    Vector3D<T> operator+(const T &m_value) const;
    Vector3D<T> operator-(const T &m_value) const;
    Vector3D<T> operator*(const Vector3D<T> &m_value) const;
    Vector3D<T> operator/(const Vector3D<T> &m_value) const;
    Vector3D<T> operator*(const T &m_value) const;
    Vector3D<T> operator/(const T &m_value) const;

    bool operator==(const Vector3D<T> &m_value) const;
    bool operator!=(const Vector3D<T> &m_value) const;

    Vector2D<T> pop(const Vector3DAxis_t &m_axis);
    void push(const Vector2D<T> &m_value, const Vector3DAxis_t &m_axis);
    void push(const Vector3D<T> &m_value, const Vector3DAxis_t &m_axis);
    void clear();
    void resize(const Vector3DSize_t &m_size);
    void resize(const Vector3DSize_t &m_size, const T &m_value);

    Vector3D<T> slice(const Vector3DAxis_t &begin = { 0, 0, 0 }, const Vector3DAxis_t &end = shape());
    Vector3D<T> map(const std::function<T(T)> &m_function) const;

    bool checkIsValid() const;
    Vector2D<T> toVector2D() const;
    Vector2D<T> getVector2DByAxis(const Vector3DAxis_t &m_axis, const size_t &m_index) const;

    Vector3DSize_t shape() const;
    
private:
    Vector1D<Vector1D<Vector1D<T>>> value;
};

#endif
