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

    bool checkIsValid() const
    {
        return x > 0 && y > 0 && z > 0;
    }
};

enum class Vector3DAxis_t
{
    X = 0,
    Y = 1,
    Z = 2
};

class Vector3D {
public:
    Vector3D(const std::vector<std::vector<std::vector<float>>> &m_value);
    Vector3D(const Vector3D &m_value);
    Vector3D();
    ~Vector3D();

    float &at(const Vector3DSize_t& m_index) const;
    Vector2D operator[](const size_t& m_index) const;

    Vector3D &operator=(const Vector3D &m_value);

    Vector3D &operator+=(const Vector3D &m_value);
    Vector3D &operator-=(const Vector3D &m_value);
    Vector3D &operator+=(const float &m_value);
    Vector3D &operator-=(const float &m_value);
    Vector3D &operator*=(const Vector3D &m_value);
    Vector3D &operator/=(const Vector3D &m_value);
    Vector3D &operator*=(const float &m_value);
    Vector3D &operator/=(const float &m_value);

    Vector3D operator+(const Vector3D &m_value) const;
    Vector3D operator-(const Vector3D &m_value) const;
    Vector3D operator+(const float &m_value) const;
    Vector3D operator-(const float &m_value) const;
    Vector3D operator*(const Vector3D &m_value) const;
    Vector3D operator/(const Vector3D &m_value) const;
    Vector3D operator*(const float &m_value) const;
    Vector3D operator/(const float &m_value) const;

    bool operator==(const Vector3D &m_value) const;
    bool operator!=(const Vector3D &m_value) const;

    Vector2D pop(const Vector3DAxis_t &m_axis, const size_t &m_index);
    void push(const Vector2D &m_value, const Vector3DAxis_t &m_axis);
    void push(const Vector3D &m_value, const Vector3DAxis_t &m_axis);
    void clear();
    void resize(const Vector3DSize_t &m_size);
    void resize(const Vector3DSize_t &m_size, const float &m_value);

    Vector3D slice(const Vector3DSize_t &begin = { 0, 0, 0 }, const Vector3DSize_t &end = shape()) const;
    Vector3D map(const std::function<float(float)> &m_function) const;

    bool checkIsValid() const;
    Vector2D toVector2D() const;
    Vector2D getVector2DByAxis(const Vector3DAxis_t &m_axis, const size_t &m_index) const;

    static Vector3DSize_t shape();
    
private:
    static std::vector<Vector2D> value;
};

#endif
