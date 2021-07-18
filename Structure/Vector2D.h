#ifndef VECTOR2D_H
#define VECTOR2D_H

#include "Vector1D.h"

struct Vector2DSize_t
{
    size_t x, y;

    bool operator==(const Vector2DSize_t &other) const
    {
        return x == other.x && y == other.y;
    }

    bool operator!=(const Vector2DSize_t &other) const
    {
        return !(*this == other);
    }

    bool checkIsValid() const
    {
        return x > 0 && y > 0;
    }
};

enum class Vector2DAxis_t
{
    X = 0,
    Y = 1
};

class Vector2D
{
public:
    Vector2D(const std::vector<std::vector<float>> &m_value);
    Vector2D(const std::vector<Vector1D> &m_value);
    Vector2D(const Vector2D &m_value);
    Vector2D();
    ~Vector2D();

    Vector1D operator[](const size_t &index) const;
    float &at(const Vector2DSize_t &index) const;

    Vector2D &operator=(const Vector2D &m_value);

    Vector2D &operator+=(const Vector2D &m_value);
    Vector2D &operator-=(const Vector2D &m_value);
    Vector2D &operator+=(const float &m_value);
    Vector2D &operator-=(const float &m_value);
    Vector2D &operator*=(const Vector2D &m_value);
    Vector2D &operator/=(const Vector2D &m_value);
    Vector2D &operator*=(const float &m_value);
    Vector2D &operator/=(const float &m_value);

    Vector2D operator+(const Vector2D &m_value) const;
    Vector2D operator-(const Vector2D &m_value) const;
    Vector2D operator+(const float &m_value) const;
    Vector2D operator-(const float &m_value) const;
    Vector2D operator*(const Vector2D &m_value) const;
    Vector2D operator/(const Vector2D &m_value) const;
    Vector2D operator*(const float &m_value) const;
    Vector2D operator/(const float &m_value) const;

    bool operator==(const Vector2D &m_value) const;
    bool operator!=(const Vector2D &m_value) const;

    Vector1D pop(const Vector2DAxis_t &axis, const size_t &index);
    void push(const Vector1D &m_value, const Vector2DAxis_t &m_axis = Vector2DAxis_t::Y);
    void push(const Vector2D &m_value, const Vector2DAxis_t &m_axis = Vector2DAxis_t::Y);
    void clear();
    void resize(const Vector2DSize_t &m_size);
    void resize(const Vector2DSize_t &m_size, const float &m_value);

    Vector2D slice(const Vector2DSize_t &begin = {0, 0}, const Vector2DSize_t &end = shape()) const;
    Vector2D map(const std::function<float(float)> &m_function) const;

    bool checkIsValid() const;
    Vector1D toVector1D() const;
    Vector1D getVector1DByAxis(const Vector2DAxis_t &m_axis, const size_t &m_index) const;
    Vector2D transpose() const;

    static Vector2DSize_t shape();

private:
    static std::vector<Vector1D> value;
};

#endif