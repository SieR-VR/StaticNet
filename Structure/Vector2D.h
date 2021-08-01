#ifndef VECTOR2D_H
#define VECTOR2D_H

#include "Vector1D.h"
#include <iostream>

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

template <typename T>
class Vector2D
{
public:
    Vector2D(const std::vector<std::vector<T>> &m_value)
    {
        for (size_t i = 0; i < m_value.size(); i++)
            value.push_back(m_value[i]);
    }

    Vector2D(const std::vector<Vector1D<T>> &m_value)
    {
        value = m_value;
    }

    Vector2D(const Vector2D &m_value)
    {
        value = m_value.value;
    }

    Vector2D()
    {
        value.clear();
    }

    ~Vector2D() {}

    T &at(const Vector2DSize_t &index)
    {
        return value[index.y].at(index.x);
    }

    Vector1D<T> at(const size_t &index, const Vector2DAxis_t &axis)
    {
        switch (axis)
        {
        case Vector2DAxis_t::X:
        {
            Vector1D<T> result;
            for (size_t i = 0; i < shape().y; i++)
                result.push(value[i].at(index));

            return result;
        }
        case Vector2DAxis_t::Y:
        {
            Vector1D<T> result = value.at(index);

            return result;
        }
        default:
            throw std::invalid_argument("Vector2D::at: invalid axis");
        }
    }

    Vector1D<T> &operator[](const size_t &index)
    {
        return value[index];
    }

    Vector2D<T> &operator=(const Vector2D &m_value)
    {
        value = m_value.value;
        return *this;
    }

    Vector2D<T> &operator+=(const Vector2D &m_value)
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector2D::operator+=: shape mismatch");
            return *this;
        }

        for (size_t i = 0; i < value.size(); i++)
            value[i] += m_value[i];

        return *this;
    }

    Vector2D<T> &operator-=(const Vector2D &m_value)
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector2D::operator-=: shape mismatch");
            return *this;
        }

        for (size_t i = 0; i < value.size(); i++)
            value[i] -= m_value[i];

        return *this;
    }

    Vector2D<T> &operator+=(const T &m_value)
    {
        for (size_t i = 0; i < value.size(); i++)
            value[i] += m_value;

        return *this;
    }

    Vector2D<T> &operator-=(const T &m_value)
    {
        for (size_t i = 0; i < value.size(); i++)
            value[i] -= m_value;

        return *this;
    }

    Vector2D<T> &operator*=(const Vector2D &m_value)
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector2D::operator*=: shape mismatch");
            return *this;
        }

        for (size_t i = 0; i < value.size(); i++)
            value[i] *= m_value[i];

        return *this;
    }

    Vector2D<T> &operator/=(const Vector2D &m_value)
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector2D::operator/=: shape mismatch");
            return *this;
        }

        for (size_t i = 0; i < value.size(); i++)
            value[i] /= m_value[i];

        return *this;
    }

    Vector2D<T> &operator*=(const T &m_value)
    {
        for (size_t i = 0; i < value.size(); i++)
            value[i] *= m_value;

        return *this;
    }

    Vector2D<T> &operator/=(const T &m_value)
    {
        for (size_t i = 0; i < value.size(); i++)
            value[i] /= m_value;

        return *this;
    }

    Vector2D<T> operator+(const Vector2D &m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector2D::operator+=: shape mismatch");
            return value;
        }

        Vector2D result;
        for (size_t i = 0; i < value.size(); i++)
            result[i] = value[i] + m_value[i];

        return result;
    }

    Vector2D<T> operator-(const Vector2D &m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector2D::operator-=: shape mismatch");
            return value;
        }

        Vector2D result;
        for (size_t i = 0; i < value.size(); i++)
            result[i] = value[i] - m_value[i];

        return result;
    }

    Vector2D<T> operator+(const T &m_value) const
    {
        Vector2D result;
        for (size_t i = 0; i < value.size(); i++)
            result[i] = value[i] + m_value;

        return result;
    }

    Vector2D<T> operator-(const T &m_value) const
    {
        Vector2D result;
        for (size_t i = 0; i < value.size(); i++)
            result[i] = value[i] - m_value;

        return result;
    }

    Vector2D<T> operator*(const Vector2D &m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector2D::operator*=: shape mismatch");
            return value;
        }

        Vector2D result;
        for (size_t i = 0; i < value.size(); i++)
            result[i] = value[i] * m_value[i];

        return result;
    }

    Vector2D<T> operator/(const Vector2D &m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector2D::operator/=: shape mismatch");
            return value;
        }

        Vector2D result;
        for (size_t i = 0; i < value.size(); i++)
            result[i] = value[i] / m_value[i];

        return result;
    }

    Vector2D<T> operator*(const T &m_value) const
    {
        Vector2D<T> result;
        result.resize(shape());

        for (size_t i = 0; i < value.size(); i++)
            result[i] = value[i] * m_value;

        return result;
    }

    Vector2D<T> operator/(const T &m_value) const
    {
        Vector2D<T> result;
        result.resize(shape());

        for (size_t i = 0; i < value.size(); i++)
            result[i] = value[i] / m_value;

        return result;
    }

    bool operator==(const Vector2D &m_value) const
    {
        if (shape() != m_value.shape())
            return false;

        for (size_t i = 0; i < value.size(); i++)
            if (value[i] != m_value[i])
                return false;

        return true;
    }

    bool operator!=(const Vector2D &m_value) const
    {
        return !(*this == m_value);
    }

    Vector1D<T> pop(const Vector2DAxis_t &axis, const size_t &index)
    {
        switch (axis)
        {
        case Vector2DAxis_t::X:
        {
            Vector1D<T> result;
            for (size_t i = 0; i < shape().y; i++)
                result.push(value[i].pop(index));

            return result;
        }
        case Vector2DAxis_t::Y:
        {
            Vector1D<T> result = value.at(index);
            value.erase(value.begin() + index);

            return result;
        }
        default:
            throw std::invalid_argument("Vector2D::pop: invalid axis");
        }
    }

    void push(Vector1D<T> m_value, const Vector2DAxis_t &m_axis)
    {
        switch (m_axis)
        {
        case Vector2DAxis_t::X:
        {
            if (shape().y != m_value.shape().x)
            {
                throw std::invalid_argument("Vector2D::push: vector y size mismatch");
                return;
            }

            for (size_t i = 0; i < m_value.shape().x; i++)
                value[i].push(m_value[i]);

            break;
        }
        case Vector2DAxis_t::Y:
        {           
            if (shape().x != m_value.shape().x)
            {
                throw std::invalid_argument("Vector2D::push: vector x size mismatch");
                return;
            }

            value.push_back(m_value);
            break;
        }
        }
    }

    void push(const Vector2D &m_value, const Vector2DAxis_t &m_axis)
    {
        switch (m_axis)
        {
        case Vector2DAxis_t::X:
        {
            if (shape().y != m_value.shape().y)
            {
                throw std::invalid_argument("Vector2D::push: vector y size mismatch");
                return;
            }

            for (size_t i = 0; i < m_value.shape().y; i++)
                value[i].push(m_value[i]);

            break;
        }
        case Vector2DAxis_t::Y:
        {
            if (shape().x != m_value.shape().x)
            {
                throw std::invalid_argument("Vector2D::push: vector x size mismatch");
                return;
            }

            for (size_t i = 0; i < m_value.shape().x; i++)
                value.push_back(m_value[i]);

            break;
        }
        }
    }

    void clear()
    {
        value.clear();
    }

    void resize(const Vector2DSize_t &m_size)
    {
        value.resize(m_size.y);
        for (size_t i = 0; i < m_size.y; i++)
            value[i].resize({m_size.x});
    }

    void resize(const Vector2DSize_t &m_size, const T &m_value)
    {
        value.resize(m_size.y);
        for (size_t i = 0; i < m_size.y; i++)
            value[i].resize({m_size.x}, m_value);
    }

    Vector2D<T> slice(const Vector2DSize_t &begin, const Vector2DSize_t &end)
    {
        Vector2D<T> result;
        result.resize({end.x - begin.x, end.y - begin.y});
        for (size_t i = 0; i < result.shape().x; i++)
        {
            for (size_t j = 0; j < result.shape().y; j++)
            {
                result.at({i, j}) = value[j + begin.y][i + begin.x];
            }
        }
        return result;
    }

    Vector2D<T> map(const std::function<T(T)> &m_function) const
    {
        Vector2D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i].map(m_function));

        return result;
    }

    bool checkIsValid() const
    {
        Vector1DSize_t size = value[0].shape();
        for (size_t i = 1; i < shape().y; i++)
            if (value[i].shape() != size)
                return false;

        return true;
    }

    Vector1D<T> toVector1D() const
    {
        Vector1D<T> result;
        for (size_t i = 0; i < shape().y; i++)
            result.push(value[i]);

        return result;
    }

    Vector2D<T> transpose() const
    {
        Vector2D<T> result;
        result.resize({shape().x, shape().y});
        for (size_t i = 0; i < shape().y; i++)
            for (size_t j = 0; j < shape().x; j++)
                result.value[j].at(i) = value[i].at(j);

        return result;
    }

    Vector2DSize_t shape() const
    {
        if(value.size() == 0) return {0, 0};
        else return {value[0].shape().x, value.size()};
    }

private:
    std::vector<Vector1D<T>> value;
};

#endif