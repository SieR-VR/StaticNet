#ifndef VECTOR1D_H
#define VECTOR1D_H

#include <vector>
#include <functional>
#include <stdexcept>

struct Vector1DSize_t
{
    size_t x;

    bool operator==(const Vector1DSize_t &other) const
    {
        return x == other.x;
    }

    bool operator!=(const Vector1DSize_t &other) const
    {
        return !(*this == other);
    }

    bool checkIsValid() const
    {
        return x > 0;
    }
};

enum class Vector1DAxis_t
{
    X = 0,
};

template <typename T>
class Vector1D
{
public:
    Vector1D(const std::vector<T> &m_value)
    {
        value = m_value;
    }

    Vector1D(const Vector1D &m_value)
    {
        value = m_value.value;
    }

    Vector1D()
    {
        value = std::vector<T>();
    }

    ~Vector1D()
    {
        value.clear();
    }

    T &operator[](const size_t &i)
    {
        return value[i];
    }

    T &at(const size_t &i)
    {
        return value[i];
    }

    Vector1D &operator=(const Vector1D &m_value)
    {
        value = m_value.value;
        return *this;
    }

    Vector1D<T> &operator+=(Vector1D<T> m_value)
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator+=: shape mismatch");
            return *this;
        }

        for (size_t i = 0; i < value.size(); i++)
            value[i] += m_value[i];

        return *this;
    }

    Vector1D<T> &operator-=(Vector1D<T> m_value)
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator-=: shape mismatch");
            return *this;
        }

        for (size_t i = 0; i < value.size(); i++)
            value[i] -= m_value[i];

        return *this;
    }

    Vector1D<T> &operator+=(const T &m_value)
    {
        for (size_t i = 0; i < value.size(); i++)
            value[i] += m_value;

        return *this;
    }

    Vector1D<T> &operator-=(const T &m_value)
    {
        for (size_t i = 0; i < value.size(); i++)
            value[i] -= m_value;

        return *this;
    }

    Vector1D<T> &operator*=(const Vector1D<T> &m_value)
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator*=: shape mismatch");
            return *this;
        }

        for (int i = 0; i < value.size(); i++)
            value[i] *= m_value.at(i);

        return *this;
    }

    Vector1D<T> &operator/=(const Vector1D<T> &m_value)
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator/=: shape mismatch");
            return *this;
        }

        for (int i = 0; i < value.size(); i++)
            value[i] /= m_value.at(i);

        return *this;
    }

    Vector1D<T> &operator*=(const T &m_value)
    {
        for (size_t i = 0; i < value.size(); i++)
            value[i] *= m_value;

        return *this;
    }

    Vector1D<T> &operator/=(const T &m_value)
    {
        for (size_t i = 0; i < value.size(); i++)
            value[i] /= m_value;

        return *this;
    }

    Vector1D<T> operator+(const Vector1D<T> &m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator+=: shape mismatch");
            return Vector1D<T>(value);
        }

        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i] + m_value[i]);

        return result;
    }

    Vector1D<T> operator-(const Vector1D<T> &m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator-=: shape mismatch");
            return Vector1D<T>(value);
        }

        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i] - m_value[i]);

        return result;
    }

    Vector1D<T> operator+(const T &m_value) const
    {
        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i] + m_value);

        return result;
    }

    Vector1D<T> operator-(const T &m_value) const
    {
        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i] - m_value);

        return result;
    }

    Vector1D<T> operator*(Vector1D<T> m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator*: shape mismatch");
            return Vector1D<T>(value);
        }

        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i] * m_value[i]);

        return result;
    }

    Vector1D<T> operator/(const Vector1D<T> &m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator/: shape mismatch");
            return Vector1D<T>(value);
        }

        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i] / m_value.at(i));

        return result;
    }

    Vector1D<T> operator*(const T &m_value)
    {
        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i] * m_value);

        return result;
    }

    Vector1D<T> operator/(const T &m_value)
    {
        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(value[i] / m_value);

        return result;
    }

    float mean() const
    {
        float sum = 0;
        for (size_t i = 0; i < value.size(); i++)
            sum += value[i];

        return sum;
    }

    bool operator==(const Vector1D<T> &m_value) const
    {
        if (shape() != m_value.shape())
        {
            throw std::invalid_argument("Vector1D::operator==: shape mismatch");
            return false;
        }

        for (size_t i = 0; i < value.size(); i++)
        {
            if (value[i] != m_value.at(i))
                return false;
        }
        return true;
    }

    bool operator!=(const Vector1D<T> &m_value) const
    {
        return !(*this == m_value);
    }

    T pop(const size_t &i)
    {
        if (i >= shape().x)
        {
            throw std::out_of_range("Vector1D::pop: index out of range");
        }

        T result;
        result = value[i];
        value.erase(value.begin() + i);

        return result;
    }

    void push(const T &m_value)
    {
        value.push_back(m_value);
    }

    void push(Vector1D<T> &m_value)
    {
        for (size_t i = 0; i < m_value.shape().x; i++)
            value.push_back(m_value[i]);
    }

    void clear()
    {
        value.clear();
    }

    void resize(const Vector1DSize_t &m_size)
    {
        if (!m_size.checkIsValid())
        {
            throw std::invalid_argument("Vector1D::resize: invalid size");
            return;
        }

        value.clear();
        value.resize(m_size.x);
    }

    void resize(const Vector1DSize_t &m_size, const T &m_value)
    {
        if (!m_size.checkIsValid())
        {
            throw std::invalid_argument("Vector1D::resize: invalid size");
            return;
        }

        value.clear();
        value.resize(m_size.x, m_value);
    }

    Vector1D<T> slice(const Vector1DSize_t &begin, const Vector1DSize_t &end) const
    {
        if (begin.x >= value.size() || end.x >= value.size())
        {
            throw std::out_of_range("Vector1D::slice: index out of range");
            return Vector1D<T>(value);
        }

        if (begin.x > end.x)
        {
            throw std::invalid_argument("Vector1D::slice: begin index is greater than end index");
            return Vector1D<T>(value);
        }

        Vector1D<T> result;
        for (int i = begin.x; i < end.x; i++)
            result.push(value[i]);

        return result;
    }

    Vector1D<T> map(const std::function<T(T)> &m_function) const
    {
        Vector1D<T> result;
        for (size_t i = 0; i < value.size(); i++)
            result.push(m_function(value[i]));

        return result;
    }

    Vector1DSize_t shape() const
    {
        return {value.size()};
    }

    std::vector<T> value;
};

template <typename T>
std::ostream &operator<<(std::ostream &m_os, Vector1D<T> m_value)
{
    m_os << std::string("[");
    for (size_t i = 0; i < m_value.shape().x; i++)
    {
        if (i == m_value.shape().x - 1)
            m_os << m_value[i] << std::string("]");
        else
            m_os << m_value[i] << std::string(", ");
    }

    return m_os;
}


#endif
