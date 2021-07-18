#include "Vector2D.h"

Vector2D::Vector2D(const std::vector<std::vector<float>> &m_value)
{
    for (size_t i = 0; i < m_value.size(); i++)
        value.push_back(m_value[i]);
}

Vector2D::Vector2D(const std::vector<Vector1D> &m_value)
{
    value = m_value;
}

Vector2D::Vector2D(const Vector2D &m_value)
{
    value = m_value.value;
}

Vector2D::Vector2D()
{
    value.clear();
}

Vector2D::~Vector2D() {}

float &Vector2D::at(const Vector2DSize_t &index) const
{
    return value[index.y].at(index.x);
}

Vector1D Vector2D::operator[](const size_t &index) const
{
    return Vector1D(value[index]);
}

Vector2D &Vector2D::operator=(const Vector2D &m_value)
{
    value = m_value.value;
    return *this;
}

Vector2D &Vector2D::operator+=(const Vector2D &m_value)
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

Vector2D &Vector2D::operator-=(const Vector2D &m_value)
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

Vector2D &Vector2D::operator+=(const float &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] += m_value;

    return *this;
}

Vector2D &Vector2D::operator-=(const float &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] -= m_value;

    return *this;
}

Vector2D &Vector2D::operator*=(const Vector2D &m_value)
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

Vector2D &Vector2D::operator/=(const Vector2D &m_value)
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

Vector2D &Vector2D::operator*=(const float &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] *= m_value;

    return *this;
}

Vector2D &Vector2D::operator/=(const float &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] /= m_value;

    return *this;
}

Vector2D Vector2D::operator+(const Vector2D &m_value) const
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

Vector2D Vector2D::operator-(const Vector2D &m_value) const
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

Vector2D Vector2D::operator+(const float &m_value) const
{
    Vector2D result;
    for (size_t i = 0; i < value.size(); i++)
        result[i] = value[i] + m_value;

    return result;
}

Vector2D Vector2D::operator-(const float &m_value) const
{
    Vector2D result;
    for (size_t i = 0; i < value.size(); i++)
        result[i] = value[i] - m_value;

    return result;
}

Vector2D Vector2D::operator*(const Vector2D &m_value) const
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

Vector2D Vector2D::operator/(const Vector2D &m_value) const
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

Vector2D Vector2D::operator*(const float &m_value) const
{
    Vector2D result;
    for (size_t i = 0; i < value.size(); i++)
        result[i] = value[i] * m_value;

    return result;
}

Vector2D Vector2D::operator/(const float &m_value) const
{
    Vector2D result;
    for (size_t i = 0; i < value.size(); i++)
        result[i] = value[i] / m_value;

    return result;
}

bool Vector2D::operator==(const Vector2D &m_value) const
{
    if (shape() != m_value.shape())
        return false;

    for (size_t i = 0; i < value.size(); i++)
        if (value[i] != m_value[i])
            return false;

    return true;
}

bool Vector2D::operator!=(const Vector2D &m_value) const
{
    return !(*this == m_value);
}

Vector1D Vector2D::pop(const Vector2DAxis_t &axis, const size_t &index)
{
    switch (axis)
    {
        case Vector2DAxis_t::X:
        {
            Vector1D result;
            for (size_t i = 0; i < shape().y; i++)
                result.push(value[i].pop(index));

            return result;
        }
        case Vector2DAxis_t::Y:
        {
            Vector1D result = value.at(index);
            value.erase(value.begin() + index);

            return result;
        }
    }
}

void Vector2D::push(const Vector1D &m_value, const Vector2DAxis_t &m_axis)
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
                value[i].push(m_value.at(i));

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

void Vector2D::push(const Vector2D &m_value, const Vector2DAxis_t &m_axis)
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

void Vector2D::clear()
{
    value.clear();
}

void Vector2D::resize(const Vector2DSize_t &m_size)
{
    value.resize(m_size.y);
    for (size_t i = 0; i < m_size.y; i++)
        value[i].resize({m_size.x});
}

void Vector2D::resize(const Vector2DSize_t &m_size, const float &m_value)
{
    value.resize(m_size.y);
    for (size_t i = 0; i < m_size.y; i++)
        value[i].resize({m_size.x}, m_value);
}

Vector2D Vector2D::slice(const Vector2DSize_t &begin, const Vector2DSize_t &end) const
{
    Vector2D result;
    result.resize({end.x - begin.x, end.y - begin.y});
    for (size_t i = 0; i < result.shape().y; i++)
    {
        for (size_t j = 0; j < result.shape().x; j++)
        {
            result.at({i, j}) = value[i + begin.y].at(j + begin.x);
        }
    }
    return result;
}

Vector2D Vector2D::map(const std::function<float(float)> &m_function) const
{
    Vector2D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i].map(m_function));

    return result;
}

bool Vector2D::checkIsValid() const
{
    Vector1DSize_t size = value[0].shape();
    for (size_t i = 1; i < shape().y; i++)
        if (value[i].shape() != size)
            return false;

    return true;
}

Vector1D Vector2D::toVector1D() const
{
    Vector1D result;
    for (size_t i = 0; i < shape().y; i++)
        result.push(value[i]);

    return result;
}

Vector2D Vector2D::transpose() const
{
    Vector2D result;
    result.resize({ shape().x, shape().y });
    for (size_t i = 0; i < shape().y; i++)
        for (size_t j = 0; j < shape().x; j++)
            result.value[j].at(i) = value[i].at(j);

    return result;
}

Vector2DSize_t Vector2D::shape()
{
    return {value[0].shape().x, value.size()};
}
