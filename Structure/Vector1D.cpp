#include "Vector1D.h"

Vector1D::Vector1D(const std::vector<float> &m_value)
{
    value = m_value;
}

Vector1D::Vector1D(const Vector1D &m_value)
{
    value = m_value.value;
}

Vector1D::Vector1D()
{
    value.clear();
}

Vector1D::~Vector1D()
{
    value.clear();
}

float Vector1D::operator[](const size_t &i)
{
    if (i >= value.size())
    {
        throw std::out_of_range("Index out of range");
        return value[0];
    }
    
    return value[i];
}

float &Vector1D::at(const size_t &i) const
{
    if(i >= value.size())
    {
        throw std::out_of_range("Index out of range");
        return value[0];
    }
    
    return value[i];
}

Vector1D &Vector1D::operator=(const Vector1D &m_value)
{
    value = m_value.value;
    return *this;
}

Vector1D &Vector1D::operator+=(const Vector1D &m_value)
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector1D::operator+=: shape mismatch");
        return *this;
    }

    for (size_t i = 0; i < value.size(); i++)
        value[i] += m_value.at(i);

    return *this;
}

Vector1D &Vector1D::operator-=(const Vector1D &m_value)
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector1D::operator-=: shape mismatch");
        return *this;
    }

    for (size_t i = 0; i < value.size(); i++)
        value[i] -= m_value.at(i);

    return *this;
}

Vector1D &Vector1D::operator+=(const float &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] += m_value;

    return *this;
}

Vector1D &Vector1D::operator-=(const float &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] -= m_value;

    return *this;
}

Vector1D &Vector1D::operator*=(const Vector1D &m_value)
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector1D::operator*=: shape mismatch");
        return *this;
    }

    for (int i = 0; i < value.size(); i++)
        value[i] *= m_value.at(i);

    return *this;
}

Vector1D &Vector1D::operator/=(const Vector1D &m_value)
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector1D::operator/=: shape mismatch");
        return *this;
    }

    for (int i = 0; i < value.size(); i++)
        value[i] /= m_value.at(i);

    return *this;
}

Vector1D &Vector1D::operator*=(const float &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] *= m_value;

    return *this;
}

Vector1D &Vector1D::operator/=(const float &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] /= m_value;

    return *this;
}

Vector1D Vector1D::operator+(const Vector1D &m_value) const
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector1D::operator+=: shape mismatch");
        return Vector1D(value);
    }

    Vector1D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] + m_value.at(i));

    return result;
}

Vector1D Vector1D::operator-(const Vector1D &m_value) const
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector1D::operator-=: shape mismatch");
        return Vector1D(value);
    }

    Vector1D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] - m_value.at(i));

    return result;
}

Vector1D Vector1D::operator+(const float &m_value) const
{
    Vector1D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] + m_value);

    return result;
}

Vector1D Vector1D::operator-(const float &m_value) const
{
    Vector1D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] - m_value);

    return result;
}

Vector1D Vector1D::operator*(const Vector1D &m_value) const
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector1D::operator*=: shape mismatch");
        return Vector1D(value);
    }

    Vector1D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] * m_value.at(i));

    return result;
}

Vector1D Vector1D::operator/(const Vector1D &m_value) const
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector1D::operator/=: shape mismatch");
        return Vector1D(value);
    }

    Vector1D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] / m_value.at(i));

    return result;
}

Vector1D Vector1D::operator*(const float &m_value) const
{
    Vector1D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] * m_value);

    return result;
}

Vector1D Vector1D::operator/(const float &m_value) const
{
    Vector1D result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] / m_value);

    return result;
}

bool Vector1D::operator==(const Vector1D &m_value) const
{
    if (shape() != m_value.shape()) {
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

bool Vector1D::operator!=(const Vector1D &m_value) const
{
    return !(*this == m_value);
}

float Vector1D::pop(const size_t &i)
{
    if(i >= shape().x || i < -1) {
        throw std::out_of_range("Vector1D::pop: index out of range");
        return;
    }

    float result;
    result = value[i];
    value.erase(value.begin() + i);

    return result;
}

void Vector1D::push(const float &m_value)
{
    value.push_back(m_value);
}

void Vector1D::push(const Vector1D &m_value)
{
    for (size_t i = 0; i < m_value.shape().x; i++)
        value.push_back(m_value.at(i));
}

void Vector1D::clear()
{
    value.clear();
}

void Vector1D::resize(const Vector1DSize_t &m_size) 
{
    if(!m_size.checkIsValid()) {
        throw std::invalid_argument("Vector1D::resize: invalid size");
        return;
    }

    value.resize(m_size.x);
}

void Vector1D::resize(const Vector1DSize_t &m_size, const float &m_value)
{
    if(!m_size.checkIsValid()) {
        throw std::invalid_argument("Vector1D::resize: invalid size");
        return;
    }
    
    value.resize(m_size.x, m_value);
}

Vector1D Vector1D::slice(const Vector1DSize_t &begin, const Vector1DSize_t &end) const
{
    if(begin.x >= value.size() || end.x >= value.size()) {
        throw std::out_of_range("Vector1D::slice: index out of range");
        return Vector1D(value);
    }

    if(begin.x > end.x) {
        throw std::invalid_argument("Vector1D::slice: begin index is greater than end index");
        return Vector1D(value);
    }

    Vector1D result;
    for(int i = begin.x; i < end.x; i++) 
        result.push(value[i]);

    return result;
}

Vector1D Vector1D::map(const std::function<float(float)> &m_function) const
{
    Vector1D result;
    for(size_t i = 0; i < value.size(); i++)
        result.push(m_function(value[i]));

    return result;
}


Vector1DSize_t Vector1D::shape()
{
    return { value.size() };
}