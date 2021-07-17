#include "Vector1D.h"

template <typename T>
Vector1D<T>::Vector1D(const std::vector<T> &m_value)
{
    value = m_value;
}

template <typename T>
Vector1D<T>::Vector1D(const Vector1D<T> &m_value)
{
    value = m_value.value;
}

template <typename T>
Vector1D<T>::~Vector1D()
{
    value.clear();
}

template <typename T>
T &Vector1D<T>::operator[](const size_t &i)
{
    return value[i];
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator=(const Vector1D<T> &m_value)
{
    *this.value = m_value.value;
    return *this;
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator+=(const Vector1D<T> &m_value)
{
    if(this->size() != m_value.size()) {
        throw std::runtime_error("Vector1D: Cannot add Vector1D with different size.");
        return *this;
    }

    for (size_t i = 0; i < value.size(); i++)
        value[i] += m_value[i];

    return *this;
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator-=(const Vector1D<T> &m_value)
{
    if(this->size() != m_value.size()) {
        throw std::runtime_error("Vector1D: Cannot subtract Vector1D with different size.");
        return *this;
    }

    for (size_t i = 0; i < value.size(); i++)
        value[i] -= m_value[i];

    return *this;
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator+=(const T &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] += m_value;

    return *this;
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator-=(const T &m_value)
{
    for (size_t i = 0; i < value.size(); i++)
        value[i] -= m_value;

    return *this;
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator*=(const Vector1D<T> &m_value)
{
    if(this->size() != m_value.size()) {
        throw std::runtime_error("Vector1D: Cannot multiply Vector1D with different size.");
        return *this;
    }

    for (int i = 0; i < value.size(); i++)
        value[i] *= m_value.value[i];

    return *this;
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator/=(const Vector1D<T> &m_value)
{
    if(this->size() != m_value.size()) {
        throw std::runtime_error("Vector1D: Cannot divide Vector1D with different size.");
        return *this;
    }

    for (int i = 0; i < value.size(); i++)
        value[i] /= m_value.value[i];

    return *this;
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator*=(const T &m_value)
{
    for (auto k : value)
        k *= m_value;

    return *this;
}

template <typename T>
Vector1D<T> &Vector1D<T>::operator/=(const T &m_value)
{
    for (auto k : value)
        k /= m_value;

    return *this;
}

template <typename T>
Vector1D<T> Vector1D<T>::operator+(const Vector1D<T> &m_value) const
{
    if(this->size() != m_value.size()) {
        throw std::runtime_error("Vector1D: Cannot add Vector1D with different size.");
        return Vector1D<T>(value);
    }

    Vector1D<T> result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] + m_value[i]);

    return result;
}

template <typename T>
Vector1D<T> Vector1D<T>::operator-(const Vector1D<T> &m_value) const
{
    if(this->size() != m_value.size()) {
        throw std::runtime_error("Vector1D: Cannot subtract Vector1D with different size.");
        return Vector1D<T>(value);
    }

    Vector1D<T> result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] - m_value[i]);

    return result;
}

template <typename T>
Vector1D<T> Vector1D<T>::operator+(const T &m_value) const
{
    Vector1D<T> result;
    for (auto k : value)
    {
        result.value.push_back(k + m_value);
    }
    return result;
}

template <typename T>
Vector1D<T> Vector1D<T>::operator-(const T &m_value) const
{
    Vector1D<T> result;
    for (auto k : value)
    {
        result.value.push_back(k - m_value);
    }
    return result;
}

template <typename T>
Vector1D<T> Vector1D<T>::operator*(const Vector1D<T> &m_value) const
{
    if(this->size() != m_value.size()) {
        throw std::runtime_error("Vector1D: Cannot multiply Vector1D with different size.");
        return Vector1D<T>(value);
    }

    Vector1D<T> result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] * m_value[i]);

    return result;
}

template <typename T>
Vector1D<T> Vector1D<T>::operator/(const Vector1D<T> &m_value) const
{
    if(this->size() != m_value.size()) {
        throw std::runtime_error("Vector1D: Cannot divide Vector1D with different size.");
        return Vector1D<T>(value);
    }

    Vector1D<T> result;
    for (size_t i = 0; i < value.size(); i++)
        result.push(value[i] / m_value[i]);

    return result;
}

template <typename T>
Vector1D<T> Vector1D<T>::operator*(const T &m_value) const
{
    Vector1D<T> result;
    for (auto k : value)
    {
        result.value.push_back(k * m_value);
    }
    return result;
}

template <typename T>
Vector1D<T> Vector1D<T>::operator/(const T &m_value) const
{
    Vector1D<T> result;
    for (auto k : value)
    {
        result.value.push_back(k / m_value);
    }
    return result;
}

template <typename T>
bool Vector1D<T>::operator==(const Vector1D<T> &m_value) const
{
    if (value.size() != m_value.value.size()) {
        throw std::runtime_error("Vector1D::operator==: size of vectors are different");
        return false;
    }

    for (size_t i = 0; i < value.size(); i++)
    {
        if (value[i] != m_value[i])
            return false;
    }
    return true;
}

template <typename T>
bool Vector1D<T>::operator!=(const Vector1D<T> &m_value) const
{
    return !(*this == m_value);
}

template <typename T>
T Vector1D<T>::pop(const size_t &i)
{
    if(i >= value.size() || i < -1) {
        throw std::out_of_range("Vector1D::pop: index out of range");
        return;
    }

    T result;
    if(i == -1) {
        result = value.back();
        value.pop_back();
    } else {
        result = value[i];
        value.erase(value.begin() + i);
    }
    return result;
}

template <typename T>
void Vector1D<T>::push(const T &m_value)
{
    value.push_back(m_value);
}

template <typename T>
void Vector1D<T>::push(const Vector1D<T> &m_value)
{
    for (auto k : m_value.value)
    {
        value.push_back(k);
    }
}

template <typename T>
void Vector1D<T>::clear()
{
    value.clear();
}

template <typename T>
void Vector1D<T>::resize(const Vector1DSize_t &m_size) 
{
    value.resize(m_size.x);
}

template <typename T>
void Vector1D<T>::resize(const Vector1DSize_t &m_size, const T &m_value)
{
    value.resize(m_size.x, m_value);
}

template <typename T>
Vector1D<T> Vector1D<T>::slice(const Vector1DSize_t &begin, const Vector1DSize_t &end) const
{
    Vector1D<T> result;
    for(int i = begin.x; i < end.x; i++) 
    {
        result.value.push_back(value[i]);
    }
    return result;
}

template <typename T>
Vector1D<T> Vector1D<T>::map(const std::function<T(T)> &m_function) const
{
    Vector1D<T> result;
    for (auto k : value)
    {
        result.value.push_back(m_function(k));
    }
    return result;
}


template <typename T>
Vector1DSize_t Vector1D<T>::shape() const
{
    return { value.size() };
}