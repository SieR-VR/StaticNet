#include "Vector2D.h"

template <typename T>
Vector2D<T>::Vector2D(const std::vector<std::vector<T>> &m_value)
{
    for (const auto &row : m_value)
    {
        value.push(row);
    }
}

template <typename T>
Vector2D<T>::Vector2D(const Vector1D<Vector1D<T>> &m_value)
{
    value = m_value;
}

template <typename T>
Vector2D<T>::Vector2D(const Vector2D<T> &m_value)
{
    value = m_value.value;
}

template <typename T>
Vector2D<T>::~Vector2D() {}

template <typename T>
T Vector2D<T>::at(const Vector2DSize_t &index) const
{
    return value[index.y][index.x];
}

template <typename T>
Vector1D<T> Vector2D<T>::operator[](const size_t &index) const
{
    return Vector1D<T>(value[index]);
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator=(const Vector2D<T> &m_value)
{
    value = m_value.value;
    return *this;
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator+=(const Vector2D<T> &m_value)
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector2D::operator+=: shape mismatch");
        return *this;
    }

    value += m_value.value;
    return *this;
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator-=(const Vector2D<T> &m_value)
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector2D::operator-=: shape mismatch");
        return *this;
    }

    value -= m_value.value;
    return *this;
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator+=(const T &m_value)
{
    value += m_value;
    return *this;
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator-=(const T &m_value)
{
    value -= m_value;
    return *this;
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator*=(const Vector2D<T> &m_value)
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector2D::operator*=: shape mismatch");
        return *this;
    }

    value *= m_value.value;
    return *this;
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator/=(const Vector2D<T> &m_value)
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector2D::operator/=: shape mismatch");
        return *this;
    }

    value /= m_value.value;
    return *this;
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator*=(const T &m_value)
{
    value *= m_value;
    return *this;
}

template <typename T>
Vector2D<T> &Vector2D<T>::operator/=(const T &m_value)
{
    value /= m_value;
    return *this;
}

template <typename T>
Vector2D<T> Vector2D<T>::operator+(const Vector2D<T> &m_value) const
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector2D::operator+=: shape mismatch");
        return value;
    }

    return Vector2D<T>(value + m_value.value);
}

template <typename T>
Vector2D<T> Vector2D<T>::operator-(const Vector2D<T> &m_value) const
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector2D::operator-=: shape mismatch");
        return value;
    }

    return Vector2D<T>(value - m_value.value);
}

template <typename T>
Vector2D<T> Vector2D<T>::operator+(const T &m_value) const
{
    return Vector2D<T>(*this) += m_value;
}

template <typename T>
Vector2D<T> Vector2D<T>::operator-(const T &m_value) const
{
    return Vector2D<T>(*this) -= m_value;
}

template <typename T>
Vector2D<T> Vector2D<T>::operator*(const Vector2D<T> &m_value) const
{
    if(shape() != m_value.shape()) {
        throw std::invalid_argument("Vector2D::operator*=: shape mismatch");
        return value;
    }
    
    return Vector2D<T>(*this) *= m_value;
}

template <typename T>
Vector2D<T> Vector2D<T>::operator/(const Vector2D<T> &m_value) const
{
    return Vector2D<T>(*this) /= m_value;
}

template <typename T>
Vector2D<T> Vector2D<T>::operator*(const T &m_value) const
{
    return Vector2D<T>(*this) *= m_value;
}

template <typename T>
Vector2D<T> Vector2D<T>::operator/(const T &m_value) const
{
    return Vector2D<T>(*this) /= m_value;
}

template <typename T>
bool Vector2D<T>::operator==(const Vector2D<T> &m_value) const
{
    return value == m_value.value;
}

template <typename T>
bool Vector2D<T>::operator!=(const Vector2D<T> &m_value) const
{
    return value != m_value.value;
}

template <typename T>
Vector1D<T> Vector2D<T>::pop(const Vector2DAxis_t &axis, const size_t &index) 
{
    switch (axis)
    {
        case Vector2DAxis_t::X:
        {
            Vector1D<T> result;
            for(const auto &v : value)
                result.push_back(v.pop(index));

            return result;
        }
        case Vector2DAxis_t::Y:
        {
            return value.pop(index);
        }
    }
}

template <typename T>
void Vector2D<T>::push(const Vector1D<T> &m_value, const Vector2DAxis_t &m_axis)
{
    switch (m_axis)
    {
        case Vector2DAxis_t::X:
        {
            if(shape().y != m_value.size()) {
                throw std::invalid_argument("Vector2D::push: vector y size mismatch");
                return;
            }

            for (int i = 0; i < m_value.size(); i++)
                value[i].push(m_value[i]);
            
            break;
        }
        case Vector2DAxis_t::Y:
        {
            if(shape().x != m_value.size()) {
                throw std::invalid_argument("Vector2D::push: vector x size mismatch");
                return;
            }

            value.push(m_value);
            break;
        }
    }
}

template <typename T>
void Vector2D<T>::push(const Vector2D<T> &m_value, const Vector2DAxis_t &m_axis)
{
    switch (m_axis)
    {
        case Vector2DAxis_t::X:
        {
            if(shape().y != m_value.shape().y) {
                throw std::invalid_argument("Vector2D::push: vector y size mismatch");
                return;
            }

            for (int i = 0; i < m_value.shape().y; i++)
                value[i].push(m_value[i]);

            break;
        }
        case Vector2DAxis_t::Y:
        {
            if(shape().x != m_value.shape().x) {
                throw std::invalid_argument("Vector2D::push: vector x size mismatch");
                return;
            }

            value.push(m_value.value);

            break;
        }
    }
}

template <typename T>
void Vector2D<T>::clear() 
{
    value.clear();
}

template <typename T>
void Vector2D<T>::resize(const Vector2DSize_t &m_size) 
{
    value.resize(m_size.y);
    for (auto &row : value)
    {
        row.resize(m_size.x);
    }
}

template <typename T>
void Vector2D<T>::resize(const Vector2DSize_t &m_size, const T &m_value)
{
    value.resize(m_size.y);
    for (auto &row : value)
    {
        row.resize(m_size.x, m_value);
    }
}

template <typename T>
Vector2D<T> Vector2D<T>::slice(const Vector2DSize_t& begin, const Vector2DSize_t& end) const
{
    Vector2D<T> result;
    result.resize(end.x - begin.x, end.y - begin.y);
    for (int i = 0; i < result.shape().y; i++)
    {
        for (int j = 0; j < result.shape().x; j++)
        {
            result.value[i][j] = value[i + begin.y][j + begin.x];
        }
    }
    return result;
}

template <typename T>
Vector2D<T> Vector2D<T>::map(const std::function<T(T)>& m_function) const
{
    Vector2D<T> result;
    for(const auto &row : value)
        result.push(row.map(m_function));

    return result;
}

template <typename T>
bool Vector2D<T>::checkIsValid() const
{
    int i = value[0].size();
    for (const auto &row : value)
    {
        if (row.size() != i)
        {
            return false;
        }
    }
    return true;
}

template <typename T>
Vector1D<T> Vector2D<T>::toVector1D() const
{
    Vector1D<T> result;
    for (const auto &row : value)
    {
        result.push(row);
    }
    return result;
}

template <typename T>
Vector2D<T> Vector2D<T>::transpose() const
{
    Vector2D<T> result;
    result.resize(shape().x, shape().y));
    for (int i = 0; i < shape().y; i++)
    {
        for (int j = 0; j < shape().x; j++)
        {
            result.value[j][i] = value[i][j];
        }
    }
    return result;
}

template <typename T>
Vector2DSize_t Vector2D<T>::shape() const
{
    return { value.size(), value[0].size() };
}
