#include "Vector3D.h"

template <typename T>
Vector3D<T>::Vector3D(const std::vector<std::vector<std::vector<T>>>& m_value)
{
    for(size_t i = 0; i < m_value.size(); i++)
        value.push(m_value[i]);
}

template <typename T>
Vector3D<T>::Vector3D(const Vector1D<Vector1D<Vector1D<T>>>& m_value)
{
    for(size_t i = 0; i < m_value.size(); i++)
        value.push(m_value[i]);
}

template <typename T>
Vector3D<T>::Vector3D(const Vector3D& m_value)
{
    value = m_value.value;
}

template <typename T>
T Vector3D<T>::at(const Vector3DSize_t &m_index) const
{
    if(m_index.x >= shape().x || m_index.y >= shape().y || m_index.z >= shape().z)
        throw std::out_of_range("Vector3D::at: Index out of range");

    return value[m_index.z][m_index.y][m_index.x];
}

template <typename T>
Vector3D<T> &Vector3D<T>::operator=(const Vector3D<T> &m_value)
{
    value = m_value.value;
    return *this;
}

template <typename T>
Vector3D<T> &Vector3D<T>::operator+=(const Vector3D<T> &m_value)
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator+=: shape mismatch");

    value += m_value.value;
    return *this;
}

template <typename T>
Vector3D<T> &Vector3D<T>::operator-=(const Vector3D<T> &m_value)
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator-=: shape mismatch");

    value -= m_value.value;
    return *this;
}

template <typename T>
Vector3D<T> &Vector3D<T>::operator+=(const T &m_value)
{
    value += m_value;
    return *this;
}

template <typename T>
Vector3D<T> &Vector3D<T>::operator-=(const T &m_value)
{
    value -= m_value;
    return *this;
}

template <typename T>
Vector3D<T> &Vector3D<T>::operator*=(const Vector3D<T> &m_value)
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator*=: shape mismatch");

    value *= m_value.value;
    return *this;
}

template <typename T>
Vector3D<T> &Vector3D<T>::operator/=(const Vector3D<T> &m_value)
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator/=: shape mismatch");

    value /= m_value.value;
    return *this;
}

template <typename T>
Vector3D<T> Vector3D<T>::operator+(const Vector3D<T> &m_value) const
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator+=: shape mismatch");

    return Vector3D<T>(value + m_value.value);
}

template <typename T>
Vector3D<T> Vector3D<T>::operator-(const Vector3D<T> &m_value) const
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator-=: shape mismatch");

    return Vector3D<T>(value - m_value.value);
}

template <typename T>
Vector3D<T> Vector3D<T>::operator+(const T &m_value) const
{
    return Vector3D<T>(value + m_value);
}

template <typename T>
Vector3D<T> Vector3D<T>::operator-(const T &m_value) const
{
    return Vector3D<T>(value - m_value);
}

template <typename T>
Vector3D<T> Vector3D<T>::operator*(const Vector3D<T> &m_value) const
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator*=: shape mismatch");

    return Vector3D<T>(value * m_value.value);
}

template <typename T>
Vector3D<T> Vector3D<T>::operator/(const Vector3D<T> &m_value) const
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator/=: shape mismatch");

    return Vector3D<T>(value / m_value.value);
}

template <typename T>
Vector3D<T> Vector3D<T>::operator*(const T &m_value) const
{
    return Vector3D<T>(value * m_value);
}

template <typename T>
Vector3D<T> Vector3D<T>::operator/(const T &m_value) const
{
    return Vector3D<T>(value / m_value);
}

template <typename T>
bool Vector3D<T>::operator==(const Vector3D<T> &m_value) const
{
    if(shape() != m_value.shape())
        return false;

    for(size_t i = 0; i < value.size(); i++)
        if(value[i] != m_value.value[i])
            return false;

    return true;
}

template <typename T>
bool Vector3D<T>::operator!=(const Vector3D<T> &m_value) const
{
    return !(*this == m_value);
}

template <typename T>
Vector2D<T> Vector3D<T>::pop(const Vector3DAxis_t &m_axis)
{
    switch (m_axis)
    {
        case Vector3DAxis_t::X
        {
            Vector2D<T> result(shape().y, shape().z);
            for(size_t z = 0; z < shape().z; z++)
                for(size_t y = 0; y < shape().y; y++)
                    result[z][y] = value[z][y].pop();

            return result;
        }
        case Vector3DAxis_t::Y
        {
            Vector2D<T> result(shape().x, shape().z);
            for(size_t z = 0; z < shape().z; z++)
                result[z] = value[z].pop();

            return result;
        }
        case Vector3DAxis_t::Z
        {
            Vector2D<T> result(shape().x, shape().y);       
            result = value.pop();

            return result;
        }
        default:
            throw std::invalid_argument("Vector3D::pop: Invalid axis");
    }
}

template <typename T>
void push(const Vector2D<T> &m_value, const Vector3DAxis_t &m_axis)
{
    switch (m_axis)
    {
        case Vector3DAxis_t::X
        {
            if(m_value.shape().y != shape().y || m_value.shape().z != shape().z)
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            for(size_t z = 0; z < shape().z; z++)
                for(size_t y = 0; y < shape().y; y++)
                    value[z][y].push(m_value[z][y]);

            break;
        }
        case Vector3DAxis_t::Y
        {
            if(m_value.shape().x != shape().x || m_value.shape().z != shape().z)
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            for(size_t z = 0; z < shape().z; z++)
                value[z].push(m_value[z]);

            break;
        }
        case Vector3DAxis_t::Z
        {
            if(m_value.shape().x != shape().x || m_value.shape().y != shape().y)
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            value.push(m_value);

            break;
        }
        default:
            throw std::invalid_argument("Vector3D::push: Invalid axis");
    }
}

template <typename T>
void push(const Vector3D<T> &m_value, const Vector3DAxis_t &m_axis)
{
    switch (m_axis)
    {
        case Vector3DAxis_t::X
        {
            if(m_value.shape().y != shape().y || m_value.shape().z != shape().z)
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            for(size_t z = 0; z < shape().z; z++) {
                for(size_t y = 0; y < shape().y; y++) {
                    value[z][y].push(m_value[z][y]);
                }
            }

            break;
        }
        case Vector3DAxis_t::Y
        {   
            if(m_value.shape().x != shape().x || m_value.shape().z != shape().z)
                throw std::invalid_argument("Vector3D::push: Invalid shape");
            
            for(size_t z = 0; z < shape().z; z++)
                value[z].push(m_value[z]);

            break;
        }
        case Vector3DAxis_t::Z
        {
            if(m_value.shape().x != shape().x || m_value.shape().y != shape().y)
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            value.push(m_value);

            break;
        }
        default:
            throw std::invalid_argument("Vector3D::push: Invalid axis");
    }
}