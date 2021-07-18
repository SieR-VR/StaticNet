#include "Vector3D.h"

Vector3D::Vector3D(const std::vector<std::vector<std::vector<float>>>& m_value)
{
    for(size_t i = 0; i < m_value.size(); i++)
        value.push_back(m_value[i]);
}

Vector3D::Vector3D(const Vector3D& m_value)
{
    value = m_value.value;
}

Vector3D::Vector3D()
{
    value.clear();
}

Vector3D::~Vector3D()
{
    value.clear();
}

float &Vector3D::at(const Vector3DSize_t &m_index) const
{
    if(m_index.x >= shape().x || m_index.y >= shape().y || m_index.z >= shape().z)
        throw std::out_of_range("Vector3D::at: Index out of range");

    return value[m_index.z].at({ m_index.x, m_index.y });
}

Vector3D &Vector3D::operator=(const Vector3D &m_value)
{
    value = m_value.value;
    return *this;
}

Vector3D &Vector3D::operator+=(const Vector3D &m_value)
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator+=: shape mismatch");

    for(size_t i = 0; i < value.size(); i++)
        value[i] += m_value[i];

    return *this;
}

Vector3D &Vector3D::operator-=(const Vector3D &m_value)
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator-=: shape mismatch");

    for(size_t i = 0; i < value.size(); i++)
        value[i] -= m_value[i];

    return *this;
}

Vector3D &Vector3D::operator+=(const float &m_value)
{
    for(size_t i = 0; i < value.size(); i++)
        value[i] += m_value;

    return *this;
}

Vector3D &Vector3D::operator-=(const float &m_value)
{
    for(size_t i = 0; i < value.size(); i++)
        value[i] -= m_value;

    return *this;
}

Vector3D &Vector3D::operator*=(const Vector3D &m_value)
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator*=: shape mismatch");

    for(size_t i = 0; i < value.size(); i++)
        value[i] *= m_value[i];

    return *this;
}

Vector3D &Vector3D::operator/=(const Vector3D &m_value)
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator/=: shape mismatch");

    for(size_t i = 0; i < value.size(); i++)
        value[i] /= m_value[i];

    return *this;
}

Vector3D Vector3D::operator+(const Vector3D &m_value) const
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator+=: shape mismatch");

    Vector3D result;
    for(size_t i = 0; i < value.size(); i++)
        result[i] = value[i] + m_value[i];

    return result;
}

Vector3D Vector3D::operator-(const Vector3D &m_value) const
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator-=: shape mismatch");

    Vector3D result;
    for(size_t i = 0; i < value.size(); i++)
        result[i] = value[i] - m_value[i];

    return result;
}

Vector3D Vector3D::operator+(const float &m_value) const
{
    Vector3D result;
    for(size_t i = 0; i < value.size(); i++)
        result[i] = value[i] + m_value;

    return result;
}

Vector3D Vector3D::operator-(const float &m_value) const
{
    Vector3D result;
    for(size_t i = 0; i < value.size(); i++)
        result[i] = value[i] - m_value;

    return result;
}

Vector3D Vector3D::operator*(const Vector3D &m_value) const
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator*=: shape mismatch");

    Vector3D result;
    for(size_t i = 0; i < value.size(); i++)
        result[i] = value[i] * m_value[i];

    return result;
}

Vector3D Vector3D::operator/(const Vector3D &m_value) const
{
    if(shape() != m_value.shape())
        throw std::invalid_argument("Vector3D::operator/=: shape mismatch");

    Vector3D result;
    for(size_t i = 0; i < value.size(); i++)
        result[i] = value[i] / m_value[i];

    return result;
}

Vector3D Vector3D::operator*(const float &m_value) const
{
    Vector3D result;
    for(size_t i = 0; i < value.size(); i++)
        result[i] = value[i] * m_value;

    return result;
}

Vector3D Vector3D::operator/(const float &m_value) const
{
    Vector3D result;
    for(size_t i = 0; i < value.size(); i++)
        result[i] = value[i] / m_value;

    return result;
}

bool Vector3D::operator==(const Vector3D &m_value) const
{
    if(shape() != m_value.shape())
        return false;

    for(size_t i = 0; i < value.size(); i++)
        if(value[i] != m_value.value[i])
            return false;

    return true;
}

bool Vector3D::operator!=(const Vector3D &m_value) const
{
    return !(*this == m_value);
}

Vector2D Vector3D::pop(const Vector3DAxis_t &m_axis, const size_t &m_index)
{
    switch (m_axis)
    {
        case Vector3DAxis_t::X:
        {
            Vector2D result;
            result.resize({ shape().y, shape().z });

            for(size_t z = 0; z < shape().z; z++)
                for(size_t y = 0; y < shape().y; y++)
                    result[z].at(y) = value[z][y].pop();

            return result;
        }
        case Vector3DAxis_t::Y:
        {
            Vector2D result;
            result.resize({ shape().x, shape().z });

            for(size_t z = 0; z < shape().z; z++)
                result[z] = value[z].pop(Vector2DAxis_t::Y, m_index);

            return result;
        }
        case Vector3DAxis_t::Z:
        {
            Vector2D result;
            result.resize({ shape().x, shape().y });

            result = value.at(m_index);
            value.erase(value.begin() + m_index);

            return result;
        }
        default:
            throw std::invalid_argument("Vector3D::pop: Invalid axis");
    }
}

void Vector3D::push(const Vector2D &m_value, const Vector3DAxis_t &m_axis)
{
    switch (m_axis)
    {
        case Vector3DAxis_t::X:
        {
            if(m_value.shape() != Vector2DSize_t({ shape().y, shape().z }))
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            for(size_t z = 0; z < shape().z; z++)
                for(size_t y = 0; y < shape().y; y++)
                    value[z][y].push(m_value[z][y]);

            break;
        }
        case Vector3DAxis_t::Y:
        {
            if(m_value.shape() != Vector2DSize_t({ shape().x, shape().z }))
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            for(size_t z = 0; z < shape().z; z++)
                value[z].push(m_value[z]);

            break;
        }
        case Vector3DAxis_t::Z:
        {
            if(m_value.shape() != Vector2DSize_t({ shape().x, shape().y }))
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            value.push_back(m_value);

            break;
        }
        default:
            throw std::invalid_argument("Vector3D::push: Invalid axis");
    }
}

void Vector3D::push(const Vector3D &m_value, const Vector3DAxis_t &m_axis)
{
    switch (m_axis)
    {
        case Vector3DAxis_t::X:
        {
            if(Vector2DSize_t({ m_value.shape().y, m_value.shape().z}) != Vector2DSize_t({ shape().y, shape().z }))
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            for(size_t z = 0; z < shape().z; z++) {
                for(size_t y = 0; y < shape().y; y++) {
                    value[z][y].push(m_value[z][y]);
                }
            }

            break;
        }
        case Vector3DAxis_t::Y:
        {   
            if(Vector2DSize_t({ m_value.shape().x, m_value.shape().z}) != Vector2DSize_t({ shape().x, shape().z }))
                throw std::invalid_argument("Vector3D::push: Invalid shape");
            
            for(size_t z = 0; z < shape().z; z++)
                value[z].push(m_value[z]);

            break;
        }
        case Vector3DAxis_t::Z:
        {
            if(Vector2DSize_t({ m_value.shape().x, m_value.shape().y}) != Vector2DSize_t({ shape().x, shape().y }))
                throw std::invalid_argument("Vector3D::push: Invalid shape");

            for(size_t z = 0; z < m_value.shape().z; z++)
                value.push_back(m_value[z]);

            break;
        }
        default:
            throw std::invalid_argument("Vector3D::push: Invalid axis");
    }
}

void Vector3D::clear()
{
    value.clear();
}

void Vector3D::resize(const Vector3DSize_t &m_size)
{
    if(!m_size.checkIsValid())
        throw std::invalid_argument("Vector3D::resize: Invalid size");

    value.resize(m_size.z);
    for(size_t z = 0; z < m_size.z; z++)
        value[z].resize({ m_size.x, m_size.y });
}

Vector3D Vector3D::slice(const Vector3DSize_t &begin, const Vector3DSize_t &end) const
{
    if(!begin.checkIsValid() || !end.checkIsValid())
        throw std::invalid_argument("Vector3D::slice: Invalid range");

    if(begin.z > end.z || begin.y > end.y || begin.x > end.x)
        throw std::invalid_argument("Vector3D::slice: Invalid range");

    Vector3D result;
    result.resize({ end.z - begin.z, end.y - begin.y, end.x - begin.x });
    for(size_t z = 0; z < result.shape().z; z++)
        for(size_t y = 0; y < result.shape().y; y++)
            for(size_t x = 0; x < result.shape().x; x++)
                result[z].at({x, y}) = value[z + begin.z].at({y + begin.y, x + begin.x});

    return result;
}

Vector3D Vector3D::map(const std::function<float(float)> &m_function) const
{
    Vector3D result;
    result.resize(shape());
    for(size_t z = 0; z < shape().z; z++)
        result[z] = value[z].map(m_function);

    return result;
}

bool Vector3D::checkIsValid() const
{
    for(size_t z = 0; z < shape().z; z++)
        if(!value[z].checkIsValid())
            return false;

    return true;
}

Vector2D Vector3D::toVector2D() const
{
    Vector2D result;
    for(size_t z = 0; z < shape().z; z++)
        result.push(value[z]);

    return result;
}

Vector2D Vector3D::getVector2DByAxis(const Vector3DAxis_t &m_axis, const size_t &m_index) const
{
    switch (m_axis)
    {
        case Vector3DAxis_t::X:
        {
            if(m_index >= shape().x)
                throw std::invalid_argument("Vector3D::getVector2DByAxis: Invalid index");

            Vector2D result;
            for(size_t z = 0; z < shape().z; z++)
                result.push(value[z].getVector1DByAxis(Vector2DAxis_t::X, m_index));

            return result;
        }
        case Vector3DAxis_t::Y:
        {
            if(m_index >= shape().y)
                throw std::invalid_argument("Vector3D::getVector2DByAxis: Invalid index");

            Vector2D result;
            for(size_t z = 0; z < shape().z; z++)
                result.push(value[z].getVector1DByAxis(Vector2DAxis_t::Y, m_index));

            return result;
        }
        case Vector3DAxis_t::Z:
        {
            if(m_index >= shape().z)
                throw std::invalid_argument("Vector3D::getVector2DByAxis: Invalid index");

            return value[m_index];
        }
        default:
            throw std::invalid_argument("Vector3D::getVector2DByAxis: Invalid axis");
    }
}

Vector3DSize_t Vector3D::shape()
{
    Vector3DSize_t result;
    result.x = value[0].shape().x;
    result.y = value[0].shape().y;
    result.z = value.size();
    return result;
}
