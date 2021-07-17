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