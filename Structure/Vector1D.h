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

class Vector1D
{
public:
    Vector1D(const std::vector<float> &m_value);
    Vector1D(const Vector1D &m_value);
    Vector1D();
    ~Vector1D();

    float operator[](const size_t &i);
    float &at(const size_t &i) const;

    Vector1D &operator=(const Vector1D &m_value);

    Vector1D &operator+=(const Vector1D &m_value);
    Vector1D &operator-=(const Vector1D &m_value);
    Vector1D &operator+=(const float &m_value);
    Vector1D &operator-=(const float &m_value);
    Vector1D &operator*=(const Vector1D &m_value);
    Vector1D &operator/=(const Vector1D &m_value);
    Vector1D &operator*=(const float &m_value);
    Vector1D &operator/=(const float &m_value);

    Vector1D operator+(const Vector1D &m_value) const;
    Vector1D operator-(const Vector1D &m_value) const;
    Vector1D operator+(const float &m_value) const;
    Vector1D operator-(const float &m_value) const;
    Vector1D operator*(const Vector1D &m_value) const;
    Vector1D operator/(const Vector1D &m_value) const;
    Vector1D operator*(const float &m_value) const;
    Vector1D operator/(const float &m_value) const;

    bool operator==(const Vector1D &m_value) const;
    bool operator!=(const Vector1D &m_value) const;

    float pop(const size_t &i = shape().x - 1);
    void push(const float &m_value);
    void push(const Vector1D &m_value);
    void clear();
    void resize(const Vector1DSize_t &m_size);
    void resize(const Vector1DSize_t &m_size, const float &m_value);

    Vector1D slice(const Vector1DSize_t &begin = { 0 }, const Vector1DSize_t &end = { value.size() }) const;
    Vector1D map(const std::function<float(float)> &m_function) const;

    static Vector1DSize_t shape();

private:
    static std::vector<float> value;
};

#endif
