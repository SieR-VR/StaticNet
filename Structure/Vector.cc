#include "Vector.h"

//--------------------------------------------------
// Constructor and destructor
//--------------------------------------------------

template <class T, size_t N>
Vector<T, N>::Vector(void)
{
    data = (Vector<T, N-1> *)malloc(sizeof(Vector<T, N-1>) * 0);
}

template <class T, size_t N>
Vector<T, N>::Vector(const Vector<T, N> &v)
{
    for(size_t i = 0; i < v.length; i++)
        data[i].~Vector();

    free(data);
    data = new Vector<T, N-1>[v.length];
    length = v.length;

    for(size_t i = 0; i < v.length; i++)
        data[i] = v[i];
}

template <class T, size_t N>
Vector<T, N>::~Vector()
{
    for(size_t i = 0; i < length; i++)
        data[i].~Vector();

    free(data);
    data = nullptr;
}

//--------------------------------------------------
// Member access function
//--------------------------------------------------

template <class T, size_t N>
Vector<T, N-1> &Vector<T, N>::operator[](const size_t &i)
{
    if(!length)
        throw std::runtime_error("Vector::at: vector doesn't have any value!");

    if(i >= length || i < 0) 
        throw std::out_of_range("Vector::operator[]: can't access out of range index!");

    return &data[i];
}

template <class T, size_t N>
Vector<T, N-1> Vector<T, N>::at(const size_t &i) const
{
    if(!length)
        throw std::runtime_error("Vector::at: vector doesn't have any value!");

    if(i >= length || i < 0) 
        throw std::out_of_range("Vector::at: can't access out of range index!");

    return data[i];
}

template <class T, size_t N>
Vector<T, N-1> Vector<T, N>::front() const
{
    if(!length)
        throw std::runtime_error("Vector::at: vector doesn't have any value!");

    return data[0];
}

template <class T, size_t N>
Vector<T, N-1> Vector<T, N>::back() const
{
    if(!length)
        throw std::runtime_error("Vector::at: vector doesn't have any value!");

    return data[length - 1];
}

//--------------------------------------------------
// Assignment and compound assignment operator
//--------------------------------------------------

template <class T, size_t N>
Vector<T, N> &Vector<T, N>::operator=(const Vector<T, N> &v)
{
    if(data) free(data);

    data = (Vector<T, N-1> *)malloc(sizeof(Vector<T, N-1>) * v.length);
    length = v.length;

    for(size_t i = 0; i < length; i++)
        data[i] = v.at(i);
}

template <class T, size_t N>
Vector<T, N> &Vector<T, N>::operator+=(const Vector<T, N> &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] += v.at(i);
}

template <class T, size_t N>
Vector<T, N> &Vector<T, N>::operator-=(const Vector<T, N> &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] -= v.at(i);
}

template <class T, size_t N>
Vector<T, N> &Vector<T, N>::operator*=(const T &s)
{
    for(size_t i = 0; i < length; i++)
        data[i] *= s;
}

template <class T, size_t N>
Vector<T, N> &Vector<T, N>::operator/=(const T &s)
{
    for(size_t i = 0; i < length; i++)
        data[i] /= s;
}

//--------------------------------------------------
// Calculation operator
//--------------------------------------------------

template <class T, size_t N>
Vector<T, N> Vector<T, N>::operator+(const Vector<T, N> &v) const
{
    Vector<T, N> result = *this;
    result += v;
    return result;
}

template <class T, size_t N>
Vector<T, N> Vector<T, N>::operator-(const Vector<T, N> &v) const
{
    Vector<T, N> result = *this;
    result -= v;
    return result;
}

template <class T, size_t N>
Vector<T, N> Vector<T, N>::operator*(const T &s) const
{
    Vector<T, N> result = *this;
    result *= s;
    return result;
}

template <class T, size_t N>
Vector<T, N> Vector<T, N>::operator/(const T &s) const
{
    Vector<T, N> result = *this;
    result /= s;
    return result;
}

//--------------------------------------------------
// Comparison operator
//--------------------------------------------------

template <class T, size_t N>
bool Vector<T, N>::operator==(const Vector<T, N> &v) const
{
    bool result = true;

    for(size_t i = 0; i < v.length; i++)
        if(data.at(i) != v.at(i)) result = false;

    return result;
}

template <class T, size_t N>
bool Vector<T, N>::operator!=(const Vector<T, N> &v) const
{
    return !(*this == v);
}

//--------------------------------------------------
// Map function
//--------------------------------------------------

template <class T, size_t N>
Vector<T, N> Vector<T, N>::map(std::function<T(T)> f) const
{
    Vector<T, N> result = *this;

    for(int i = 0; i < length; i++)
        result[i] = result[i].map(f);

    return result;
}

//--------------------------------------------------
// Push and pop function
//--------------------------------------------------

template <class T, size_t N>
void Vector<T, N>::push_back(const Vector<T, N-1> &v)
{
    if(length && data[0].shape() != v.shape())
        throw std::length_error("Vector::push_back: shape mismatch");

    data = realloc(data, sizeof(Vector<T, N-1>) * (length + 1));
    length++;

    data[length - 1] = v;
}

template <class T, size_t N>
Vector<T, N - 1> Vector<T, N>::pop_back(void)
{
    Vector<T, N-1> result = data[length - 1];
    data[length - 1].~Vector();

    data = realloc(data, sizeof(Vector<T, N-1>) * (length - 1));
    length--;

    return result;
}

template <class T, size_t N>
void Vector<T, N>::push_front(const Vector<T, N-1> &v)
{
    if(length && data[0].shape() != v.shape())
        throw std::length_error("Vector::push_back: shape mismatch");

    data = realloc(data, sizeof(Vector<T, N-1>) * (length + 1));
    length++;

    memmove(data[1], data[0], sizeof(Vector<T, N-1> *) * (length - 1));

    data[0] = v;
}

template <class T, size_t N>
Vector<T, N - 1> Vector<T, N>::pop_front(void)
{
    Vector<T, N-1> result = data[0];
    data[0].~Vector();
    memmove(data[0], data[1], sizeof(Vector<T, N-1> *) * (length - 1));

    data = realloc(data, sizeof(Vector<T, N-1>) * (length - 1));
    length--;

    return result;
}

//--------------------------------------------------
// Slice
//--------------------------------------------------

template <class T, size_t N>
Vector<T, N> Vector<T, N>::slice(const Vector<size_t, 1> &start, const Vector<size_t, 1> &end) const
{
    if(N != start.length || N != end.length)
        throw std::length_error("Vector::slice: Dimension size mismatch");

    bool flag = true;
    for(size_t i = 0; i < start.length; i++)
        if(start.at(i) > end.at(i)) flag = false;

    if(!flag)
        throw std::length_error("Vector::slice: Some member of start is bigger than end's one");

    Vector<T, N> result;
    Vector<size_t, 1> start_copy = start, end_copy = end;

    for(size_t i = start_copy.back(); i < end_copy.back(); i++)
        result.push_back(at(i).slice(start_copy.pop_back(), end_copy.pop_back()));

    return result;
}

//--------------------------------------------------
// Clear and resize
//--------------------------------------------------

template <class T, size_t N>
void Vector<T, N>::clear(void)
{
    for(size_t i = 0; i < length; i++)
        data[i].clear();

    free(data);
    data = (Vector<T, N-1> *)malloc(sizeof(Vector<T, N-1>) * 0);
}

template <class T, size_t N>
void Vector<T, N>::resize(const Vector<size_t, 1> &n, const T &s)
{
    if(N != n.length)
        throw std::length_error("Vector::resize: Dimension size mismatch");

    Vector<size_t, 1> n_copy = n;

    clear();
    data = realloc(data, sizeof(Vector<T, N-1> *) * n_copy.back());
    length = n_copy.pop_back();

    for(size_t i = 0; i < length; i++)
        data[i].resize(n_copy, s);
}

//--------------------------------------------------
// Shape
//--------------------------------------------------

template <class T, size_t N>
Vector<size_t, 1> Vector<T, N>::shape(void) const
{
    return data.at(0).shape().push_back(length);
}

//--------------------------------------------------
// Vector<T, 1>
//--------------------------------------------------

//--------------------------------------------------
// Constructor and destructor
//--------------------------------------------------

template <class T>
Vector<T, 1>::Vector(void)
{
    data = (T *)malloc(sizeof(T) * 0);
    length = 0;
}

template <class T>
Vector<T, 1>::~Vector(void)
{
    free(data);
    data = nullptr;
    length = 0;
}

//--------------------------------------------------
// Member access function
//--------------------------------------------------

template <class T>
T &Vector<T, 1>::operator[](const size_t &i)
{
    if(!length)
        throw std::runtime_error("Vector::at: vector doesn't have any value!");

    if(i >= length || i < 0) 
        throw std::out_of_range("Vector::operator[]: can't access out of range index!");

    return &data[i];
}

template <class T>
T Vector<T, 1>::at(const size_t &i) const
{
    if(!length)
        throw std::runtime_error("Vector::at: vector doesn't have any value!");

    if(i >= length || i < 0) 
        throw std::out_of_range("Vector::at: can't access out of range index!");

    return data[i];
}

template <class T>
T Vector<T, 1>::front(void) const
{
    if(!length)
        throw std::runtime_error("Vector::at: vector doesn't have any value!");

    return data[0];
}

template <class T>
T Vector<T, 1>::back(void) const
{
    if(!length)
        throw std::runtime_error("Vector::at: vector doesn't have any value!");

    return data[length - 1];
}

//--------------------------------------------------
// Assignment and compound assignment operator
//--------------------------------------------------
template <class T>
Vector<T, 1> &Vector<T, 1>::operator=(const Vector<T, 1> &v)
{
    if(data) free(data);

    data = (T *)malloc(sizeof(T) * v.length);
    length = v.length;

    for(size_t i = 0; i < length; i++)
        data[i] = v.at(i);
}

template <class T>
Vector<T, 1> &Vector<T, 1>::operator+=(const Vector<T, 1> &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] += v.at(i);
}

template <class T>
Vector<T, 1> &Vector<T, 1>::operator-=(const Vector<T, 1> &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] -= v.at(i);
}

template <class T>
Vector<T, 1> &Vector<T, 1>::operator*=(const T &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] *= v;
}

template <class T>
Vector<T, 1> &Vector<T, 1>::operator/=(const T &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] /= v;
}

//--------------------------------------------------
// Calculation operator
//--------------------------------------------------

template <class T>
Vector<T, 1> &Vector<T, 1>::operator+=(const Vector<T, 1> &v)
{
    Vector<T, 1> result = *this;

    for(size_t i = 0; i < length; i++)
        data[i] += v.at(i);
}

template <class T>
Vector<T, 1> &Vector<T, 1>::operator-=(const Vector<T, 1> &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] -= v.at(i);
}

template <class T>
Vector<T, 1> &Vector<T, 1>::operator*=(const T &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] *= v;
}

template <class T>
Vector<T, 1> &Vector<T, 1>::operator/=(const T &v)
{
    for(size_t i = 0; i < length; i++)
        data[i] /= v;
}

//--------------------------------------------------
// Calculation operator
//--------------------------------------------------

template <class T>
Vector<T, 1> Vector<T, 1>::operator+(const Vector<T, 1> &v) const
{
    Vector<T, 1> result = *this;
    result += v;
    return result;
}

template <class T>
Vector<T, 1> Vector<T, 1>::operator-(const Vector<T, 1> &v) const
{
    Vector<T, 1> result = *this;
    result -= v;
    return result;
}

template <class T>
Vector<T, 1> Vector<T, 1>::operator*(const T &v) const
{
    Vector<T, 1> result = *this;
    result *= v;
    return result;
}

template <class T>
Vector<T, 1> Vector<T, 1>::operator/(const T &v) const
{
    Vector<T, 1> result = *this;
    result /= v;
    return result;
}

//--------------------------------------------------
// Comparison operator
//--------------------------------------------------

template <class T>
bool Vector<T, 1>::operator==(const Vector<T, 1> &v) const
{
    bool result = true;

    for(size_t i = 0; i < length; i++)
        if(at(i) != v.at(i)) result = false;

    return result;
}

template <class T>
bool Vector<T, 1>::operator==(const Vector<T, 1> &v) const
{
    return !(*this == v);
}

//--------------------------------------------------
// Map function
//--------------------------------------------------

template <class T>
Vector<T, 1> Vector<T, 1>::map(std::function<T(T)> f) const
{
    Vector<T, 1> result = *this;

    for(int i = 0; i < length; i++)
        result[i] = f(result[i]);

    return result;
}

//--------------------------------------------------
// Push and pop function
//--------------------------------------------------

template <class T>
void Vector<T, 1>::push_back(const T &v)
{
    data = realloc(data, sizeof(T) * (length + 1));
    length++;

    data[length - 1] = v;
}

template <class T>
T Vector<T, 1>::pop_back(void)
{
    T result = data[length - 1];

    data = realloc(data, sizeof(T) * (length - 1));
    length--;

    return result;
}

template <class T>
void Vector<T, 1>::push_front(const T &v)
{
    data = realloc(data, sizeof(T) * (length + 1));
    length++;

    memmove(data[1], data[0], sizeof(T *) * (length - 1));
    data[0] = v;
}

template <class T>
T Vector<T, 1>::pop_front(void)
{
    T result = data[0];
    memmove(data[0], data[1], sizeof(T *) * (length - 1));

    data = realloc(data, sizeof(T) * (length - 1));
    length--;

    return result;
}

//--------------------------------------------------
// Slice
//--------------------------------------------------

template <class T>
Vector<T, 1> Vector<T, 1>::slice(const Vector<size_t, 1> &start, const Vector<size_t, 1> &end) const
{
    if(1 != start.length || 1 != end.length)
        throw std::length_error("Vector::slice: Dimension size mismatch");

    bool flag = start[0] > end[0];
    if(flag)
        throw std::length_error("Vector::slice: Some member of start is bigger than end's one");

    Vector<T, 1> result;
    for(size_t i = start[0]; i < end[0]; i++)
        result.push_back(at(i));

    return result;
}

template <class T>
Vector<T, 1> Vector<T, 1>::slice(const size_t &start, const size_t &end) const
{
    bool flag = start[0] > end[0];
    if(flag)
        throw std::length_error("Vector::slice: Some member of start is bigger than end's one");

    Vector<T, 1> result;
    for(size_t i = start; i < end; i++)
        result.push_back(at(i));

    return result;
}

//--------------------------------------------------
// Clear and resize
//--------------------------------------------------

template <class T>
void Vector<T, 1>::clear(void)
{
    free(data);
    data = (T *)malloc(sizeof(T) * 0);
}

template <class T>
void Vector<T, 1>::resize(const Vector<size_t, 1> &n, const T &s)
{
    if(n.length != 1)
        throw std::length_error("Vector::resize: Dimension size mismatch");

    clear();
    data = realloc(data, sizeof(T) * n.back());
    length = n.back();

    for(size_t i = 0; i < length; i++)
        data[i] = s;
}

template <class T>
void Vector<T, 1>::resize(const size_t &n, const T &s)
{
    clear();
    data = realloc(data, sizeof(T) * n);
    length = n;

    for(size_t i = 0; i < length; i++)
        data[i] = s;
}

//--------------------------------------------------
// Utility
//--------------------------------------------------

template <class T>
T Vector<T, 1>::dot(const Vector<T, 1> &v) const
{
    Vector<T, 1> result = *this;
    for(int i = 0; i < length; i++)
        result[i] *= v.at(i);

    return result.sum();
}

template <class T>
T Vector<T, 1>::sum(void) const
{
    T result = 0;

    for(int i = 0; i < length; i++)
        result += data[i];

    return result;
}

//--------------------------------------------------
// Shape
//--------------------------------------------------

template <class T>
Vector<size_t, 1> Vector<T, 1>::shape(void) const
{
    Vector<size_t, 1> result;
    result.push_back(length);
    return result;
}