#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <deque>
#include <cstdlib>
#include <memory.h>
#include <stdexcept>
#include <functional>
#include <condition_variable>
#include <initializer_list>

#include "Tools/StackTrace.h"

namespace SingleNet
{
    template <class T, size_t N>
    class Vector
    {
    public:
        typedef typename std::conditional<N == 1, T, Vector<T, N - 1>>::type VectorT;
        typedef typename std::deque<VectorT>::iterator iterator;
        typedef typename std::deque<VectorT>::const_iterator const_iterator;

        //--------------------------------------------------
        // Constructors
        //--------------------------------------------------

        Vector(void)
        {
            data.clear();
            length = 0;
        }

        Vector(std::initializer_list<T> list)
        {
            data.resize(list.size());
            length = list.size();

            iterator i = data.begin();

            for (auto &e : list)
                *i++ = e;
        }

        Vector(const Vector<T, N> &v)
        {
            data.resize(v.length);
            length = v.length;

            for (size_t i = 0; i < v.length; i++)
                data[i] = v.at(i);
        }

        ~Vector()
        {
            clear();
        }

        //--------------------------------------------------
        // Member access function
        //--------------------------------------------------

        VectorT &operator[](const size_t &i)
        {
            if (!length)
                throw Tools::StackTrace("Vector::at: vector doesn't have any value!");

            if (i >= length || i < 0)
                throw Tools::StackTrace("Vector::operator[]: can't access out of range index!");

            return data[i];
        }

        VectorT at(const size_t &i) const
        {
            if (!length)
                throw Tools::StackTrace("Vector::at: vector doesn't have any value!");

            if (i >= length || i < 0)
                throw Tools::StackTrace("Vector::at: can't access out of range index!");

            return data[i];
        }

        VectorT front() const
        {
            if (!length)
                throw Tools::StackTrace("Vector::at: vector doesn't have any value!");

            return data[0];
        }

        VectorT back() const
        {
            if (!length)
                throw Tools::StackTrace("Vector::at: vector doesn't have any value!");

            return data[length - 1];
        }

        //--------------------------------------------------
        // Assignment and compound assignment operator
        //--------------------------------------------------

        Vector<T, N> &operator=(const Vector<T, N> &v)
        {
            clear();

            data = v.data;
            length = v.length;

            return *this;
        }

        Vector<T, N> &operator+=(const Vector<T, N> &v)
        {
            if (shape() != v.shape())
                throw Tools::StackTrace("Vector::operator+=: shape mismatch");

            for (size_t i = 0; i < length; i++)
                data[i] += v.at(i);

            return *this;
        }

        Vector<T, N> &operator-=(const Vector<T, N> &v)
        {
            if (shape() != v.shape())
                throw Tools::StackTrace("Vector::operator-=: shape mismatch");

            for (size_t i = 0; i < length; i++)
                data[i] -= v.at(i);

            return *this;
        }

        Vector<T, N> &operator*=(const T &s)
        {
            for (size_t i = 0; i < length; i++)
                data[i] *= s;

            return *this;
        }

        Vector<T, N> &operator/=(const T &s)
        {
            for (size_t i = 0; i < length; i++)
                data[i] /= s;

            return *this;
        }

        //--------------------------------------------------
        // Calculation operator
        //--------------------------------------------------

        Vector<T, N> operator+(const Vector<T, N> &v) const
        {
            if (shape() != v.shape())
                throw Tools::StackTrace("Vector::operator+: shape mismatch");

            Vector<T, N> result = *this;
            result += v;
            return result;
        }

        Vector<T, N> operator-(const Vector<T, N> &v) const
        {
            if (shape() != v.shape())
                throw Tools::StackTrace("Vector::operator-: shape mismatch");

            Vector<T, N> result = *this;
            result -= v;
            return result;
        }

        Vector<T, N> operator*(const T &s) const
        {
            Vector<T, N> result = *this;
            result *= s;
            return result;
        }

        Vector<T, N> operator/(const T &s) const
        {
            Vector<T, N> result = *this;
            result /= s;
            return result;
        }

        //--------------------------------------------------
        // Comparison operator
        //--------------------------------------------------

        bool operator==(const Vector<T, N> &v) const
        {
            if (length != v.length)
                throw Tools::StackTrace("Vector::operator==: shape mismatch");

            bool result = true;

            for (size_t i = 0; i < v.length; i++)
                if (at(i) != v.at(i))
                    result = false;

            return result;
        }

        bool operator!=(const Vector<T, N> &v) const
        {
            if (length != v.length)
                throw Tools::StackTrace("Vector::operator!=: shape mismatch");

            return !(*this == v);
        }

        //--------------------------------------------------
        // Map function
        //--------------------------------------------------

        Vector<T, N> map(std::function<T(T)> f) const
        {
            Vector<T, N> result = *this;

            for (int i = 0; i < length; i++)
            {
                if constexpr (std::is_same<VectorT, T>::value)
                    result[i] = f(result[i]);
                else
                    result[i] = result[i].map(f);
            }

            return result;
        }

        //--------------------------------------------------
        // Push and pop function
        //--------------------------------------------------

        void push_back(const VectorT &v)
        {
            if constexpr (!std::is_same<VectorT, T>::value)
                if (length && data[0].shape() != v.shape())
                    throw Tools::StackTrace("Vector::push_back: shape mismatch");

            data.push_back(v);
            length++;
        }

        VectorT pop_back(void)
        {
            if (!length)
                throw Tools::StackTrace("Vector::pop_back: vector is empty");

            VectorT result = data[length - 1];
            if constexpr (!std::is_same<VectorT, T>::value)
                data[length - 1].~Vector();

            data.pop_back();
            length--;

            return result;
        }

        void push_front(const VectorT &v)
        {
            if constexpr (!std::is_same<VectorT, T>::value && length && data[0].shape() != v.shape())
                throw Tools::StackTrace("Vector::push_front: shape mismatch");

            data.insert(data.begin(), v);
            length++;

            data[0] = v;
        }

        VectorT pop_front(void)
        {
            if (!length)
                throw Tools::StackTrace("Vector::pop_front: vector is empty");

            VectorT result = data[0];
            if constexpr (!std::is_same<VectorT, T>::value)
                data[0].~Vector();

            data.erase(data.begin());
            length--;

            return result;
        }

        //--------------------------------------------------
        // Slice
        //--------------------------------------------------

        Vector<T, N> slice(const Vector<size_t, 1> &start, const Vector<size_t, 1> &end) const
        {
            if (N != start.length || N != end.length)
                throw Tools::StackTrace("Vector::slice: Dimension size mismatch");

            bool flag = true;
            for (size_t i = 0; i < start.length; i++)
                if (start.at(i) > end.at(i))
                    flag = false;

            if (!flag)
                throw Tools::StackTrace("Vector::slice: Some member of start is bigger than end's one");

            Vector<T, N> result;
            Vector<size_t, 1> start_copy = start, end_copy = end;

            for (size_t i = start_copy.back(); i < end_copy.back(); i++)
            {
                if constexpr (std::is_same<VectorT, T>::value)
                    result.push_back(data[i]);
                else
                {
                    start_copy.pop_back();
                    end_copy.pop_back();

                    result.push_back(at(i).slice(start_copy, end_copy));
                }
            }

            return result;
        }

        //--------------------------------------------------
        // Clear and resize
        //--------------------------------------------------

        void clear(void)
        {
            if constexpr (!std::is_same<VectorT, T>::value)
                for (size_t i = 0; i < length; i++)
                    data[i].clear();

            data.clear();
            length = 0;
        }

        void resize(const Vector<size_t, 1> &n, const T &s)
        {
            if (N != n.length)
                throw Tools::StackTrace("Vector::resize: Dimension size mismatch");

            Vector<size_t, 1> n_copy = n;

            clear();
            data.resize(n_copy.back());
            length = n_copy.pop_back();

            if constexpr (!std::is_same<VectorT, T>::value)
                for (size_t i = 0; i < length; i++)
                    data[i].resize(n_copy, s);
            else
                for (size_t i = 0; i < length; i++)
                    data[i] = s;
        }

        Vector<T, 1> from(const std::vector<T> &v)
        {
            Vector<T, 1> result;

            Vector<size_t, 1> n;
            n.push_back(v.size());

            result.resize(n);
            for (size_t i = 0; i < v.size(); i++)
                result[i] = v.at(i);

            return result;
        }

        //--------------------------------------------------
        // Shape
        //--------------------------------------------------

        Vector<size_t, 1> shape(void) const
        {
            Vector<size_t, 1> result;
            if constexpr (std::is_same<VectorT, T>::value)
            {
                result.push_back(length);
                return result;
            }
            else
            {
                result = at(0).shape();
                result.push_back(length);
                return result;
            }
        }

        size_t length;

    private:
        std::deque<VectorT> data;
    };

    namespace Utils
    {
        //--------------------------------------------------
        // Vector<T, 1>
        //--------------------------------------------------
        template <typename T>
        Vector<T, 1> reverse(const Vector<T, 1> &v)
        {
            Vector<T, 1> result;

            for (size_t i = v.length - 1; i < v.length; i--)
                result.push_back(v.at(i));

            return result;
        }

        template <typename T>
        T sum(const Vector<T, 1> &v)
        {
            T result = 0;

            for (size_t i = 0; i < v.length; i++)
                result += v.at(i);

            return result;
        }

        template <typename T>
        T dot(const Vector<T, 1> &v1, const Vector<T, 1> &v2)
        {
            if (v1.length != v2.length)
                throw Tools::StackTrace("Utils::dot: size mismatch");

            T result = 0;
            for (size_t i = 0; i < v1.length; i++)
                result += v1.at(i) * v2.at(i);

            return result;
        }

        template <typename T>
        std::pair<T, size_t> max(const Vector<T, 1> &v)
        {
            if (v.length == 0)
                throw Tools::StackTrace("Utils::max: vector is empty");

            T max = v.at(0);
            size_t index = 0;

            for (size_t i = 1; i < v.length; i++)
            {
                if (v.at(i) > max)
                {
                    max = v.at(i);
                    index = i;
                }
            }

            return std::make_pair(max, index);
        }

        //--------------------------------------------------
        // Vector<T, 2>
        //--------------------------------------------------

        template <typename T>
        Vector<T, 2> transpose(const Vector<T, 2> &v)
        {
            Vector<T, 2> result;
            result.resize(reverse(v.shape()), 0);

            for (int i = 0; i < v.shape()[0]; i++)
                for (int j = 0; j < v.shape()[1]; j++)
                    result[j][i] = v.at(i).at(j);

            return result;
        }

        template <typename T>
        Vector<T, 2> dot(const Vector<T, 2> &v1, const Vector<T, 2> &v2)
        {
            if (v1.shape()[1] != v2.shape()[0])
                throw Tools::StackTrace("Utils::dot: size mismatch");

            Vector<size_t, 1> shape = {v1.shape()[0], v2.shape()[1]};

            Vector<T, 2> result;
            result.resize(shape, 0);

            for (int i = 0; i < v1.shape()[0]; i++)
                for (int j = 0; j < v2.shape()[1]; j++)
                    for (int k = 0; k < v1.shape()[1]; k++)
                        result[i][j] += v1.at(i).at(k) * v2.at(i).at(j);

            return result;
        }
    }

    template <typename T, size_t N>
    std::ostream &operator<<(std::ostream &m_os, const Vector<T, N> &m_value)
    {
        m_os << std::string("[");
        for (size_t i = 0; i < m_value.length; i++)
        {
            if (i == m_value.length - 1)
                m_os << m_value.at(i);
            else
                m_os << m_value.at(i) << std::string(", ");
        }
        m_os << std::string("]");
        return m_os;
    }
}

#endif