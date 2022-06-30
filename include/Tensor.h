#ifndef TENSOR_H_
#define TENSOR_H_

#include <functional>
#include <initializer_list>
#include <type_traits>
#include <ostream>
#include <numeric>
#include <random>

namespace SingleNet
{
    template <class T, size_t D, size_t... D_>
    class SymbolicTensor;
    template <class T, size_t D, size_t... D_>
    class Tensor;
    template <class T, size_t D, size_t... D_>
    class TensorRef;

    namespace TensorUtils
    {
        template <size_t... Dims>
        constexpr size_t get_size()
        {
            return (... * Dims);
        }

        template <size_t... Dims>
        constexpr size_t get_dim()
        {
            return sizeof...(Dims);
        }

        template <class T, bool Cond, size_t... Dims>
        struct sub_cond
        {
            typedef Tensor<T, Dims...> type;
        };

        template <class T, size_t... Dims>
        struct sub_cond<T, false, Dims...>
        {
            typedef T type;
        };

        template <class T, bool Cond, size_t... Dims>
        struct sub_cond_ref
        {
            typedef TensorRef<T, Dims...> type;
        };

        template <class T, size_t... Dims>
        struct sub_cond_ref<T, false, Dims...>
        {
            typedef T &type;
        };

        template <class T, bool Cond, size_t FD, size_t... Dims>
        struct cond_apply
        {
            typedef Tensor<T, Dims...> type;
        };

        template <class T, size_t FD, size_t... Dims>
        struct cond_apply<T, false, FD, Dims...>
        {
            typedef Tensor<T, FD> type;
        };
    }

    template <class T, size_t D, size_t... D_>
    class SymbolicTensor
    {
        using SymbolicThis = SymbolicTensor<T, D, D_...>;
        using SymbolicRef = SymbolicTensor<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<T, D, D_...>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;

    public:
        virtual SubRef operator[](size_t i) = 0;

        bool operator==(This &other)
        {
            for (size_t i = 0; i < D; i++)
                if (!((*this)[i] == other[i]))
                    return false;

            return true;
        }

        bool operator==(ThisRef other)
        {
            for (size_t i = 0; i < D; i++)
                if (!((*this)[i] == other[i]))
                    return false;

            return true;
        }

        bool operator!=(This &other)
        {
            for (size_t i = 0; i < D; i++)
                if (!((*this)[i] != other[i]))
                    return false;

            return true;
        }

        bool operator!=(ThisRef other)
        {
            for (size_t i = 0; i < D; i++)
                if (!((*this)[i] != other[i]))
                    return false;

            return true;
        }

        This operator+(This &other)
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] + other[i];

            return result;
        }

        This operator+(ThisRef other)
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] + other[i];

            return result;
        }

        SymbolicThis &operator+=(This &other)
        {
            for (size_t i = 0; i < D; i++)
                (this->operator[](i)) += other[i];

            return (*this);
        }

        auto &operator+=(ThisRef other)
        {
            for (size_t i = 0; i < D; i++)
                (this->operator[](i)) += other[i];

            return *this;
        }

        This operator-(This &other)
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] - other[i];

            return result;
        }

        This operator-(ThisRef other)
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] - other[i];

            return result;
        }

        auto &operator-=(const This &other)
        {
            for (size_t i = 0; i < D; i++)
                (this->operator[](i)) -= other[i];

            return (*this);
        }

        auto &operator-=(const ThisRef other)
        {
            for (size_t i = 0; i < D; i++)
                (this->operator[](i)) -= other[i];

            return (*this);
        }

        This operator*(T scalar)
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] * scalar;

            return result;
        }

        This &operator*=(T scalar)
        {
            for (size_t i = 0; i < D; i++)
                (*this[i]) *= scalar;

            return (*this);
        }

        This operator/(T scalar)
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] / scalar;

            return result;
        }

        This &operator/=(T scalar)
        {
            for (size_t i = 0; i < D; i++)
                (*this[i]) /= scalar;

            return (*this);
        }

        template <class Other>
        Tensor<Other, D, D_...> map(const std::function<Other(T)> &f)
        {
            if constexpr (sizeof...(D_))
            {
                Tensor<Other, D, D_...> result;
                for (size_t i = 0; i < D; i++)
                    result[i] = (*this)[i].map(f);

                return result;
            }
            else
            {
                Tensor<Other, D> result;
                for (size_t i = 0; i < D; i++)
                    result[i] = f((*this)[i]);

                return result;
            }
        }

        template <class Other, size_t FD>
        TensorUtils::cond_apply<Other, sizeof...(D_), FD, D, D_...>::type apply(const std::function<Tensor<Other, FD>(Tensor<T, FD>)> &f)
        {
            if constexpr (sizeof...(D_) > 0)
            {
                Tensor<Other, D, D_...> result;
                for (size_t i = 0; i < D; i++)
                    result[i] = (*this)[i].apply(f);

                return result;
            }
            else
            {
                Tensor<T, FD> deref = (*this);
                Tensor<Other, FD> result = f(deref);
                return result;
            }
        }

    public:
        const size_t rank = TensorUtils::get_dim<D, D_...>();
        const size_t shape[TensorUtils::get_dim<D, D_...>()] = {D, D_...};
        const size_t size = TensorUtils::get_size<D, D_...>();
    };

    template <class T, size_t D, size_t... D_>
    class Tensor : public SymbolicTensor<T, D, D_...>
    {
        friend class TensorRef<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<T, D, D_...>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;

    public:
        Tensor(SymbolicTensor<T, D, D_...> &symbolic)
            : begin_iter(data), end_iter(data + this->size)
        {
        }

        Tensor(const T &value = T())
            : begin_iter(data), end_iter(data + this->size)
        {
            for (size_t i = 0; i < this->size; i++)
                data[i] = value;
        }

        Tensor(const This &other) : begin_iter(data), end_iter(data + this->size)
        {
            for (size_t i = 0; i < this->size; i++)
                data[i] = other.data[i];
        }

        Tensor(const ThisRef other) : begin_iter(data), end_iter(data + this->size)
        {
            for (size_t i = 0; i < this->size; i++)
                (*this)[i] = other[i];
        }

        Tensor(const std::initializer_list<Sub> &list) : begin_iter(data), end_iter(data + this->size)
        {
            size_t idx = 0;
            for (const auto &sub : list)
                (*this)[idx++] = Sub(sub);
        }

        ThisRef &ref()
        {
            return (*this);
        }

        const ThisRef &ref() const
        {
            return (*this);
        }

        This &operator=(const This &other)
        {
            for (size_t i = 0; i < this->size; i++)
                data[i] = other.data[i];

            return (*this);
        }

        SubRef operator[](size_t i)
        {
            if constexpr (sizeof...(D_))
                return SubRef(data + i * TensorUtils::get_size<D_...>());
            else
                return data[i];
        }

        const SubRef operator[](size_t i) const
        {
            if constexpr (sizeof...(D_))
                return SubRef(data + i * TensorUtils::get_size<D_...>());
            else
                return const_cast<T *>(data)[i];
        }

    private:
        T data[TensorUtils::get_size<D, D_...>()];

    public:
        T *const begin_iter;
        T *const end_iter;
    };

    template <class T, size_t D, size_t... D_>
    class TensorRef : public SymbolicTensor<T, D, D_...>
    {
        friend class Tensor<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<T, D, D_...>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;

    public:
        TensorRef(This &origin) : data_start(origin.data), begin_iter(data_start), end_iter(data_start + this->size) {}
        TensorRef(T *const data_start) : data_start(data_start), begin_iter(data_start), end_iter(data_start + this->size) {}
        TensorRef(const T *data_start)
            : data_start(const_cast<T *>(data_start)), 
              begin_iter(const_cast<T *>(data_start)), 
              end_iter(const_cast<T *>(data_start) + this->size) {}

        ThisRef &operator=(const This &origin)
        {
            for (int i = 0; i < this->size; i++)
                data_start[i] = origin.data[i];

            return *this;
        }

        SubRef operator[](size_t i)
        {
            if constexpr (sizeof...(D_))
                return SubRef(data_start + i * TensorUtils::get_size<D_...>());
            else
                return data_start[i];
        }

        const SubRef operator[](size_t i) const
        {
            if constexpr (sizeof...(D_))
                return SubRef(data_start + i * TensorUtils::get_size<D_...>());
            else
                return data_start[i];
        }

    private:
        T *const data_start;

    public:
        T *const begin_iter;
        T *const end_iter;
    };

    template <typename T, size_t D1, size_t D2>
    Tensor<T, D2, D1> get_transposed(const Tensor<T, D1, D2> &origin)
    {
        Tensor<T, D2, D1> result;

        for (size_t i = 0; i < D1; i++)
            for (size_t j = 0; j < D2; j++)
                result[j][i] = origin[i][j];

        return result;
    }

    template <typename T, size_t D1, size_t D2, size_t D3>
    Tensor<T, D1, D3> dot(const Tensor<T, D1, D2> &a, const Tensor<T, D2, D3> &b)
    {
        Tensor<T, D1, D3> result;

        auto b_transposed = get_transposed(b);

#pragma omp parallel for default(shared)
        for (size_t i = 0; i < D1; i++)
            for (size_t j = 0; j < D3; j++)
            {
                auto a_sub = a[i];
                auto b_sub = b_transposed[j];
                result[i][j] = std::inner_product(a_sub.begin_iter, a_sub.end_iter, b_sub.begin_iter, T());
            }

        return result;
    }

    template <typename T, size_t D1, size_t D2>
    Tensor<T, D1, D2> conv(const Tensor<T, D1, D2> &a, const Tensor<T, D1, D2> &b)
    {
        Tensor<T, D1, D2> result;

#pragma omp parallel for default(shared)
        for (size_t i = 0; i < D1; i++)
            for (size_t j = 0; j < D2; j++)
                result[i][j] = a[i][j] * b[i][j];

        return result;
    }

    template <typename T, size_t D>
    size_t argmax(Tensor<T, D> &t)
    {
        T max = std::numeric_limits<T>::lowest();
        size_t max_idx = -1;

        for (size_t i = 0; i < D; i++)
            if (t[i] > max)
            {
                max = t[i];
                max_idx = i;
            }

        return max_idx;
    }
}

template <typename T, size_t D, size_t... D_>
std::ostream &operator<<(std::ostream &os, SingleNet::TensorRef<T, D, D_...> tensor)
{
    os << "[";
    for (size_t i = 0; i < D; i++)
        os << tensor[i] << ((i == D - 1) ? "" : ", ");
    os << "]";

    return os;
}

template <typename T, size_t D, size_t... D_>
std::ostream &operator<<(std::ostream &os, SingleNet::Tensor<T, D, D_...> tensor)
{
    os << "[";
    for (size_t i = 0; i < D; i++)
        os << tensor[i] << ((i == D - 1) ? "" : ", ");
    os << "]";

    return os;
}

#endif