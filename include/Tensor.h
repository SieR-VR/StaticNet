#ifndef TENSOR_H_
#define TENSOR_H_

#include <functional>
#include <initializer_list>
#include <type_traits>
#include <ostream>
#include <numeric>

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
    }

    template <class T, size_t D, size_t... D_>
    class SymbolicTensor
    {
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

        This &operator+=(This &other)
        {
            for (size_t i = 0; i < D; i++)
                (*this[i]) += other[i];

            return (*this);
        }

        This operator+=(ThisRef other)
        {
            for (size_t i = 0; i < D; i++)
                (*this[i]) += other[i];

            return (*this);
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

        This &operator-=(This &other)
        {
            for (size_t i = 0; i < D; i++)
                (*this[i]) -= other[i];

            return (*this);
        }

        This operator-=(ThisRef other)
        {
            for (size_t i = 0; i < D; i++)
                (*this[i]) -= other[i];

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

    public:
        const size_t dimension = TensorUtils::get_dim<D, D_...>();
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
        Tensor(const This &other) : begin_iter(data), end_iter(data + this->size)
        {
            for (size_t i = 0; i < this->size; i++)
                data[i] = other.data[i];
        }

        Tensor() : begin_iter(data), end_iter(data + this->size) {}

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

        SubRef operator[](size_t i)
        {
            if constexpr (sizeof...(D_))
                return SubRef(data + i * TensorUtils::get_size<D_...>());
            else
                return data[i];
        }

    private:
        T data[TensorUtils::get_size<D, D_...>()];
    public:
        T* const begin_iter;
        T* const end_iter;
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
        TensorRef(const This &origin) : 
            data_start(origin.data), begin_iter(data_start), end_iter(data_start + this->size) {}
        TensorRef(T *const data_start) : 
            data_start(data_start), begin_iter(data_start), end_iter(data_start + this->size) {}

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

    private:
        T *const data_start;

    public:
        T* const begin_iter;
        T* const end_iter;
    };

    template <typename T, size_t D1, size_t D2>
    Tensor<T, D2, D1> get_transposed(Tensor<T, D1, D2> &origin) {
        Tensor<T, D2, D1> result;

        for(size_t i = 0; i < D1; i++)
            for (size_t j = 0; j < D2; j++)
                result[j][i] = origin[i][j];

        return result;
    }

    template <typename T, size_t D1, size_t D2, size_t D3>
    Tensor<T, D1, D3> dot(Tensor<T, D1, D2> &a, Tensor<T, D2, D3> &b) {
        Tensor<T, D1, D3> result;

        auto b_transposed = get_transposed(b);

        #pragma omp parallel for default(shared)
        for (size_t i = 0; i < D1; i++)
            for (size_t j = 0; j < D3; j++) {
                auto a_sub = a[i];
                auto b_sub = b_transposed[j];
                result[i][j] = std::inner_product(a_sub.begin_iter, a_sub.end_iter, b_sub.begin_iter, T());
            }

        return result;
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