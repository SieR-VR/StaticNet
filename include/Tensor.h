#ifndef TENSOR_H_
#define TENSOR_H_

#include <functional>
#include <initializer_list>
#include <type_traits>
#include <ostream>
#include <numeric>
#include <vector>

#include "Utils/Random.h"

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
        constexpr size_t get_rank()
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

        template <class T, bool Cond, size_t... Dims>
        struct sub_ref_array
        {
            typedef TensorRef<T, Dims...> type;
        };

        template <class T, size_t... Dims>
        struct sub_ref_array<T, false, Dims...>
        {
            typedef T type;
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
        friend class Tensor<T, D, D_...>;
        friend class TensorRef<T, D, D_...>;

        using SymbolicThis = SymbolicTensor<T, D, D_...>;
        using SymbolicRef = SymbolicTensor<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<T, D, D_...>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;
        using RefElement = typename TensorUtils::sub_ref_array<T, sizeof...(D_), D_...>::type;

    public:
        virtual SubRef operator[](size_t i) = 0;
        virtual const SubRef operator[](size_t i) const = 0;

        bool operator==(const This &other) const
        {
            for (size_t i = 0; i < D; i++)
                if (!((*this)[i] == other[i]))
                    return false;

            return true;
        }

        bool operator==(const ThisRef other) const
        {
            for (size_t i = 0; i < D; i++)
                if (!((*this)[i] == other[i]))
                    return false;

            return true;
        }

        bool operator!=(const This &other) const
        {
            for (size_t i = 0; i < D; i++)
                if (!((*this)[i] != other[i]))
                    return false;

            return true;
        }

        bool operator!=(const ThisRef other) const
        {
            for (size_t i = 0; i < D; i++)
                if (!((*this)[i] != other[i]))
                    return false;

            return true;
        }

        template <class U>
        This operator+(const SymbolicTensor<U, D, D_...> &other) const
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] + other[i];

            return result;
        }

        template <class U>
        auto &operator+=(const SymbolicTensor<U, D, D_...> &other)
        {
            for (size_t i = 0; i < D; i++)
                (this->operator[](i)) += other[i];

            return (*this);
        }

        template <class U>
        This operator-(const SymbolicTensor<U, D, D_...> &other) const
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] - other[i];

            return result;
        }

        template <class U>
        auto &operator-=(const SymbolicTensor<U, D, D_...> &other)
        {
            for (size_t i = 0; i < D; i++)
                (*this)[i] -= other[i];

            return (*this);
        }

        template <class U>
        This operator*(U scalar) const
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] * scalar;

            return result;
        }

        template <class U>
        This &operator*=(U scalar)
        {
            for (size_t i = 0; i < D; i++)
                (*this[i]) *= scalar;

            return (*this);
        }

        template <class U>
        This operator/(U scalar) const
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = (*this)[i] / scalar;

            return result;
        }

        template <class U>
        This &operator/=(U scalar)
        {
            for (size_t i = 0; i < D; i++)
                (*this[i]) /= scalar;

            return (*this);
        }

        This operator-() const
        {
            This result;

            for (size_t i = 0; i < D; i++)
                result[i] = -(*this)[i];

            return result;
        }

        template <class Other>
        Tensor<Other, D, D_...> map(const std::function<Other(T)> &f) const
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
        typename TensorUtils::cond_apply<Other, sizeof...(D_), FD, D, D_...>::type apply(const std::function<Tensor<Other, FD>(Tensor<T, FD>)> &f) const
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
                Tensor<Other, FD> result = f(Tensor<T, FD>(*this));
                return result;
            }
        }

        Sub reduce() const
        {
            Sub result = T();

            for (size_t i = 0; i < D; i++)
                result += (*this)[i];

            return result;
        }

    public:
        static constexpr size_t rank = TensorUtils::get_rank<D, D_...>();
        static constexpr size_t shape[TensorUtils::get_rank<D, D_...>()] = {D, D_...};
        static constexpr size_t size = TensorUtils::get_size<D, D_...>();
    };

    template <class T, size_t D, size_t... D_>
    class Tensor : public SymbolicTensor<T, D, D_...>
    {
        friend class TensorRef<T, D, D_...>;
        friend class SymbolicTensor<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<T, D, D_...>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;

    public:
        Tensor(const T &value = T())
            : begin_iter(data), end_iter(data + this->size)
        {
            for (size_t i = 0; i < this->size; i++)
                data[i] = value;
        }

        Tensor(const SymbolicTensor<T, D, D_...> &symbolic)
            : begin_iter(data), end_iter(data + this->size)
        {
            for (size_t i = 0; i < this->size; i++)
                (*this)[i] = symbolic[i];
        }

        Tensor(const std::initializer_list<Sub> &list) : begin_iter(data), end_iter(data + this->size)
        {
            size_t idx = 0;
            for (const auto &sub : list)
                (*this)[idx++] = Sub(sub);
        }

        static This random()
        {
            if constexpr (sizeof...(D_)) {
                This result;
                for (size_t i = 0; i < D; i++)
                    result[i] = Sub::random();
                return result;
            }
            else {
                This result;
                for (size_t i = 0; i < D; i++)
                    result[i] = Random::rand<T>();
                return result;
            }
        }

        ThisRef &ref()
        {
            return (*this);
        }

        const ThisRef &ref() const
        {
            return (*this);
        }

        This &operator=(const SymbolicTensor<T, D, D_...> &other)
        {
            for (size_t i = 0; i < this->size; i++)
                data[i] = other.data[i];

            return (*this);
        }

        SubRef operator[](size_t i)
        {
            if constexpr (sizeof...(D_))
                return data + i * TensorUtils::get_size<D_...>();
            else
                return data[i];
        }

        const SubRef operator[](size_t i) const
        {
            if constexpr (sizeof...(D_))
                return data + i * TensorUtils::get_size<D_...>();
            else
                return const_cast<T *>(data)[i];
        }

        template <size_t ...P>
        TensorRef<T, P...> reshape_ref() const
        {
            return TensorRef<T, P...>(this->data);
        }

        template <size_t ...P>
        Tensor<T, P...> reshape() const
        {
            static_assert(TensorUtils::get_size<P...>() == this->size, "Tensor reshape error");

            Tensor<T, P...> result;
            for (size_t i = 0; i < TensorUtils::get_size<P...>(); i++)
                result.data[i] = (*this).data[i];

            return result;
        }

    public:
        T data[TensorUtils::get_size<D, D_...>()];
        T * begin_iter;
        T * end_iter;
    };

    template <class T, size_t D, size_t... D_>
    class TensorRef : public SymbolicTensor<T, D, D_...>
    {
        friend class Tensor<T, D, D_...>;
        friend class SymbolicTensor<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<T, D, D_...>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;
        using RefElement = typename TensorUtils::sub_ref_array<T, sizeof...(D_), D_...>::type;

    public:
        TensorRef() 
            : data(nullptr), 
              begin_iter(nullptr), 
              end_iter(static_cast<T *>(nullptr) + this->size)
        {
        }
        TensorRef(This &origin)
            : data(origin.data), 
              begin_iter(data), 
              end_iter(data + this->size)
        {
        }
        TensorRef(T *const data_start)
            : data(data_start), 
              begin_iter(data_start), 
              end_iter(data_start + this->size)
        {
        }
        TensorRef(const T *data_start)
            : data(const_cast<T *>(data_start)),
              begin_iter(const_cast<T *>(data_start)),
              end_iter(const_cast<T *>(data_start) + this->size)
        {
        }

        ThisRef &operator=(const This &origin)
        {
            for (int i = 0; i < this->size; i++)
                data[i] = origin.data[i];

            return *this;
        }

        ThisRef &operator=(const ThisRef &origin_ref)
        {
            for (int i = 0; i < this->size; i++)
                data[i] = origin_ref.data[i];

            return *this;
        }

        SubRef operator[](size_t i)
        {
            if constexpr (sizeof...(D_))
                return data + i * TensorUtils::get_size<D_...>();
            else
                return data[i];
        }

        const SubRef operator[](size_t i) const
        {
            if constexpr (sizeof...(D_))
                return data + i * TensorUtils::get_size<D_...>();
            else
                return data[i];
        }

        This deref() const
        {
            This result;
            for (size_t i = 0; i < this->size; i++)
                result[i] = data[i];
            return result;
        }

    public:
        T *const data;
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

    template <typename T, size_t D, size_t ...D_>
    Tensor<T, D, D_...> hadamard(const SymbolicTensor<T, D, D_...> &a, const SymbolicTensor<T, D, D_...> &b)
    {
        Tensor<T, D, D_...> result;

        for (size_t i = 0; i < D; i++)
            result[i] = hadamard(a[i], b[i]);

        return result;
    }

    template <typename T, size_t D1, size_t D2>
    Tensor<T, D1, D2> hadamard(const SymbolicTensor<T, D1, D2> &a, const SymbolicTensor<T, D1, D2> &b)
    {
        Tensor<T, D1, D2> result;

#pragma omp parallel for default(shared)
        for (size_t i = 0; i < D1; i++)
            for (size_t j = 0; j < D2; j++)
                result[i][j] = a[i][j] * b[i][j];

        return result;
    }

    template <typename T, size_t D>
    Tensor<T, D> hadamard(const SymbolicTensor<T, D> &a, const SymbolicTensor<T, D> &b)
    {
        Tensor<T, D> result;

#pragma omp parallel for default(shared)
        for (size_t i = 0; i < D; i++)
            result[i] = a[i] * b[i];

        return result;
    }

    template <typename T, size_t IDim, size_t KDim>
    Tensor<T, IDim-KDim+1, IDim-KDim+1> conv(const SymbolicTensor<T, IDim, IDim> &input, const SymbolicTensor<T, KDim, KDim> &kernel)
    {
        static_assert(IDim >= KDim, "Kernel must be same or bigger than input");

        Tensor<T, IDim-KDim+1, IDim-KDim+1> result;

#pragma omp parallel for default(shared)
        for (size_t i = 0; i < IDim-KDim+1; i++)
            for (size_t j = 0; j < IDim-KDim+1; j++)
                for (size_t k = 0; k < KDim; k++)
                    for (size_t l = 0; l < KDim; l++)
                        result[i][j] += input[i+k][j+l] * kernel[k][l];

        return result;
    }

    template <typename T, size_t D>
    Tensor<T, D, D> flip(const SymbolicTensor<T, D, D> &input)
    {
        Tensor<T, D, D> result;

        for (size_t i = 0; i < D; i++)
            for (size_t j = 0; j < D; j++)
                result[i][j] = input[D-i-1][D-j-1];

        return result;
    }
    
    template <typename T, size_t D, size_t P>
    Tensor<T, D+2*P, D+2*P> pad(const SymbolicTensor<T, D, D> &input, T pad_value = T())
    {
        Tensor<T, D+2*P, D+2*P> result { pad_value };

        for (size_t i = 0; i < D; i++)
            for (size_t j = 0; j < D; j++)
                result[P + i][P + j] = input[i][j];

        return result;
    }

    template <typename T, size_t IDim, size_t ODim>
    Tensor<T, ODim, ODim> pool(const SymbolicTensor<T, IDim, IDim> &input, std::function<T(Tensor<T, IDim/ODim, IDim/ODim>)> pool_func)
    {
        constexpr size_t KernelSize = IDim/ODim;
        Tensor<T, ODim, ODim> result;

        for (size_t i = 0; i < ODim; i++)
            for (size_t j = 0; j < ODim; j++) {
                Tensor<T, KernelSize, KernelSize> sub_input;
                for (size_t k = 0; k < KernelSize; k++)
                    for (size_t l = 0; l < KernelSize; l++)
                        sub_input[k][l] = input[i*KernelSize+k][j*KernelSize+l];

                result[i][j] = pool_func(sub_input);
            }

        return result;
    }

    template <typename T, size_t IDim, size_t ODim>
    Tensor<T, IDim, IDim> unpool(const SymbolicTensor<T, ODim, ODim> &input, std::function<Tensor<T, IDim/ODim, IDim/ODim>(T)> unpool_func)
    {
        constexpr size_t KernelSize = IDim/ODim;
        Tensor<T, IDim, IDim> result;

        for (size_t i = 0; i < ODim; i++)
            for (size_t j = 0; j < ODim; j++) {
                Tensor<T, KernelSize, KernelSize> sub_input = unpool_func(input[i][j]);
                for (size_t k = 0; k < KernelSize; k++)
                    for (size_t l = 0; l < KernelSize; l++)
                        result[i*KernelSize+k][j*KernelSize+l] = sub_input[k][l];
            }

        return result;
    } 

    template <typename T, size_t D>
    size_t argmax(const SymbolicTensor<T, D> &t)
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