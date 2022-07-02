#ifndef TENSOR_H_
#define TENSOR_H_

#include <functional>
#include <initializer_list>
#include <type_traits>
#include <iostream>
#include <numeric>
#include <vector>

#include "Utils/Random.h"

namespace SingleNet
{
    template <class T, size_t D, size_t... D_>
    class SymbolicTensor;
    template <class T, size_t D, size_t... D_>
    class Tensor;
    template <typename...>
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
            typedef TensorRef<Tensor<T, Dims...>, Tensor<T, Dims...>> type;
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

        template <size_t D, size_t... D_>
        size_t get_index(size_t i[sizeof...(D_) + 1]) {
            if constexpr (sizeof...(D_)) {
                return get_index<D_>(i + 1) + i[0] * get_size<D_...>(); 
            }
            else {
                return i[0];
            }
        }
    }

    template <class T, size_t D, size_t... D_>
    class SymbolicTensor
    {
        using SymbolicThis = SymbolicTensor<T, D, D_...>;
        using SymbolicRef = SymbolicTensor<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<This, This>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;

        friend class This;
        friend class ThisRef;

    public:

        template <size_t SliceD, size_t... SliceD_>
        class RawIterator {
            friend class SymbolicThis;
            SymbolicThis *_this;
            size_t _index = 0;

        public:
            RawIterator(SymbolicThis *this_, size_t start[sizeof...(SliceD_) + 1]) : _this(this_) {
                _index = TensorUtils::get_index<SliceD>(start);
            }

            T &operator*() { return _this->data[_this->_index]; }

            RawIterator &operator++() {
                return *this;
            }

            bool operator!=(const RawIterator &other) {
                return _this->_index != other._index;
            }

            bool operator==(const RawIterator &other) {
                return _this->_index == other._index;
            }
        };

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

        Sub reduce() const
        {
            Sub result = T();

            for (size_t i = 0; i < D; i++)
                result += (*this)[i];

            return result;
        }

        template <size_t... P>
        Tensor<T, P...> reshape();

        template <size_t... P>
        TensorRef<This, Tensor<T, P...>> reshape_ref();

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

        template <size_t SliceD, size_t... SliceD_>
        Tensor<T, SliceD, SliceD_...> slice(size_t start[sizeof...(SliceD_) + 1], size_t end[sizeof...(SliceD_) + 1]) const
        {
            Tensor<T, SliceD, SliceD_...> result;

            if constexpr (sizeof...(SliceD_))
            {
                for (size_t i = start[0]; i < end[0]; i++)
                    result[i] = (*this)[i].template slice<SliceD_>(start + 1, end + 1);

                return result;
            }
            else
            {
                for (size_t i = start[0]; i < end[0]; i++)
                    result[i] = (*this)[i];

                return result;
            }
        }

        template <size_t SliceD, size_t... SliceD_>
        TensorRef<This, Tensor<T, SliceD, SliceD_...>> slice_ref(size_t slice_start[sizeof...(SliceD_) + 1], size_t slice_end[sizeof...(SliceD_) + 1])
        {
            return TensorRef<This, Tensor<T, SliceD, SliceD_...>>(this, slice_start, slice_end);
        }

    public:
        T *data = nullptr;

        static constexpr size_t rank = TensorUtils::get_rank<D, D_...>();
        static constexpr size_t shape[TensorUtils::get_rank<D, D_...>()] = {D, D_...};
        static constexpr size_t size = TensorUtils::get_size<D, D_...>();

        T *begin_iter;
        T *end_iter;
    };

    template <class T, size_t D, size_t... D_>
    class Tensor : public SymbolicTensor<T, D, D_...>
    {
        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<This, This>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;

        friend class ThisRef;

    public:
        Tensor(const T &value = T())
        {
            this->data = new T[TensorUtils::get_size<D, D_...>()];
            std::fill(this->data, this->data + TensorUtils::get_size<D, D_...>(), value);

            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }

        Tensor(This &&other)
        {
            this->data = other.data;
            other.data = nullptr;

            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }

        Tensor(const This &other)
        {
            this->data = new T[TensorUtils::get_size<D, D_...>()];
            memcpy(this->data, other.data, sizeof(T) * this->size);

            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }

        Tensor(const ThisRef &other)
        {
            this->data = new T[TensorUtils::get_size<D, D_...>()];
            memcpy(this->data, other.data, sizeof(T) * this->size);

            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }

        Tensor(const std::initializer_list<Sub> &list)
        {
            this->data = new T[TensorUtils::get_size<D, D_...>()];
            size_t idx = 0;
            for (const auto &sub : list)
                (*this)[idx++] = Sub(sub);

            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }

        ~Tensor()
        {
            if (this->data)
                delete[] this->data;
        }

        SubRef operator[](size_t i) override
        {
            if constexpr (sizeof...(D_))
                return this->data + i * TensorUtils::get_size<D_...>();
            else
                return this->data[i];
        }

        const SubRef operator[](size_t i) const override
        {
            if constexpr (sizeof...(D_))
                return this->data + i * TensorUtils::get_size<D_...>();
            else
                return const_cast<T *>(this->data)[i];
        }

        static This random()
        {
            if constexpr (sizeof...(D_))
            {
                This result;
                for (size_t i = 0; i < D; i++)
                    result[i] = Sub::random();
                return result;
            }
            else
            {
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

        This &operator=(const This &other)
        {
            memcpy(this->data, other.data, sizeof(T) * this->size);
            return (*this);
        }

        template <size_t... P>
        Tensor<T, P...> reshape() const
        {
            static_assert(TensorUtils::get_size<P...>() == this->size, "Tensor reshape error");

            Tensor<T, P...> result;
            memcpy(result.data, this->data, this->size * sizeof(T));

            return result;
        }

        template <size_t... P>
        TensorRef<This, Tensor<T, P...>> reshape_ref() const
        {
            return TensorRef<This, Tensor<T, P...>>(this->data);
        }

        template <class Other, size_t FD>
        typename TensorUtils::cond_apply<Other, sizeof...(D_), FD, D, D_...>::type apply(const std::function<Tensor<Other, FD>(const Tensor<T, FD> &)> &f) const
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
                Tensor<Other, FD> result = f(*this);
                return result;
            }
        }
    };

    template <class T, size_t D, size_t... D_, size_t SliceD, size_t... SliceD_>
    class TensorRef<Tensor<T, D, D_...>, Tensor<T, SliceD, SliceD_...>> : public SymbolicTensor<T, SliceD, SliceD_...>
    {
        using This = Tensor<T, SliceD, SliceD_...>;
        using ThisRef = TensorRef<This, This>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(SliceD_), SliceD_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(SliceD_), SliceD_...>::type;

        friend class SymbolicThis;
        friend class This;

    public:
        TensorRef(This &origin)
        {
            this->data = origin.data;
            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }

        TensorRef(ThisRef &origin)
        {
            this->data = origin.data;
            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }
        
        TensorRef(This &origin, size_t *slice_start, size_t *slice_end)
        {
            this->data = origin.data;
            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;

            this->slice_start = slice_start;
            this->slice_end = slice_end;
        }

        TensorRef(ThisRef &origin, size_t *slice_start, size_t *slice_end)
        {
            this->data = origin.data;
            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;

            this->slice_start = slice_start;
            this->slice_end = slice_end;
        }

        TensorRef(T *const data_start)
        {
            this->data = data_start;
            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }

        TensorRef(const T *data_start)
        {
            this->data = const_cast<T *>(data_start);
            this->data = data_start;
            this->begin_iter = this->data;
            this->end_iter = this->data + this->size;
        }

        ThisRef &operator=(const This &origin)
        {
            memcpy(this->data, origin.data, sizeof(T) * this->size);
            return *this;
        }

        ThisRef &operator=(const ThisRef &origin_ref)
        {
            memcpy(this->data, origin_ref.data, sizeof(T) * this->size);
            return *this;
        }

        SubRef operator[](size_t i) override
        {
            if constexpr (sizeof...(D_))
                return this->data + i * TensorUtils::get_size<D_...>();
            else
                return this->data[i];
        }

        const SubRef operator[](size_t i) const override
        {
            if constexpr (sizeof...(D_))
                return this->data + i * TensorUtils::get_size<D_...>();
            else
                return this->data[i];
        }

        template <size_t... P>
        Tensor<T, P...> reshape() const
        {
            static_assert(TensorUtils::get_size<P...>() == this->size, "Tensor reshape error");

            Tensor<T, P...> result;
            memcpy(result.data, this->data, this->size * sizeof(T));

            return result;
        }

        template <size_t... P>
        TensorRef<This, Tensor<T, P...>> reshape_ref() const
        {
            return TensorRef<This, Tensor<T, P...>>(this->data);
        }

        This deref() const
        {
            This result;
            for (size_t i = 0; i < this->size; i++)
                result[i] = this->data[i];
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
        typename TensorUtils::cond_apply<Other, sizeof...(D_), FD, D, D_...>::type apply(const std::function<Tensor<Other, FD>(const Tensor<T, FD> &)> &f) const
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
                Tensor<Other, FD> result = f(*this);
                return result;
            }
        }

    public:
        size_t *slice_start = nullptr;
        size_t *slice_end = nullptr;
    };

    template <typename T, size_t D1, size_t D2>
    Tensor<T, D2, D1> get_transposed(const SymbolicTensor<T, D1, D2> &origin)
    {
        Tensor<T, D2, D1> result;

        for (size_t i = 0; i < D1; i++)
            for (size_t j = 0; j < D2; j++)
                result[j][i] = origin[i][j];

        return result;
    }

    template <typename T, size_t D1, size_t D2, size_t D3>
    Tensor<T, D1, D3> dot(const SymbolicTensor<T, D1, D2> &a, const SymbolicTensor<T, D2, D3> &b)
    {
        Tensor<T, D1, D3> result;

        auto b_transposed = get_transposed(b);

#pragma omp parallel for default(shared)
        for (int i = 0; i < D1; i++)
            for (int j = 0; j < D3; j++)
            {
                auto a_sub = a[i];
                auto b_sub = b_transposed[j];
                result[i][j] = std::inner_product(a_sub.begin_iter, a_sub.end_iter, b_sub.begin_iter, T());
            }

        return result;
    }

    template <typename T, size_t D, size_t... D_>
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
        for (int i = 0; i < D1; i++)
            for (int j = 0; j < D2; j++)
                result[i][j] = a[i][j] * b[i][j];

        return result;
    }

    template <typename T, size_t D>
    Tensor<T, D> hadamard(const SymbolicTensor<T, D> &a, const SymbolicTensor<T, D> &b)
    {
        Tensor<T, D> result;

#pragma omp parallel for default(shared)
        for (int i = 0; i < D; i++)
            result[i] = a[i] * b[i];

        return result;
    }

    template <typename T, size_t ICDim, size_t OCDim, size_t IDim, size_t KDim>
    Tensor<T, OCDim, IDim - KDim + 1, IDim - KDim + 1> conv(const Tensor<T, ICDim, IDim, IDim> &input, const Tensor<T, OCDim / ICDim, KDim, KDim> &kernel)
    {
        static_assert(IDim >= KDim, "Kernel must be same or bigger than input");
        static_assert(OCDim % ICDim == 0, "Output channel must be multiple of input channel");

        Tensor<T, OCDim, IDim - KDim + 1, IDim - KDim + 1> result;

        return result;
    }

    template <typename T, size_t D>
    Tensor<T, D, D> flip(const SymbolicTensor<T, D, D> &input)
    {
        Tensor<T, D, D> result;

#pragma omp parallel for default(shared)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                result[i][j] = input[D - i - 1][D - j - 1];

        return result;
    }

    template <typename T, size_t D, size_t P>
    Tensor<T, D + 2 * P, D + 2 * P> pad(const SymbolicTensor<T, D, D> &input, T pad_value = T())
    {
        Tensor<T, D + 2 * P, D + 2 * P> result{pad_value};

#pragma omp parallel for default(shared)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                result[P + i][P + j] = input[i][j];

        return result;
    }

    template <typename T, size_t IDim, size_t ODim>
    Tensor<T, ODim, ODim> pool(const Tensor<T, IDim, IDim> &input, std::function<T(Tensor<T, IDim / ODim, IDim / ODim>)> pool_func)
    {
        constexpr size_t KernelSize = IDim / ODim;
        Tensor<T, ODim, ODim> result;

#pragma omp parallel for default(shared)
        for (int i = 0; i < ODim; i++)
            for (int j = 0; j < ODim; j++)
            {
                Tensor<T, KernelSize, KernelSize> sub_input;
                for (int k = 0; k < KernelSize; k++)
                    for (int l = 0; l < KernelSize; l++)
                        sub_input[k][l] = input[i * KernelSize + k][j * KernelSize + l];

                result[i][j] = pool_func(sub_input);
            }

        return result;
    }

    template <typename T, size_t IDim, size_t ODim>
    Tensor<T, IDim, IDim> unpool(const Tensor<T, ODim, ODim> &input, std::function<Tensor<T, IDim / ODim, IDim / ODim>(T)> unpool_func)
    {
        constexpr size_t KernelSize = IDim / ODim;
        Tensor<T, IDim, IDim> result;

#pragma omp parallel for default(shared)
        for (int i = 0; i < ODim; i++)
            for (int j = 0; j < ODim; j++)
            {
                Tensor<T, KernelSize, KernelSize> sub_input = unpool_func(input[i][j]);
                for (int k = 0; k < KernelSize; k++)
                    for (int l = 0; l < KernelSize; l++)
                        result[i * KernelSize + k][j * KernelSize + l] = sub_input[k][l];
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
std::ostream &operator<<(std::ostream &os, SingleNet::SymbolicTensor<T, D, D_...> tensor)
{
    os << "[";
    for (size_t i = 0; i < D; i++)
        os << tensor[i] << ((i == D - 1) ? "" : ", ");
    os << "]";

    return os;
}

#endif