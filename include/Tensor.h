#ifndef TENSOR_H_
#define TENSOR_H_

#include <functional>
#include <initializer_list>
#include <type_traits>
#include <iostream>
#include <numeric>
#include <vector>
#include <cstring>

#include "Utils/Random.h"

namespace SingleNet
{
    template <typename...>
    class TensorRef;
    template <class T, size_t D, size_t... D_>
    class Tensor;
    template <size_t...>
    struct Transpose;

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

        template <class T, class Origin, bool Cond, size_t... Dims>
        struct sub_cond_ref
        {
            typedef TensorRef<Origin, Tensor<T, Dims...>> type;
        };

        template <class T, class Origin, size_t... Dims>
        struct sub_cond_ref<T, Origin, false, Dims...>
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
        size_t get_index(size_t i[sizeof...(D_) + 1])
        {
            if constexpr (sizeof...(D_))
            {
                return get_index<D_...>(i + 1) + i[0] * get_size<D_...>();
            }
            else
            {
                return i[0];
            }
        }

        template <size_t, typename...>
        struct tensor_add_rank;

        template <class T, size_t R, size_t... D_>
        struct tensor_add_rank<R, Tensor<T, D_...>>
        {
            typedef Tensor<T, R, D_...> type;
        };

        template <size_t R, size_t i, size_t D, size_t... Dims>
        constexpr size_t get_rank_dim()
        {
            if constexpr (R == i)
            {
                return D;
            }
            else
            {
                return get_rank_dim<R, i + 1, Dims...>();
            }
        }

        template <typename...>
        struct transpose_helper;

        template <class T, size_t R, size_t... R_, size_t... D_>
        struct transpose_helper<Tensor<T, D_...>, Transpose<R, R_...>>
        {
            static_assert(R <= sizeof...(D_) + 1, "Transpose rank is too large");
            typedef tensor_add_rank<get_rank_dim<R, 0, D_...>(), typename transpose_helper<Tensor<T, D_...>, Transpose<R_...>>::type>::type type;
        };

        template <class T, size_t R, size_t... D_>
        struct transpose_helper<Tensor<T, D_...>, Transpose<R>>
        {
            static_assert(R <= sizeof...(D_) + 1, "Transpose rank is too large");
            typedef Tensor<T, get_rank_dim<R, 0, D_...>()> type;
        };

        template <size_t Rank, size_t i, size_t TDim, size_t... TDims>
        std::array<size_t, Rank> get_transposed_index(const std::array<size_t, Rank>& idx)
        {
            if constexpr (sizeof...(TDims))
            {
                std::array<size_t, Rank> result = get_transposed_index<Rank, i + 1, TDims...>(idx);
                result[i] = idx[TDim];
                return result;
            }
            else
            {
                std::array<size_t, Rank> result;
                result[i] = idx[TDim];
                return result;
            }
        }
    }

    template <class T, size_t D, size_t... D_, size_t SliceD, size_t... SliceD_>
    class TensorRef<Tensor<T, D, D_...>, Tensor<T, SliceD, SliceD_...>>
    {
    public:
        template <class U>
        using Origin = Tensor<U, D, D_...>;
        template <class U>
        using Slice = Tensor<U, SliceD, SliceD_...>;

        using This = TensorRef<Origin<T>, Tensor<T, SliceD, SliceD_...>>;
        using SubRef = typename TensorUtils::sub_cond_ref<T, Origin<T>, sizeof...(SliceD_), SliceD_...>::type;

        class raw_iterator
        {
            T *data;
            size_t _start_index;
            size_t _index = 0;
            size_t _slice_index = 0;

        public:
            static size_t get_iter_step(size_t _slice_index)
            {
                if constexpr (sizeof...(SliceD_))
                {
                    if (_slice_index / TensorUtils::get_size<SliceD_...>() != (_slice_index + 1) / TensorUtils::get_size<SliceD_...>())
                        return TensorUtils::get_size<D_...>() - TensorUtils::get_size<SliceD_...>() + 1;
                    else
                        return TensorRef<Tensor<T, D_...>, Tensor<T, SliceD_...>>::raw_iterator::get_iter_step(_slice_index % TensorUtils::get_size<D_...>());
                }
                else
                    return 1;
            }

            raw_iterator(T *data, size_t start = 0, size_t _slice_index = 0) : data(data), _start_index(start), _slice_index(_slice_index == UINT64_MAX ? TensorUtils::get_size<SliceD, SliceD_...>() : _slice_index) {
            }

            T &operator*() { return data[_start_index + _index]; }

            raw_iterator &operator++()
            {
                _index += get_iter_step(_slice_index);
                _slice_index++;
                return *this;
            }

            bool operator!=(const raw_iterator &other)
            {
                return this->_slice_index != other._slice_index;
            }

            bool operator==(const raw_iterator &other)
            {
                return this->_slice_index == other._slice_index;
            }
        };

        TensorRef(const Tensor<T, D, D_...> *const origin, size_t slice_start = 0)
            : slice_start(slice_start)
        {
            this->origin = const_cast<Tensor<T, D, D_...> *>(origin);
        }

        template <size_t OtherSlice, size_t... OtherSlice_>
        TensorRef(const TensorRef<Tensor<T, D, D_...>, Tensor<T, OtherSlice, OtherSlice_...>> &other, size_t slice_start = 0)
            : slice_start(other.slice_start + slice_start)
        {
            this->origin = other.origin;
        }

        SubRef operator[](size_t i)
        {
            if constexpr (sizeof...(SliceD_))
                return SubRef(origin, slice_start + i * TensorUtils::get_size<SliceD_...>());
            else
                return origin->data[slice_start + i];
        }

        const SubRef operator[](size_t i) const
        {
            if constexpr (sizeof...(SliceD_))
                return SubRef(origin, slice_start + i * TensorUtils::get_size<SliceD_...>());
            else
                return origin->data[slice_start + i];
        }

        template <class U, size_t... OtherOriginDim>
        This &operator=(const TensorRef<Tensor<U, OtherOriginDim...>, Slice<U>> &other)
        {
            for (size_t i = 0; i < SliceD; ++i)
                (*this)[i] = other[i];
            return *this;
        }

        template <class U, size_t... OtherOriginDim>
        bool operator==(const TensorRef<Tensor<U, OtherOriginDim...>, Slice<U>> &other) const
        {
            for (size_t i = 0; i < SliceD; i++)
                if ((*this)[i] != other[i])
                    return false;

            return true;
        }

        template <class U, size_t... OtherOriginDim>
        bool operator!=(const TensorRef<Tensor<U, OtherOriginDim...>, Slice<U>> &other) const
        {
            for (size_t i = 0; i < SliceD; i++)
                if ((*this)[i] == other[i])
                    return false;

            return true;
        }

        template <class U, size_t... OtherOriginDim>
        Slice<T> operator+(const TensorRef<Tensor<U, OtherOriginDim...>, Slice<U>> &other) const
        {
            Slice<T> result;
            for (size_t i = 0; i < SliceD; i++)
                result[i] = (*this)[i] + other[i];
            return result;
        }

        template <class U, size_t... OtherOriginDim>
        This &operator+=(const TensorRef<Tensor<U, OtherOriginDim...>, Slice<U>> &other)
        {
            for (size_t i = 0; i < SliceD; i++)
                (*this)[i] += other[i];
            return *this;
        }

        template <class U, size_t... OtherOriginDim>
        Slice<T> operator-(const TensorRef<Tensor<U, OtherOriginDim...>, Slice<U>> &other) const
        {
            Slice<T> result;
            for (size_t i = 0; i < SliceD; i++)
                result[i] = (*this)[i] - other[i];
            return result;
        }

        template <class U, size_t... OtherOriginDim>
        This &operator-=(const TensorRef<Tensor<U, OtherOriginDim...>, Slice<U>> &other)
        {
            for (size_t i = 0; i < SliceD; i++)
                (*this)[i] -= other[i];
            return *this;
        }

        template <class U>
        Slice<T> operator*(U scalar) const
        {
            Slice<T> result;
            for (size_t i = 0; i < SliceD; i++)
                result[i] = (*this)[i] * scalar;
            return result;
        }

        template <class U>
        This &operator*=(U scalar)
        {
            for (size_t i = 0; i < SliceD; i++)
                (*this)[i] *= scalar;
            return *this;
        }

        template <class U>
        Slice<T> operator/(U scalar) const
        {
            Slice<T> result;
            for (size_t i = 0; i < SliceD; i++)
                result[i] = (*this)[i] / scalar;
            return result;
        }

        template <class U>
        This &operator/=(U scalar)
        {
            for (size_t i = 0; i < SliceD; i++)
                (*this)[i] /= scalar;
            return *this;
        }

        Slice<T> operator-() const
        {
            Slice<T> result;
            for (size_t i = 0; i < SliceD; i++)
                result[i] = -(*this)[i];
            return result;
        }

        typename TensorUtils::sub_cond<T, sizeof...(SliceD_), SliceD_...>::type reduce() const
        {
            if constexpr (sizeof...(SliceD_))
            {
                Tensor<T, SliceD_...> result;
                for (size_t i = 0; i < SliceD; i++)
                    result += (*this)[i];

                return result;
            }
            else
            {
                T result;
                for (size_t i = 0; i < SliceD; i++)
                    result += (*this)[i];

                return result;
            }
        }

        template <size_t... P>
        Tensor<T, P...> reshape() const
        {
            static_assert(TensorUtils::get_size<P...>() == TensorUtils::get_size<SliceD, SliceD_...>(), "Tensor reshape error");

            Tensor<T, P...> result;
            memcpy(result.data, this->origin->data, TensorUtils::get_size<SliceD, SliceD_...>() * sizeof(T));

            return result;
        }

        template <size_t... P>
        TensorRef<Tensor<T, D, D_...>, Tensor<T, P...>> reshape_ref()
        {
            return TensorRef<Tensor<T, D, D_...>, Tensor<T, P...>>(*this);
        }

        This deref() const
        {
            This result;
            for (size_t i = 0; i < this->size; i++)
                result[i] = (*this)[i];
            return result;
        }

        template <class TDst, class Th, size_t i, size_t... TDim>
        static void assign_transpose(
            TDst &dst,
            const Th &src,
            std::array<size_t, sizeof...(TDim)> &indices)
        {
            if constexpr (i == sizeof...(TDim) - 1)
            {
                for (size_t j = 0; j < SliceD; j++) {
                    indices[i] = j;
                    dst.get(TensorUtils::get_transposed_index<sizeof...(TDim), 0, TDim...>(indices)) = src.get(indices);
                }
            }
            else
            {
                for (size_t j = 0; j < SliceD; j++) {
                    indices[i] = j;
                    TensorRef<Tensor<T, D, D_...>, Tensor<T, SliceD_...>>::template assign_transpose<TDst, Th, i + 1, TDim...>(dst, src, indices);
                }
            }
        }

        template <size_t N, size_t i = 0>
        T& get(const std::array<size_t, N> &indices)
        {
            if constexpr (i == N - 1)
                return (*this)[indices[i]];
            else
                return (*this)[indices[i]].template get<N, i + 1>(indices);
        }

        template <size_t N, size_t i = 0>
        const T& get(const std::array<size_t, N> &indices) const
        {
            if constexpr (N - 1 == i)
                return (*this)[indices[i]];
            else
                return (*this)[indices[i]].template get<N, (i + 1)>(indices);
        }

        template <size_t... TDim>
        typename TensorUtils::transpose_helper<Tensor<T, SliceD, SliceD_...>, Transpose<TDim...>>::type transpose() const
        {
            static_assert(TensorUtils::get_rank<TDim...>() == TensorUtils::get_rank<SliceD, SliceD_...>(), "Tensor transpose error");

            std::array<size_t, sizeof...(TDim)> indices;
            typename TensorUtils::transpose_helper<Tensor<T, SliceD, SliceD_...>, Transpose<TDim...>>::type result;
            assign_transpose<
                typename TensorUtils::transpose_helper<Tensor<T, SliceD, SliceD_...>, Transpose<TDim...>>::type,
                This,
                0,
                TDim...
            >(result, *this, indices);

            return result;
        }

        template <class Other>
        Tensor<Other, SliceD, SliceD_...> map(const std::function<Other(T)> &f) const
        {
            if constexpr (sizeof...(SliceD_))
            {
                Tensor<Other, SliceD, SliceD_...> result;
                for (size_t i = 0; i < SliceD; i++)
                    result[i] = (*this)[i].map(f);

                return result;
            }
            else
            {
                Tensor<Other, SliceD> result;
                for (size_t i = 0; i < SliceD; i++)
                    result[i] = f((*this)[i]);

                return result;
            }
        }

        template <class Other, size_t FD>
        typename TensorUtils::cond_apply<Other, sizeof...(SliceD_), FD, SliceD, SliceD_...>::type apply(const std::function<Tensor<Other, FD>(Tensor<T, FD>)> f) const
        {
            if constexpr (sizeof...(SliceD_) > 0)
            {
                Tensor<Other, SliceD, SliceD_...> result;
                for (size_t i = 0; i < SliceD; i++)
                    result[i] = (*this)[i].apply(f);

                return result;
            }
            else
            {
                Tensor<Other, FD> result = f(*this);
                return result;
            }
        }

        raw_iterator begin()
        {
            return raw_iterator(this->origin->data, slice_start);
        }

        raw_iterator end()
        {
            return raw_iterator(this->origin->data, slice_start, UINT64_MAX);
        }

    public:
        Tensor<T, D, D_...> *origin = nullptr;
        size_t slice_start = 0;
    };

    template <class T, size_t D, size_t... D_>
    class Tensor : public TensorRef<Tensor<T, D, D_...>, Tensor<T, D, D_...>>
    {
        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<This, This>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, This, sizeof...(D_), D_...>::type;

    public:
        Tensor(const T &value = T())
            : TensorRef<This, This>(this)
        {
            this->data = new T[TensorUtils::get_size<D, D_...>()];
            std::fill(this->data, this->data + TensorUtils::get_size<D, D_...>(), value);
        }

        Tensor(This &other)
            : TensorRef<This, This>(this)
        {
            if (other.data) {
                this->data = other.data;
                other.data = nullptr;
            }
        }

        template <class OtherOrigin>
        Tensor(const TensorRef<OtherOrigin, This> &other)
            : TensorRef<This, This>(this)
        {
            this->data = new T[TensorUtils::get_size<D, D_...>()];
            for (size_t i = 0; i < D; i++)
                (*this)[i] = other[i];
        }

        Tensor(const std::initializer_list<Sub> &list)
            : TensorRef<This, This>(this)
        {
            this->data = new T[TensorUtils::get_size<D, D_...>()];
            size_t idx = 0;
            for (const auto &sub : list)
                (*this)[idx++] = sub;
        }

        ~Tensor()
        {
            if (this->data)
                delete[] this->data;
        }

        class raw_iterator
        {
            T *data;
            size_t _start_index;
            size_t _index = 0;
            size_t _slice_index = 0;

        public:
            raw_iterator(T *data, size_t start, size_t _slice_index = 0) : data(data), _start_index(start) {
                if (_slice_index == UINT64_MAX)
                    _slice_index = TensorUtils::get_size<D, D_...>();
            }

            T &operator*() { return data[_start_index + _index]; }

            raw_iterator &operator++()
            {
                _index++;
                _slice_index++;
                return *this;
            }

            bool operator!=(const raw_iterator &other)
            {
                return this->_slice_index != other._slice_index;
            }

            bool operator==(const raw_iterator &other)
            {
                return this->_slice_index == other._slice_index;
            }
        };

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
            memcpy(this->data, other.data, sizeof(T) * TensorUtils::get_size<D, D_...>());
            return (*this);
        }

        T *data = nullptr;
    };

    template <class AOrigin, class BOrigin, class T, size_t D1, size_t D2, size_t D3>
    Tensor<T, D1, D3> dot(const TensorRef<AOrigin, Tensor<T, D1, D2>> &a, const TensorRef<BOrigin, Tensor<T, D2, D3>> &b)
    {
        Tensor<T, D1, D3> result;
        auto b_transposed = b.template transpose<1, 0>();

#pragma omp parallel for default(shared)
        for (int i = 0; i < D1; i++)
            for (int j = 0; j < D3; j++)
            {
                auto a_sub = a[i];
                auto b_sub = b_transposed[j];
                result[i][j] = std::inner_product(a_sub.begin(), a_sub.end(), b_sub.begin(), T());
            }

        return result;
    }

    template <class AOrigin, class BOrigin, class T, size_t D, size_t... D_>
    Tensor<T, D, D_...> hadamard(const TensorRef<AOrigin, Tensor<T, D, D_...>> &a, const TensorRef<BOrigin, Tensor<T, D, D_...>> &b)
    {
        Tensor<T, D, D_...> result;

        for (size_t i = 0; i < D; i++)
            result[i] = hadamard(a[i], b[i]);

        return result;
    }

    template <class AOrigin, class BOrigin, class T, size_t D1, size_t D2>
    Tensor<T, D1, D2> hadamard(const TensorRef<AOrigin, Tensor<T, D1, D2>> &a, const TensorRef<BOrigin, Tensor<T, D1, D2>> &b)
    {
        Tensor<T, D1, D2> result;

#pragma omp parallel for default(shared)
        for (int i = 0; i < D1; i++)
            for (int j = 0; j < D2; j++)
                result[i][j] = a[i][j] * b[i][j];

        return result;
    }

    template <class AOrigin, class BOrigin, class T, size_t D>
    Tensor<T, D> hadamard(const TensorRef<AOrigin, Tensor<T, D>> &a, const TensorRef<BOrigin, Tensor<T, D>> &b)
    {
        Tensor<T, D> result;

#pragma omp parallel for default(shared)
        for (int i = 0; i < D; i++)
            result[i] = a[i] * b[i];

        return result;
    }

    template <class IOrigin, class T, size_t Batch, size_t Feature, size_t IDim, size_t K>
    Tensor<T, K * K, Batch * Feature * IDim * IDim> im2col(const TensorRef<IOrigin, Tensor<T, Batch, Feature, IDim, IDim>> &origin)
    {
        Tensor<T, K * K, Batch * Feature * IDim * IDim> result;

        return result;
    }

    template <class IOrigin, class KOrigin, class T, size_t Batch, size_t FN, size_t C, size_t IDim, size_t KDim>
    Tensor<T, Batch, FN, IDim - KDim + 1, IDim - KDim + 1> conv(const TensorRef<IOrigin, Tensor<T, Batch, C, IDim, IDim>> &input, const TensorRef<KOrigin, Tensor<T, FN, C, KDim, KDim>> &kernel)
    {
        static_assert(IDim >= KDim, "Kernel must be same or bigger than input");

        Tensor<T, Batch, FN, IDim - KDim + 1, IDim - KDim + 1> result;
        return result;
    }

    template <class Origin, class T, size_t D>
    Tensor<T, D, D> flip(const TensorRef<Origin, Tensor<T, D, D>> &input)
    {
        Tensor<T, D, D> result;

#pragma omp parallel for default(shared)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                result[i][j] = input[D - i - 1][D - j - 1];

        return result;
    }

    template <class Origin, class T, size_t C, size_t D>
    Tensor<T, C, D, D> flip(const TensorRef<Origin, Tensor<T, C, D, D>> &input)
    {
        Tensor<T, C, D, D> result;

        for (size_t i = 0; i < C; i++)
            result[i] = flip(input[i]);

        return result;
    }

    template <class Origin, class T, size_t D, size_t P>
    Tensor<T, D + 2 * P> pad1d(const TensorRef<Origin, Tensor<T, D>> &input, T pad_value = T())
    {
        Tensor<T, D + 2 * P> result = pad_value;

        for (size_t i = 0; i < D; i++)
            result[P + i] = input[i];

        return result;
    }

    template <class Origin, class T, size_t D, size_t P>
    Tensor<T, D + 2 * P, D + 2 * P> pad2d(const TensorRef<Origin, Tensor<T, D, D>> &input, T pad_value = T())
    {
        Tensor<T, D + 2 * P, D + 2 * P> result{pad_value};

#pragma omp parallel for default(shared)
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++)
                result[P + i][P + j] = input[i][j];

        return result;
    }

    template <class Origin, class T, size_t C, size_t D, size_t P>
    Tensor<T, C, D + 2 * P, D + 2 * P> pad2d(const TensorRef<Origin, Tensor<T, C, D, D>> &input, T pad_value = T())
    {
        Tensor<T, C, D + 2 * P, D + 2 * P> result{pad_value};

        for (size_t i = 0; i < C; i++)
            result[i] = pad(input[i], pad_value);

        return result;
    }

    template <typename T, size_t IDim, size_t ODim>
    Tensor<T, ODim, ODim> pool(const Tensor<T, IDim, IDim> &input, std::function<T(const Tensor<T, IDim / ODim, IDim / ODim>&)> pool_func)
    {
        constexpr size_t KernelSize = IDim / ODim;
        Tensor<T, ODim, ODim> result;

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

    template <class Origin, class T, size_t IDim, size_t KDim>
    Tensor<T, IDim * KDim, IDim * KDim> unpool(const TensorRef<Origin, Tensor<T, IDim, IDim>> &input, std::function<Tensor<T, KDim, KDim>(T)> unpool_func)
    {
        constexpr size_t ODim = IDim * KDim;
        constexpr size_t KernelSize = KDim;
        Tensor<T, IDim * KDim, IDim * KDim> result;

        for (int i = 0; i < IDim; i++)
            for (int j = 0; j < IDim; j++)
            {
                Tensor<T, KernelSize, KernelSize> sub_input = unpool_func(input[i][j]);
                for (int k = 0; k < KernelSize; k++)
                    for (int l = 0; l < KernelSize; l++)
                        result[i * KernelSize + k][j * KernelSize + l] = sub_input[k][l];
            }

        return result;
    }

    template <class Origin, class T, size_t D>
    size_t argmax(const TensorRef<Origin, Tensor<T, D>> &t)
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

template <class Origin, class T, size_t D, size_t... D_>
std::ostream &operator<<(std::ostream &os, const SingleNet::template TensorRef<Origin, SingleNet::Tensor<T, D, D_...>> &tensor)
{
    os << "[";
    for (size_t i = 0; i < D; i++)
        os << tensor[i] << ((i == D - 1) ? "" : ", ");
    os << "]";

    return os;
}

#endif