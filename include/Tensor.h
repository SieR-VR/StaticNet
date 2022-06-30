#ifndef TENSOR_H_
#define TENSOR_H_

#include <functional>
#include <initializer_list>
#include <type_traits>
#include <ostream>

namespace SingleNet {
    template <class T, size_t D, size_t ...D_>
    class Tensor;
    template <class T, size_t D, size_t ...D_>
    class TensorRef;

    namespace TensorUtils {
        template <size_t ...Dims>
        constexpr size_t get_size() {
            return (... * Dims);
        }

        template <size_t ...Dims>
        constexpr size_t get_dim() {
            return sizeof...(Dims);
        }

        template <class T, bool Cond, size_t ...Dims>
        struct sub_cond {
            typedef Tensor<T, Dims...> type;
        };

        template <class T, size_t ...Dims>
        struct sub_cond<T, false, Dims...> {
            typedef T type;
        };

        template <class T, bool Cond, size_t ...Dims>
        struct sub_cond_ref {
            typedef TensorRef<T, Dims...> type; 
        };

        template <class T, size_t ...Dims>
        struct sub_cond_ref<T, false, Dims...> {
            typedef T& type;
        };
    }

    template <class T, size_t D, size_t ...D_>
    class Tensor {
        friend class TensorRef<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<T, D, D_...>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;

    public:
        explicit Tensor(const This& other) {
            for (size_t i = 0; i < this->size; i++)
                data[i] = other.data[i];
        }

        Tensor(const std::initializer_list<Sub> &list) {
            size_t idx = 0;
            for (const auto &sub : list)
                (*this)[idx++] = Sub(sub);
        }

        ThisRef &ref() {
            return (*this);
        }

        SubRef operator[](size_t i) {
            if constexpr (sizeof...(D_)) {
                return SubRef(data + i * TensorUtils::get_size<D_...>());
            }
            else {
                return data[i];
            }
        }

    private:
        T data[TensorUtils::get_size<D, D_...>()];
    public:
        const size_t dimension = TensorUtils::get_dim<D, D_...>();
        const size_t dims[TensorUtils::get_dim<D, D_...>()] = { D, D_... };
        const size_t size = TensorUtils::get_size<D, D_...>();
    };

    template <class T, size_t D, size_t ...D_>
    class TensorRef {
        friend class Tensor<T, D, D_...>;

        using This = Tensor<T, D, D_...>;
        using ThisRef = TensorRef<T, D, D_...>;

        using Sub = typename TensorUtils::sub_cond<T, sizeof...(D_), D_...>::type;
        using SubRef = typename TensorUtils::sub_cond_ref<T, sizeof...(D_), D_...>::type;

    public:
        explicit TensorRef(const This& origin): data_start(origin.data) {}
        explicit TensorRef(T* data_start): data_start(data_start) {}

        ThisRef &operator=(const This &origin) {
            for (int i = 0; i < this->size; i++)
                data_start[i] = origin.data[i];

            return *this;
        }

        SubRef operator[](size_t i) {
            if constexpr (sizeof...(D_)) {
                return SubRef(data_start + i * TensorUtils::get_size<D_...>());
            }
            else {
                return data_start[i];
            }
        }

    private:
        T* data_start;
    public:
        const size_t dimension = TensorUtils::get_dim<D, D_...>();
        const size_t dims[TensorUtils::get_dim<D, D_...>()] = { D, D_... };
        const size_t size = TensorUtils::get_size<D, D_...>();
    };
}

template <typename T, size_t D, size_t ...D_>
std::ostream &operator<<(std::ostream &os, SingleNet::TensorRef<T, D, D_...> tensor) {
    os << "[";
    for (size_t i = 0; i < D; i++)
        os << tensor[i] << ((i == D-1) ? "" : ", ");
    os << "]";

    return os;
}

template <typename T, size_t D, size_t ...D_>
std::ostream &operator<<(std::ostream &os, SingleNet::Tensor<T, D, D_...>& tensor) {
    os << "[";
    for (size_t i = 0; i < D; i++)
        os << tensor[i] << ((i == D-1) ? "" : ", ");
    os << "]";

    return os;
}

#endif