#include "VectorCUDA.h"

cudaError_t cuda_check(cudaError_t err, int line, const char *file, bool abort = true)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + file + ":" + std::to_string(line));
    }
    return err;
}

#define CUDA_CHECK(err) cuda_check((err), __LINE__, __FILE__)
#define MAX_THREADS_PER_BLOCK 1024

__device__ void CUDA_GetDim(size_t required_size, size_t *block, size_t *threads)
{
    *threads = (required_size > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : required_size;
    *block = (required_size / *threads) + 1;
}

template <size_t N>
__global__ void CUDA_Add(void *v1, void *v2, void *v3, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v3))[index] = (static_cast<float *>(v1))[index] + (static_cast<float *>(v2))[index];
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_Add<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index],
            (static_cast<void **>(v2))[index],
            (static_cast<void **>(v3))[index], shape_reversed);
    }
}

template <size_t N>
__global__ void CUDA_AddAssign(void *v1, void *v2, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v1))[index] += (static_cast<float *>(v2))[index];
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_AddAssign<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index],
            (static_cast<void **>(v2))[index], shape_reversed);
    }
}

__global__ void CUDA_Add_bias(void *v1, void *v2, void *v3, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[1] <= index)
        return;

    size_t block, thread;
    CUDA_GetDim(shape_reversed[0], &block, &thread);
    CUDA_Add<1><<<block, thread>>>(
        (static_cast<void **>(v1))[index], v2,
        (static_cast<void **>(v3))[index], shape_reversed);
}

__global__ void CUDA_Add_bias_assign(void *v1, void *v2, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[1] <= index)
        return;

    size_t block, thread;
    CUDA_GetDim(shape_reversed[0], &block, &thread);
    CUDA_AddAssign<1><<<block, thread>>>(
        (static_cast<void **>(v1))[index], v2, shape_reversed);
}

template <size_t N>
__global__ void CUDA_Sub(void *v1, void *v2, void *v3, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v3))[index] = (static_cast<float *>(v1))[index] - (static_cast<float *>(v2))[index];
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_Sub<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index],
            (static_cast<void **>(v2))[index],
            (static_cast<void **>(v3))[index], shape_reversed);
    }
}

__global__ void CUDA_Sub_bias(void *v1, void *v2, void *v3, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[1] <= index)
        return;

    size_t block, thread;
    CUDA_GetDim(shape_reversed[0], &block, &thread);
    CUDA_Sub<1><<<block, thread>>>(
        (static_cast<void **>(v1))[index], v2,
        (static_cast<void **>(v3))[index], shape_reversed);
}

template <size_t N>
__global__ void CUDA_SubAssign(void *v1, void *v2, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v1))[index] -= (static_cast<float *>(v2))[index];
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_SubAssign<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index],
            (static_cast<void **>(v2))[index], shape_reversed);
    }
}

__global__ void CUDA_Sub_bias_assign(void *v1, void *v2, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[1] <= index)
        return;

    size_t block, thread;
    CUDA_GetDim(shape_reversed[0], &block, &thread);
    CUDA_SubAssign<1><<<block, thread>>>(
        (static_cast<void **>(v1))[index], v2, shape_reversed);
}

template <size_t N>
__global__ void CUDA_Mul(void *src, void *dst, float value, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(dst))[index] = (static_cast<float *>(src))[index] * value;
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_Mul<N - 1><<<block, thread>>>(
            (static_cast<void **>(src))[index],
            (static_cast<void **>(dst))[index], value, shape_reversed);
    }
}

template <size_t N>
__global__ void CUDA_MulAssign(void *v1, float value, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v1))[index] *= value;
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_MulAssign<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index], value, shape_reversed);
    }
}

template <size_t N>
__global__ void CUDA_Times(void *v1, void *v2, void *v3, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v3))[index] = (static_cast<float *>(v1))[index] * (static_cast<float *>(v2))[index];
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_Times<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index],
            (static_cast<void **>(v2))[index],
            (static_cast<void **>(v3))[index], shape_reversed);
    }
}

template <size_t N>
__global__ void CUDA_Copy(void *v1, void *v2, size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v2))[index] = (static_cast<float *>(v1))[index];
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_Copy<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index],
            (static_cast<void **>(v2))[index], shape_reversed);
    }
}

template <size_t N>
__global__ void CUDA_Map(void *v1, void *v2, float (*func)(float), size_t *shape_reversed)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v2))[index] = func((static_cast<float *>(v1))[index]);
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_Map<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index],
            (static_cast<void **>(v2))[index],
            func, shape_reversed);
    }
}

__global__ void CUDA_Transpose_helper(void *v1, void *v2, size_t *shape, size_t index_previous)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape[1] <= index)
        return;

    (static_cast<float **>(v2))[index][index_previous] = (static_cast<float **>(v1))[index_previous][index];
}

__global__ void CUDA_Transpose(void *v1, void *v2, size_t *shape)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (shape[0] <= index)
        return;

    size_t block, thread;
    CUDA_GetDim(shape[1], &block, &thread);
    CUDA_Transpose_helper<<<block, thread>>>(v1, v2, shape, index);
}

__global__ void CUDA_Dot_1D(void *v1, void *v2, void *result, size_t size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (size <= index)
        return;

    float temp = (static_cast<float *>(v1))[index] * (static_cast<float *>(v2))[index];
    atomicAdd(static_cast<float *>(result), temp);
}

__global__ void CUDA_Dot_2D1D(void *v1, void *v2, void *result, size_t size, size_t v1_size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < v1_size)
    {
        size_t block, thread;
        CUDA_GetDim(size, &block, &thread);
        CUDA_Dot_1D<<<block, thread>>>(((void **)v1)[index], v2, &((float *)result)[index], size);
    }
}

__global__ void CUDA_Dot_2D(void *v1, void *v2, void *result, size_t size, size_t v1_size, size_t v2_size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < v1_size)
    {
        size_t block, thread;
        CUDA_GetDim(v2_size, &block, &thread);
        CUDA_Dot_2D1D<<<block, thread>>>(v2, ((void **)v1)[index], ((void **)result)[index], size, v2_size);
    }
}

__global__ void CUDA_Sum_1D(void *v1, void *result, size_t size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (size <= index)
        return;

    atomicAdd(static_cast<float *>(result), (static_cast<float *>(v1))[index]);
}

__global__ void CUDA_Sum_2D1D(void *v1, void *result, size_t size, size_t v1_size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if (index < v1_size)
    {
        size_t block, thread;
        CUDA_GetDim(size, &block, &thread);
        CUDA_Sum_1D<<<block, thread>>>(((void **)v1)[index], &((float *)result)[index], size);
    }
}

namespace SingleNet
{
    template <size_t N>
    void *malloc_device(const Vector<size_t, 1> &shape_reversed)
    {
        if constexpr (N == 1)
        {
            void *dst = nullptr;
            CUDA_CHECK(cudaMalloc(&dst, sizeof(float) * shape_reversed[0]));
            return dst;
        }
        else
        {
            void *dst_device, **dst_host;

            CUDA_CHECK(cudaMalloc(&dst_device, shape_reversed[N - 1] * sizeof(void *)));
            dst_host = (void **)malloc(shape_reversed[N - 1] * sizeof(void *));

            for (size_t i = 0; i < shape_reversed[N - 1]; i++)
            {
                dst_host[i] = malloc_device<N - 1>(shape_reversed);
            }

            CUDA_CHECK(cudaMemcpy(dst_device, dst_host, shape_reversed[N - 1] * sizeof(void *), cudaMemcpyHostToDevice));
            free(dst_host);

            return dst_device;
        }

        return nullptr;
    }

    template <size_t N>
    void free_device(void *device_pointer, const Vector<size_t, 1> &shape_reversed)
    {
        if constexpr (N == 1)
        {
            CUDA_CHECK(cudaFree(device_pointer));
        }
        else
        {
            void **host_pointer = (void **)malloc(sizeof(void *) * shape_reversed[N - 1]);
            CUDA_CHECK(cudaMemcpy(host_pointer, device_pointer, sizeof(void *) * shape_reversed[N - 1], cudaMemcpyDeviceToHost));

            for (size_t i = 0; i < shape_reversed[N - 1]; i++)
            {
                free_device<N - 1>(host_pointer[i], shape_reversed);
            }

            CUDA_CHECK(cudaFree(device_pointer));
            free(host_pointer);
        }
    }

    template <size_t N>
    void *memcpy_device(const Vector<size_t, 1> &shape_reversed, const Vector<float, N> &v)
    {
        if constexpr (N == 1)
        {
            void *dst = nullptr;
            CUDA_CHECK(cudaMalloc(&dst, sizeof(float) * shape_reversed[0]));
            CUDA_CHECK(cudaMemcpy(dst, v.data(), sizeof(float) * shape_reversed[0], cudaMemcpyHostToDevice));
            return dst;
        }
        else
        {
            void *dst_device, **dst_host;

            CUDA_CHECK(cudaMalloc(&dst_device, shape_reversed[N - 1] * sizeof(void *)));
            dst_host = (void **)malloc(shape_reversed[N - 1] * sizeof(void *));

            for (size_t i = 0; i < shape_reversed[N - 1]; i++)
            {
                dst_host[i] = memcpy_device<N - 1>(shape_reversed, v[i]);
            }

            CUDA_CHECK(cudaMemcpy(dst_device, dst_host, shape_reversed[N - 1] * sizeof(void *), cudaMemcpyHostToDevice));
            free(dst_host);

            return dst_device;
        }

        return nullptr;
    }

    template <size_t N>
    Vector<float, N> memcpy_host(const Vector<size_t, 1> &shape_reversed, void *device_pointer)
    {
        if constexpr (N == 1)
        {
            Vector<float, 1> v(shape_reversed[0]);
            CUDA_CHECK(cudaMemcpy(v.data(), device_pointer, sizeof(float) * shape_reversed[0], cudaMemcpyDeviceToHost));
            return v;
        }
        else
        {
            void **host_pointer = (void **)malloc(sizeof(void *) * shape_reversed[N - 1]);
            CUDA_CHECK(cudaMemcpy(host_pointer, device_pointer, sizeof(void *) * shape_reversed[N - 1], cudaMemcpyDeviceToHost));

            Vector<float, N> v(shape_reversed[N - 1]);
            for (size_t i = 0; i < shape_reversed[N - 1]; i++)
            {
                v[i] = memcpy_host<N - 1>(shape_reversed, host_pointer[i]);
            }

            free(host_pointer);
            return v;
        }

        return Vector<float, N>();
    }

    template <size_t N>
    VectorCUDA<N>::VectorCUDA()
    {
        this->m_pDeviceData = nullptr;
        this->m_shape = Vector<size_t, 1>();
    }

    template <size_t N>
    VectorCUDA<N>::VectorCUDA(const Vector<float, N> &v)
    {
        this->m_pDeviceData = memcpy_device(reverse(shape(v)), v);
        this->m_shape = shape(v);
    }

    template <size_t N>
    VectorCUDA<N>::VectorCUDA(const VectorCUDA<N> &v)
    {
        *this = v.copy();
    }

    template <size_t N>
    VectorCUDA<N>::VectorCUDA(VectorCUDA<N> &v)
    {
        this->m_pDeviceData = v.m_pDeviceData;
        this->m_shape = v.m_shape;

        v.m_pDeviceData = nullptr;
        v.m_shape = Vector<size_t, 1>();
    }

    template <size_t N>
    VectorCUDA<N>::~VectorCUDA()
    {
        if (this->m_pDeviceData)
            free_device<N>(this->m_pDeviceData, (this->m_shape));
    }

    template <size_t N>
    VectorCUDA<N> &VectorCUDA<N>::operator=(const Vector<float, N> &v)
    {
        if (this->m_pDeviceData)
            free_device<N>(this->m_pDeviceData, (this->m_shape));

        this->m_pDeviceData = memcpy_device(reverse(shape(v)), v);
        this->m_shape = shape(v);

        return *this;
    }
    
    template <size_t N>
    VectorCUDA<N> &VectorCUDA<N>::operator=(const VectorCUDA<N> &v)
    {
        *this = v.copy();
        return *this;
    }

    template <size_t N>
    VectorCUDA<N> &VectorCUDA<N>::operator=(VectorCUDA<N> &v)
    {
        if (this->m_pDeviceData)
            free_device<N>(this->m_pDeviceData, (this->m_shape));

        this->m_pDeviceData = v.m_pDeviceData;
        this->m_shape = v.m_shape;

        v.m_pDeviceData = nullptr;
        v.m_shape = Vector<size_t, 1>();

        return *this;
    }

    template <size_t N>
    VectorCUDA<N> VectorCUDA<N>::operator+(const VectorCUDA<N> &v) const
    {
        if (this->m_shape != v.m_shape)
        {
            throw std::runtime_error("VectorCUDA<N>::operator+: shape not match");
        }

        VectorCUDA<N> dst;
        dst.m_pDeviceData = malloc_device<N>(reverse(this->m_shape));
        dst.m_shape = this->m_shape;

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(this->m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (m_shape[0] < MAX_THREADS_PER_BLOCK) ? m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (m_shape[0] / threads) + 1;

        CUDA_Add<N><<<blocks, threads>>>(this->m_pDeviceData, v.m_pDeviceData, dst.m_pDeviceData, shape_ptr_device);

        cudaFree(shape_ptr_device);
        return dst;
    }

    template <size_t N>
    VectorCUDA<N> VectorCUDA<N>::operator-(const VectorCUDA<N> &v) const
    {
        if (this->m_shape != v.m_shape)
        {
            throw std::runtime_error("VectorCUDA<N>::operator-: shape not match");
        }

        VectorCUDA<N> dst;
        dst.m_pDeviceData = malloc_device<N>(reverse(this->m_shape));
        dst.m_shape = this->m_shape;

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(this->m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (m_shape[0] < MAX_THREADS_PER_BLOCK) ? m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (m_shape[0] / threads) + 1;

        CUDA_Sub<N><<<blocks, threads>>>(this->m_pDeviceData, v.m_pDeviceData, dst.m_pDeviceData, shape_ptr_device);

        cudaFree(shape_ptr_device);
        return dst;
    }

    template <size_t N>
    VectorCUDA<N> &VectorCUDA<N>::operator+=(const VectorCUDA<N> &v)
    {
        if (this->m_shape != v.m_shape)
        {
            throw std::runtime_error("VectorCUDA<N>::operator+=: shape not match");
        }

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(this->m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (m_shape[0] < MAX_THREADS_PER_BLOCK) ? m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (m_shape[0] / threads) + 1;

        CUDA_AddAssign<N><<<blocks, threads>>>(this->m_pDeviceData, v.m_pDeviceData, shape_ptr_device);
        cudaFree(shape_ptr_device);

        return *this;
    }

    template <size_t N>
    VectorCUDA<N> &VectorCUDA<N>::operator-=(const VectorCUDA<N> &v)
    {
        if (this->m_shape != v.m_shape)
        {
            throw std::runtime_error("VectorCUDA<N>::operator-=: shape not match");
        }

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(this->m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (m_shape[0] < MAX_THREADS_PER_BLOCK) ? m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (m_shape[0] / threads) + 1;

        CUDA_SubAssign<N><<<blocks, threads>>>(this->m_pDeviceData, v.m_pDeviceData, shape_ptr_device);
        cudaFree(shape_ptr_device);

        return *this;
    }

    template <size_t N>
    VectorCUDA<N> VectorCUDA<N>::operator*(const float &s) const
    {
        VectorCUDA<N> dst;
        dst.m_pDeviceData = malloc_device<N>(reverse(this->m_shape));
        dst.m_shape = this->m_shape;

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(this->m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (m_shape[0] < MAX_THREADS_PER_BLOCK) ? m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (m_shape[0] / threads) + 1;

        CUDA_Mul<N><<<blocks, threads>>>(this->m_pDeviceData, dst.m_pDeviceData, s, shape_ptr_device);

        cudaFree(shape_ptr_device);
        return dst;
    }

    template <size_t N>
    VectorCUDA<N> VectorCUDA<N>::operator/(const float &s) const
    {
        if (s == 0)
        {
            throw std::runtime_error("VectorCUDA<N>::operator/: divisor is zero");
        }
        return (*this * (1.0f / s));
    }

    template <size_t N>
    VectorCUDA<N> &VectorCUDA<N>::operator*=(const float &s)
    {
        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(this->m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (m_shape[0] < MAX_THREADS_PER_BLOCK) ? m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (m_shape[0] / threads) + 1;

        CUDA_MulAssign<N><<<blocks, threads>>>(this->m_pDeviceData, s, shape_ptr_device);

        cudaFree(shape_ptr_device);
        return *this;
    }

    template <size_t N>
    VectorCUDA<N> &VectorCUDA<N>::operator/=(const float &s)
    {
        if (s == 0)
        {
            throw std::runtime_error("VectorCUDA<N>::operator/=: divisor is zero");
        }

        *this *= (1.0f / s);
        return *this;
    }

    VectorCUDA<2> operator+(const VectorCUDA<2> &v1, const VectorCUDA<1> &v2)
    {
        if (v1.m_shape[1] != v2.m_shape[0])
        {
            throw std::runtime_error("VectorCUDA<2>::operator+: shape not match");
        }

        VectorCUDA<2> dst;
        dst.m_pDeviceData = malloc_device<2>(v1.m_shape);
        dst.m_shape = v1.m_shape;

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * 2);
        cudaMemcpy(shape_ptr_device, reverse(v1.m_shape).data(), sizeof(size_t) * 2, cudaMemcpyHostToDevice);

        size_t threads = (v1.m_shape[0] < MAX_THREADS_PER_BLOCK) ? v1.m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (v1.m_shape[0] / threads) + 1;

        CUDA_Add_bias<<<blocks, threads>>>(v1.m_pDeviceData, v2.m_pDeviceData, dst.m_pDeviceData, shape_ptr_device);
        cudaFree(shape_ptr_device);

        return dst;
    }

    VectorCUDA<2> operator-(const VectorCUDA<2> &v1, const VectorCUDA<1> &v2)
    {
        if (v1.m_shape[1] != v2.m_shape[0])
        {
            throw std::runtime_error("VectorCUDA<2>::operator-: shape not match");
        }

        VectorCUDA<2> dst;
        dst.m_pDeviceData = malloc_device<2>(v1.m_shape);
        dst.m_shape = v1.m_shape;

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * 2);
        cudaMemcpy(shape_ptr_device, reverse(v1.m_shape).data(), sizeof(size_t) * 2, cudaMemcpyHostToDevice);

        size_t threads = (v1.m_shape[0] < MAX_THREADS_PER_BLOCK) ? v1.m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (v1.m_shape[0] / threads) + 1;

        CUDA_Sub_bias<<<blocks, threads>>>(v1.m_pDeviceData, v2.m_pDeviceData, dst.m_pDeviceData, shape_ptr_device);
        cudaFree(shape_ptr_device);

        return dst;
    }

    VectorCUDA<2> &operator+=(VectorCUDA<2> &v1, const VectorCUDA<1> &v2)
    {
        if (v1.m_shape[1] != v2.m_shape[0])
        {
            throw std::runtime_error("VectorCUDA<2>::operator+=: shape not match");
        }

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * 2);
        cudaMemcpy(shape_ptr_device, reverse(v1.m_shape).data(), sizeof(size_t) * 2, cudaMemcpyHostToDevice);

        size_t threads = (v1.m_shape[0] < MAX_THREADS_PER_BLOCK) ? v1.m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (v1.m_shape[0] / threads) + 1;

        CUDA_Add_bias_assign<<<blocks, threads>>>(v1.m_pDeviceData, v2.m_pDeviceData, shape_ptr_device);
        cudaFree(shape_ptr_device);

        return v1;
    }

    VectorCUDA<2> &operator-=(VectorCUDA<2> &v1, const VectorCUDA<1> &v2)
    {
        if (v1.m_shape[1] != v2.m_shape[0])
        {
            throw std::runtime_error("VectorCUDA<2>::operator-=: shape not match");
        }

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * 2);
        cudaMemcpy(shape_ptr_device, reverse(v1.m_shape).data(), sizeof(size_t) * 2, cudaMemcpyHostToDevice);

        size_t threads = (v1.m_shape[0] < MAX_THREADS_PER_BLOCK) ? v1.m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (v1.m_shape[0] / threads) + 1;

        CUDA_Sub_bias_assign<<<blocks, threads>>>(v1.m_pDeviceData, v2.m_pDeviceData, shape_ptr_device);
        cudaFree(shape_ptr_device);

        return v1;
    }

    template <size_t N>
    VectorCUDA<N> VectorCUDA<N>::map(float (*func)(float)) const
    {
        VectorCUDA<N> dst;
        dst.m_pDeviceData = malloc_device<N>(reverse(this->m_shape));
        dst.m_shape = this->m_shape;

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(this->m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (m_shape[0] < MAX_THREADS_PER_BLOCK) ? m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (m_shape[0] / threads) + 1;

        CUDA_Map<N><<<blocks, threads>>>(this->m_pDeviceData, dst.m_pDeviceData, func, shape_ptr_device);

        cudaFree(shape_ptr_device);
        return dst;
    }

    template <size_t N>
    VectorCUDA<N> VectorCUDA<N>::copy() const
    {
        VectorCUDA<N> dst;
        dst.m_pDeviceData = malloc_device<N>(reverse(this->m_shape));
        dst.m_shape = this->m_shape;

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(this->m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (m_shape[0] < MAX_THREADS_PER_BLOCK) ? m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (m_shape[0] / threads) + 1;

        CUDA_Copy<N><<<blocks, threads>>>(this->m_pDeviceData, dst.m_pDeviceData, shape_ptr_device);

        cudaFree(shape_ptr_device);
        return dst;
    }

    VectorCUDA<1> mean(const VectorCUDA<2> &v)
    {
        VectorCUDA<1> dst;
        dst.m_pDeviceData = malloc_device<1>({v.m_shape[1]});
        dst.m_shape = {v.m_shape[1]};

        size_t threads = (v.m_shape[0] < MAX_THREADS_PER_BLOCK) ? v.m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (v.m_shape[0] / threads) + 1;

        CUDA_Sum_2D1D<<<blocks, threads>>>(v.m_pDeviceData, dst.m_pDeviceData, v.m_shape[1], v.m_shape[0]);
        return dst;
    }

    VectorCUDA<2> transpose(const VectorCUDA<2> &v)
    {
        VectorCUDA<2> dst;
        dst.m_pDeviceData = malloc_device<2>(v.m_shape);
        dst.m_shape = reverse(v.m_shape);

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * 2);
        cudaMemcpy(shape_ptr_device, v.m_shape.data(), sizeof(size_t) * 2, cudaMemcpyHostToDevice);

        size_t threads = (v.m_shape[0] < MAX_THREADS_PER_BLOCK) ? v.m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (v.m_shape[0] / threads) + 1;

        CUDA_Transpose<<<blocks, threads>>>(v.m_pDeviceData, dst.m_pDeviceData, shape_ptr_device);

        cudaFree(shape_ptr_device);
        return dst;
    }

    VectorCUDA<2> dot(const VectorCUDA<2> &v1, const VectorCUDA<2> &v2)
    {
        if (v1.m_shape[1] != v2.m_shape[0])
        {
            throw std::runtime_error("VectorCUDA<2>::dot: shape not match");
        }

        VectorCUDA<2> dst;
        dst.m_pDeviceData = malloc_device<2>({v2.m_shape[1], v1.m_shape[0]});
        dst.m_shape = {v1.m_shape[0], v2.m_shape[1]};

        size_t threads = (v1.m_shape[0] < MAX_THREADS_PER_BLOCK) ? v1.m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (v1.m_shape[0] / threads) + 1;

        CUDA_Dot_2D<<<blocks, threads>>>(v1.m_pDeviceData, v2.m_pDeviceData, dst.m_pDeviceData, v1.m_shape[1], v1.m_shape[0], v2.m_shape[1]);

        return dst;
    }

    template <size_t N>
    VectorCUDA<N> times(const VectorCUDA<N> &v1, const VectorCUDA<N> &v2)
    {
        if (v1.m_shape != v2.m_shape)
        {
            throw std::runtime_error("VectorCUDA<N>::times: shape not match");
        }

        VectorCUDA<N> dst;
        dst.m_pDeviceData = malloc_device<N>(v1.m_shape);
        dst.m_shape = v1.m_shape;

        size_t *shape_ptr_device;
        cudaMalloc(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, reverse(v1.m_shape).data(), sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t threads = (v1.m_shape[0] < MAX_THREADS_PER_BLOCK) ? v1.m_shape[0] : MAX_THREADS_PER_BLOCK;
        size_t blocks = (v1.m_shape[0] / threads) + 1;

        CUDA_Times<N><<<blocks, threads>>>(v1.m_pDeviceData, v2.m_pDeviceData, dst.m_pDeviceData, shape_ptr_device);

        cudaFree(shape_ptr_device);
        return dst;
    }

    template VectorCUDA<3> times(const VectorCUDA<3> &v1, const VectorCUDA<3> &v2);
    template VectorCUDA<2> times(const VectorCUDA<2> &v1, const VectorCUDA<2> &v2);
    template VectorCUDA<1> times(const VectorCUDA<1> &v1, const VectorCUDA<1> &v2);

    template <size_t N>
    Vector<float, N> to_cpu(const VectorCUDA<N> &v)
    {
        return memcpy_host<N>(reverse(v.m_shape), v.m_pDeviceData);
    }

    template Vector<float, 3> to_cpu(const VectorCUDA<3> &v);
    template Vector<float, 2> to_cpu(const VectorCUDA<2> &v);
    template Vector<float, 1> to_cpu(const VectorCUDA<1> &v);
}

template class SingleNet::VectorCUDA<3>;
template class SingleNet::VectorCUDA<2>;
template class SingleNet::VectorCUDA<1>;
