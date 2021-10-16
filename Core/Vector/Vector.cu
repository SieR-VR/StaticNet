#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <stdexcept>

#include "Vector.h"

#define MAX_BLOCKS 256

cudaError_t cuda_check(cudaError_t err, int line, const char *file, bool abort = true)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + file + ":" + std::to_string(line));
    }
    return err;
}

#define CUDA_CHECK(err) cuda_check((err), __LINE__, __FILE__)

__device__ void CUDA_GetDim(size_t required_size, size_t *block, size_t *threads)
{
    *block = MAX_BLOCKS;
    *threads = required_size / MAX_BLOCKS + 1;
}

template <size_t N>
__global__ void CUDA_Add(void *v1, void *v2, void *v3, size_t *shape_reversed)
{
    int index = MAX_BLOCKS * threadIdx.x + blockIdx.x;

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
__global__ void CUDA_Sub(void *v1, void *v2, void *v3, size_t *shape_reversed)
{
    int index = MAX_BLOCKS * threadIdx.x + blockIdx.x;

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

template <size_t N>
__global__ void CUDA_Mul(void *v1, float *value, size_t *shape_reversed)
{
    int index = MAX_BLOCKS * threadIdx.x + blockIdx.x;

    if (shape_reversed[N - 1] <= index)
        return;

    if constexpr (N == 1)
    {
        (static_cast<float *>(v1))[index] *= *value;
    }
    else
    {
        size_t block, thread;
        CUDA_GetDim(shape_reversed[N - 2], &block, &thread);
        CUDA_Mul<N - 1><<<block, thread>>>(
            (static_cast<void **>(v1))[index], value, shape_reversed);
    }
}

__global__ void CUDA_Dot_1D(void *v1, void *v2, void *result, size_t size)
{
    int index = MAX_BLOCKS * threadIdx.x + blockIdx.x;

    if (size <= index)
        return;

    float temp = (static_cast<float *>(v1))[index] * (static_cast<float *>(v2))[index];
    atomicAdd(static_cast<float *>(result), temp);
}

__global__ void CUDA_Dot_2D1D(void *v1, void *v2, void *result, size_t size, size_t v1_size)
{
    int index = MAX_BLOCKS * threadIdx.x + blockIdx.x;

    if (index < v1_size)
    {
        size_t block, thread;
        CUDA_GetDim(size, &block, &thread);
        CUDA_Dot_1D<<<block, thread>>>(((void **)v1)[index], v2, &((float *)result)[index], size);
    }
}

__global__ void CUDA_Dot_2D(void *v1, void *v2, void *result, size_t size, size_t v1_size, size_t v2_size)
{
    int index = MAX_BLOCKS * threadIdx.x + blockIdx.x;

    if (index < v1_size) 
    {
        size_t block, thread;
        CUDA_GetDim(v2_size, &block, &thread);
        CUDA_Dot_2D1D<<<block, thread>>>(v2, ((void **)v1)[index], ((void **)result)[index], size, v2_size);
    }
}

namespace SingleNet
{

    template <size_t N>
    void *CUDA_Memcpy(void *src, const Vector<size_t, 1> &shape_reversed)
    {
        if constexpr (N == 1)
        {
            void *dst = nullptr;

            CUDA_CHECK(cudaMallocManaged(&dst, sizeof(float) * shape_reversed[0]));
            CUDA_CHECK(cudaMemcpy(dst, src, sizeof(float) * shape_reversed[0], cudaMemcpyHostToDevice));

            return dst;
        }
        else
        {
            void *dst_device, **dst_host;

            CUDA_CHECK(cudaMallocManaged(&dst_device, shape_reversed[N - 1] * sizeof(void *)));
            dst_host = (void **)malloc(shape_reversed[N - 1] * sizeof(void *));

            for (size_t i = 0; i < shape_reversed[N - 1]; i++)
            {
                dst_host[i] = CUDA_Memcpy<N - 1>((static_cast<void **>(src))[i], shape_reversed);
            }

            CUDA_CHECK(cudaMemcpy(dst_device, dst_host, shape_reversed[N - 1] * sizeof(void *), cudaMemcpyHostToDevice));
            free(dst_host);

            return dst_device;
        }

        return nullptr;
    }

    template <size_t N>
    void *CUDA_Malloc(const Vector<size_t, 1> &shape_reversed)
    {
        if constexpr (N == 1)
        {
            void *dst = nullptr;
            CUDA_CHECK(cudaMallocManaged(&dst, sizeof(float) * shape_reversed[0]));
            return dst;
        }
        else
        {
            void *dst_device, **dst_host;

            CUDA_CHECK(cudaMallocManaged(&dst_device, shape_reversed[N - 1] * sizeof(void *)));
            dst_host = (void **)malloc(shape_reversed[N - 1] * sizeof(void *));

            for (size_t i = 0; i < shape_reversed[N - 1]; i++)
            {
                dst_host[i] = CUDA_Malloc<N - 1>(shape_reversed);
            }

            CUDA_CHECK(cudaMemcpy(dst_device, dst_host, shape_reversed[N - 1] * sizeof(void *), cudaMemcpyHostToDevice));
            free(dst_host);

            return dst_device;
        }

        return nullptr;
    }

    template <size_t N>
    void CUDA_Free(void *device_pointer, const Vector<size_t, 1> &shape_reversed)
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
                CUDA_Free<N - 1>(host_pointer[i], shape_reversed);
            }

            CUDA_CHECK(cudaFree(device_pointer));
            free(host_pointer);
        }
    }

    template <size_t N>
    void *Host_Memcpy(void *device_src, const Vector<size_t, 1> &shape_reversed)
    {
        if constexpr (N == 1)
        {
            void *dst = malloc(sizeof(float) * shape_reversed[0]);
            CUDA_CHECK(cudaMemcpy(dst, device_src, sizeof(float) * shape_reversed[0], cudaMemcpyDeviceToHost));
            return dst;
        }
        else
        {
            void **device_src_copy = (void **)malloc(shape_reversed[N - 1] * sizeof(void *));
            void **dst_host = (void **)malloc(shape_reversed[N - 1] * sizeof(void *));

            cudaError_t err = cudaMemcpy(device_src_copy, device_src, shape_reversed[N - 1] * sizeof(void *), cudaMemcpyDeviceToHost);
            CUDA_CHECK(err);

            for (size_t i = 0; i < shape_reversed[N - 1]; i++)
            {
                dst_host[i] = Host_Memcpy<N - 1>(device_src_copy[i], shape_reversed);
            }

            free(device_src_copy);
            return dst_host;
        }

        return nullptr;
    }

    template <size_t N>
    void Host_Free(void *ptr, const Vector<size_t, 1> &shape_reversed)
    {
        if constexpr (N == 1)
        {
            free(ptr);
        }
        else
        {
            for (size_t i = 0; i < shape_reversed[N - 1]; i++)
            {
                Host_Free<N - 1>(
                    (static_cast<void **>(ptr))[i],
                    shape_reversed);
            }

            free(ptr);
        }
    }

    template <size_t N>
    Vector<float, N> operator+(const Vector<float, N> &v1, const Vector<float, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::runtime_error("Vector<float>::Shape mismatch");

        void *v1_ptr = to_pointer(v1);
        void *v2_ptr = to_pointer(v2);

        void *v1_ptr_device = CUDA_Memcpy<N>(v1_ptr, reverse(shape(v1)));
        void *v2_ptr_device = CUDA_Memcpy<N>(v2_ptr, reverse(shape(v2)));

        void *v3_ptr_device = CUDA_Malloc<N>(reverse(shape(v1)));

        void *shape_ptr = to_pointer(reverse(shape(v1)));
        size_t *shape_ptr_device;

        cudaMallocManaged(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, shape_ptr, sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t block = MAX_BLOCKS;
        size_t threads = (v1.size() / MAX_BLOCKS) + 1;

        CUDA_Add<N><<<block, threads>>>(v1_ptr_device, v2_ptr_device, v3_ptr_device, shape_ptr_device);

        void *v3_ptr = Host_Memcpy<N>(v3_ptr_device, reverse(shape(v1)));

        CUDA_Free<N>(v1_ptr_device, reverse(shape(v1)));
        CUDA_Free<N>(v2_ptr_device, reverse(shape(v2)));
        CUDA_Free<N>(v3_ptr_device, reverse(shape(v1)));
        cudaFree(shape_ptr_device);

        Vector<float, N> result = from_pointer<float, N>(v3_ptr, reverse(shape(v1)));
        return result;
    }

    template Vector<float, 3> operator+(const Vector<float, 3> &v1, const Vector<float, 3> &v2);
    template Vector<float, 2> operator+(const Vector<float, 2> &v1, const Vector<float, 2> &v2);
    template Vector<float, 1> operator+(const Vector<float, 1> &v1, const Vector<float, 1> &v2);

    template <size_t N>
    Vector<float, N> operator-(const Vector<float, N> &v1, const Vector<float, N> &v2)
    {
        if (shape(v1) != shape(v2))
            throw std::runtime_error("Vector<float>::Shape mismatch");

        void *v1_ptr = to_pointer(v1);
        void *v2_ptr = to_pointer(v2);

        void *v1_ptr_device = CUDA_Memcpy<N>(v1_ptr, reverse(shape(v1)));
        void *v2_ptr_device = CUDA_Memcpy<N>(v2_ptr, reverse(shape(v2)));

        void *v3_ptr_device = CUDA_Malloc<N>(reverse(shape(v1)));

        void *shape_ptr = to_pointer(reverse(shape(v1)));
        size_t *shape_ptr_device;

        cudaMallocManaged(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, shape_ptr, sizeof(size_t) * N, cudaMemcpyHostToDevice);

        size_t block = MAX_BLOCKS;
        size_t threads = (v1.size() / MAX_BLOCKS) + 1;

        CUDA_Sub<N><<<block, threads>>>(v1_ptr_device, v2_ptr_device, v3_ptr_device, shape_ptr_device);

        void *v3_ptr = Host_Memcpy<N>(v3_ptr_device, reverse(shape(v1)));

        CUDA_Free<N>(v1_ptr_device, reverse(shape(v1)));
        CUDA_Free<N>(v2_ptr_device, reverse(shape(v2)));
        CUDA_Free<N>(v3_ptr_device, reverse(shape(v1)));
        cudaFree(shape_ptr_device);

        Vector<float, N> result = from_pointer<float, N>(v3_ptr, reverse(shape(v1)));
        return result;
    }

    template Vector<float, 3> operator-(const Vector<float, 3> &v1, const Vector<float, 3> &v2);
    template Vector<float, 2> operator-(const Vector<float, 2> &v1, const Vector<float, 2> &v2);
    template Vector<float, 1> operator-(const Vector<float, 1> &v1, const Vector<float, 1> &v2);

    template <size_t N>
    Vector<float, N> &operator+=(Vector<float, N> &v1, const Vector<float, N> &v2)
    {
        v1 = v1 + v2;
        return v1;
    }

    template Vector<float, 3> &operator+=(Vector<float, 3> &v1, const Vector<float, 3> &v2);
    template Vector<float, 2> &operator+=(Vector<float, 2> &v1, const Vector<float, 2> &v2);
    template Vector<float, 1> &operator+=(Vector<float, 1> &v1, const Vector<float, 1> &v2);

    template <size_t N>
    Vector<float, N> &operator-=(Vector<float, N> &v1, const Vector<float, N> &v2)
    {
        v1 = v1 - v2;
        return v1;
    }

    template Vector<float, 3> &operator-=(Vector<float, 3> &v1, const Vector<float, 3> &v2);
    template Vector<float, 2> &operator-=(Vector<float, 2> &v1, const Vector<float, 2> &v2);
    template Vector<float, 1> &operator-=(Vector<float, 1> &v1, const Vector<float, 1> &v2);

    template <size_t N>
    Vector<float, N> operator*(const Vector<float, N> &v1, const float &s)
    {
        void *v1_ptr = to_pointer(v1);
        void *v1_ptr_device = CUDA_Memcpy<N>(v1_ptr, reverse(shape(v1)));

        void *shape_ptr = to_pointer(reverse(shape(v1)));
        size_t *shape_ptr_device;

        cudaMallocManaged(&shape_ptr_device, sizeof(size_t) * N);
        cudaMemcpy(shape_ptr_device, shape_ptr, sizeof(size_t) * N, cudaMemcpyHostToDevice);

        float *s_ptr_device;
        cudaMallocManaged(&s_ptr_device, sizeof(float));
        cudaMemcpy(s_ptr_device, &s, sizeof(float), cudaMemcpyHostToDevice);

        size_t block = MAX_BLOCKS;
        size_t threads = (v1.size() / MAX_BLOCKS) + 1;

        CUDA_Mul<N><<<block, threads>>>(v1_ptr_device, s_ptr_device, shape_ptr_device);

        void *result_ptr = Host_Memcpy<N>(v1_ptr_device, reverse(shape(v1)));

        CUDA_Free<N>(v1_ptr_device, reverse(shape(v1)));
        cudaFree(shape_ptr_device);
        cudaFree(s_ptr_device);

        Vector<float, N> result = from_pointer<float, N>(result_ptr, reverse(shape(v1)));
        return result;
    }

    template Vector<float, 3> operator*(const Vector<float, 3> &v1, const float &s);
    template Vector<float, 2> operator*(const Vector<float, 2> &v1, const float &s);
    template Vector<float, 1> operator*(const Vector<float, 1> &v1, const float &s);

    template <size_t N>
    Vector<float, N> &operator*=(Vector<float, N> &v1, const float &s)
    {
        v1 = v1 * s;
        return v1;
    }

    template Vector<float, 3> &operator*=(Vector<float, 3> &v1, const float &s);
    template Vector<float, 2> &operator*=(Vector<float, 2> &v1, const float &s);
    template Vector<float, 1> &operator*=(Vector<float, 1> &v1, const float &s);

    template <size_t N>
    Vector<float, N> operator/(const Vector<float, N> &v1, const float &s)
    {
        return v1 * (1.0f / s);
    }

    template Vector<float, 3> operator/(const Vector<float, 3> &v1, const float &s);
    template Vector<float, 2> operator/(const Vector<float, 2> &v1, const float &s);
    template Vector<float, 1> operator/(const Vector<float, 1> &v1, const float &s);

    template <size_t N>
    Vector<float, N> &operator/=(Vector<float, N> &v1, const float &s)
    {
        v1 = v1 * (1.0f / s);
        return v1;
    }

    template Vector<float, 3> &operator/=(Vector<float, 3> &v1, const float &s);
    template Vector<float, 2> &operator/=(Vector<float, 2> &v1, const float &s);
    template Vector<float, 1> &operator/=(Vector<float, 1> &v1, const float &s);

    float dot(const Vector<float, 1> &v1, const Vector<float, 1> &v2)
    {
        if (shape(v1) != shape(v2))
        {
            throw std::runtime_error("dot product error: shape mismatch");
        }

        void *v1_ptr = to_pointer(v1);
        void *v2_ptr = to_pointer(v2);

        void *v1_ptr_device = CUDA_Memcpy<1>(v1_ptr, shape(v1));
        void *v2_ptr_device = CUDA_Memcpy<1>(v2_ptr, shape(v2));

        void *result_ptr_device = nullptr;
        cudaMallocManaged(&result_ptr_device, sizeof(float));

        size_t block = MAX_BLOCKS;
        size_t threads = (v1.size() / MAX_BLOCKS) + 1;

        CUDA_Dot_1D<<<block, threads>>>(v1_ptr_device, v2_ptr_device, result_ptr_device, v1.size());
        cudaError_t err = cudaDeviceSynchronize();

        if (err != cudaSuccess)
        {
            throw std::runtime_error("dot product error: " + std::string(cudaGetErrorString(err)));
        }

        float result = 0.0f;
        cudaMemcpy(&result, result_ptr_device, sizeof(float), cudaMemcpyDeviceToHost);

        CUDA_Free<1>(v1_ptr_device, shape(v1));
        CUDA_Free<1>(v2_ptr_device, shape(v2));
        cudaFree(result_ptr_device);

        return result;
    }

    Vector<float, 1> dot(const Vector<float, 2> &v1, const Vector<float, 1> &v2)
    {
        if (shape(v1)[1] != shape(v2)[0])
        {
            throw std::runtime_error("dot product error: shape mismatch");
        }

        try
        {
            void *v1_ptr = to_pointer(v1);
            void *v2_ptr = to_pointer(v2);

            void *v1_ptr_device = CUDA_Memcpy<2>(v1_ptr, reverse(shape(v1)));
            void *v2_ptr_device = CUDA_Memcpy<1>(v2_ptr, shape(v2));

            void *result_ptr_device;
            cudaMallocManaged(&result_ptr_device, shape(v1)[0] * sizeof(float));

            size_t block = MAX_BLOCKS;
            size_t threads = (v1.size() / MAX_BLOCKS) + 1;

            CUDA_Dot_2D1D<<<block, threads>>>(v1_ptr_device, v2_ptr_device, result_ptr_device, shape(v1)[1], shape(v1)[0]);
            cudaError_t err = cudaDeviceSynchronize();

            if (err != cudaSuccess)
            {
                throw std::runtime_error("dot product error: " + std::string(cudaGetErrorString(err)));
            }

            void *result_ptr = malloc(shape(v1)[0] * sizeof(float));
            cudaMemcpy(result_ptr, result_ptr_device, shape(v1)[0] * sizeof(float), cudaMemcpyDeviceToHost);

            CUDA_Free<2>(v1_ptr_device, reverse(shape(v1)));
            CUDA_Free<1>(v2_ptr_device, shape(v2));
            cudaFree(result_ptr_device);

            Vector<float, 1> result = from_pointer<float, 1>(result_ptr, {shape(v1)[0]});
            return result;
        }
        catch (std::exception &e)
        {
            throw std::runtime_error("Vector<float, 1> dot():\n\t" + std::string(e.what()));
        }
    }

    Vector<float, 2> dot(const Vector<float, 2> &v1, const Vector<float, 2> &v2)
    {
        if (shape(v1)[1] != shape(v2)[0])
        {
            throw std::runtime_error("dot product error: shape mismatch");
        }
        try {
            Vector<float, 2> v2_transposed = transpose(v2);

            void *v1_ptr = to_pointer(v1);
            void *v2_ptr = to_pointer(v2_transposed);

            void *v1_ptr_device = CUDA_Memcpy<2>(v1_ptr, reverse(shape(v1)));
            void *v2_ptr_device = CUDA_Memcpy<2>(v2_ptr, reverse(shape(v2_transposed)));

            void *result_ptr_device = CUDA_Malloc<2>({shape(v2)[1], shape(v1)[0]});

            size_t block = MAX_BLOCKS;
            size_t threads = (v1.size() / MAX_BLOCKS) + 1;

            CUDA_Dot_2D<<<block, threads>>>(v1_ptr_device, v2_ptr_device, result_ptr_device, shape(v1)[1], shape(v1)[0], shape(v2)[1]);
            cudaError_t err = cudaDeviceSynchronize();

            if (err != cudaSuccess)
            {
                throw std::runtime_error("dot product error: " + std::string(cudaGetErrorString(err)));
            }

            void *result_ptr = Host_Memcpy<2>(result_ptr_device, {shape(v2)[1], shape(v1)[0]});

            CUDA_Free<2>(v1_ptr_device, reverse(shape(v1)));
            CUDA_Free<2>(v2_ptr_device, shape(v2));
            CUDA_Free<2>(result_ptr_device, {shape(v2)[1], shape(v1)[0]});

            Vector<float, 2> result = from_pointer<float, 2>(result_ptr, {shape(v2)[1], shape(v1)[0]});
            return result;
        }
        catch (std::exception &e)
        {
            throw std::runtime_error("Vector<float, 2> dot():\n\t" + std::string(e.what()));
        }
    }
}