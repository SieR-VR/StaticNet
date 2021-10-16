add_library(
    Vector STATIC
    Vector.cu
)

set_target_properties(
    Vector
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "52"
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)