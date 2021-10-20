add_library(Defines
    Defines.cu
)

set_target_properties(
    Defines
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)