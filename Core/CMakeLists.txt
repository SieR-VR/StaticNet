add_subdirectory(Datasets)
add_subdirectory(Defines)
add_subdirectory(Vector)

add_library(Layers
    Layers.cu
)

set_target_properties(
    Layers
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
)