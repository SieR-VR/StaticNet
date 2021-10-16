add_compile_options(
    -std=c++17
    -g
    -G
    -Xcompiler
    -rdynamic
    -rdc=true)

add_library(
    Vector_cuda STATIC
    Vector_cuda.cu
)

set_target_properties(
    Vector_cuda
    PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)