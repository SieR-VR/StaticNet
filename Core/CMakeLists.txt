add_subdirectory(Datasets)
add_subdirectory(Defines)
add_subdirectory(Vector)

add_compile_options(-std=c++17)
add_library(Layers
    Layers.cpp
)