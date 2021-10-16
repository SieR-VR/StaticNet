add_subdirectory(Datasets)
add_subdirectory(Defines)
add_subdirectory(Vector)

add_library(Layers
    Layers.cpp
)

set_target_properties(
    Layers PROPERTIES
    CXX_STANDARD 17
)