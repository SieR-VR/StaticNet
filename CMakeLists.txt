cmake_minimum_required(VERSION 3.11)

project(
    SingleNet
    LANGUAGES CUDA CXX
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -rdc=true -cudart=shared -lcudadevrt")

add_subdirectory(Core)

add_executable(
    SingleNet 
    main.cpp
)

set_target_properties(
    SingleNet PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CXX_STANDARD 17
)

target_link_libraries(
    SingleNet
    Layers
    Datasets
    Defines
    Vector
)