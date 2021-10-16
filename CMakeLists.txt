cmake_minimum_required(VERSION 3.11)

project(
    SingleNet
    LANGUAGES CUDA CXX
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_compile_options(-std=c++17)
add_subdirectory(Core)

add_executable(
    SingleNet 
    main.cpp
)

target_link_libraries(
    SingleNet
    cudart
)

target_link_libraries(
    SingleNet
    Layers
    Datasets
    Defines
    Vector_cuda
)