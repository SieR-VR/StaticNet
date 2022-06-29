cmake_minimum_required(VERSION 3.11)

project(
    SingleNet
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
endif()

add_subdirectory(Core)

add_executable(
    SingleNet 
    main.cpp
)

target_link_libraries(
    SingleNet
    Layers
    Datasets
    Defines
)