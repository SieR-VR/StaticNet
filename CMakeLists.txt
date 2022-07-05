cmake_minimum_required(VERSION 3.11)
project(StaticNet VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(StaticNet_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB StaticNet_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.h
)

add_library(StaticNet STATIC ${StaticNet_SOURCES})
target_include_directories(StaticNet PUBLIC ${StaticNet_INCLUDE_DIR})

include(CTest)
add_subdirectory(test)