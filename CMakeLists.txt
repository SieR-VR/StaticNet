cmake_minimum_required(VERSION 3.11)
project(SingleNet VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(SINGLENET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB SINGLENET_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cc
)

add_library(singlenet STATIC ${SINGLENET_SOURCES})
target_include_directories(singlenet PUBLIC ${SINGLENET_INCLUDE_DIR})

add_subdirectory(test)