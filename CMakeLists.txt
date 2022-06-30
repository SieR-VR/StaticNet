cmake_minimum_required(VERSION 3.11)
project(SingleNet VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(SINGLENET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SINGLENET_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cc
)

add_library(singlenet STATIC ${SINGLENET_SOURCES})
target_include_directories(singlenet PUBLIC ${SINGLENET_INCLUDE_DIR})

add_subdirectory(test)