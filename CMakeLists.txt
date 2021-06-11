cmake_minimum_required(VERSION 3.4.3)
project(MyFixNet)

set(CMAKE_CXX_FLAGS "-m64 -mavx")

add_executable(MyFixNet
    main.cpp
    Tools/load_mnist.cpp
    Tools/vector_helper.cpp
    Float/LinearRegression.cpp
    Float/LogisticClassification.cpp
    Float/LogisticRegression.cpp
    Fixed/Fixed8bit.cpp
    Fixed/FixedLinearRegression.cpp
)