cmake_minimum_required(VERSION 3.18)
project(CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

add_executable(cudaExecute
    src/main.cpp
    src/kernels.cu
)

target_include_directories(cudaExecute PRIVATE include)

set_target_properties(cudaExecute PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
