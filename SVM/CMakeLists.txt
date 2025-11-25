cmake_minimum_required(VERSION 3.10)

project(SVM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

add_executable(svmExecute
    src/main.cpp
)

target_include_directories(svmExecute PRIVATE include)