cmake_minimum_required(VERSION 3.23)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(sparsegpu CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
#set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_ARCHITECTURES 75)
#set(CMAKE_CUDA_ARCHITECTURES all)

add_library(sparsegpu)
include_directories(include)
add_subdirectory(include)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)

