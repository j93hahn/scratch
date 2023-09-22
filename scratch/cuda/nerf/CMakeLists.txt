# CMakeLists.txt - inspiration taken from f2nerf (https://github.com/Totoro97/f2-nerf/blob/main/CMakeLists.txt)

cmake_minimum_required(VERSION 3.18)
project(cudarf
        VERSION 0.1
        DESCRIPTION "Raytracing with CUDA"
        LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 14)

# setup CUDA compilation
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
