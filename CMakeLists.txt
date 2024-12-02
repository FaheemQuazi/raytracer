cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_BIN_PATH "/usr/local/cuda")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_CXX_STANDARD 20)

project(raytracer LANGUAGES CUDA CXX)

# VCPKG Deps
find_package(glm CONFIG REQUIRED)

# X11 Deps
find_package(X11 REQUIRED)

# Include TIRA from the submodule
include(${CMAKE_SOURCE_DIR}/deps/tira/cmake/FindTIRA.cmake)
add_library(TIRA INTERFACE)
target_sources(TIRA PUBLIC
    FILE_SET HEADERS
        BASE_DIRS ${CMAKE_SOURCE_DIR}/deps/tira
)

# Configure Executable
add_executable(raytracer main.cpp)
target_include_directories(raytracer PUBLIC ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(raytracer PRIVATE glm::glm TIRA X11 png jpeg)
target_link_directories(raytracer PUBLIC /usr/local/cuda/lib64)

add_executable(cudatracer main_gpu.cpp rt.cu)
target_include_directories(cudatracer PUBLIC ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(cudatracer PRIVATE glm::glm TIRA X11)
target_link_directories(cudatracer PUBLIC /usr/local/cuda/lib64)
target_compile_options(cudatracer PRIVATE -std=c++20) 
