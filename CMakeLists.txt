cmake_minimum_required(VERSION 3.25)

project(raytracer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

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
target_link_libraries(raytracer PRIVATE glm::glm TIRA X11)