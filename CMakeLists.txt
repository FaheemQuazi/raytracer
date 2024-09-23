cmake_minimum_required(VERSION 3.25)

project(raytracer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# VCPKG Deps
find_package(glm CONFIG REQUIRED)

# Include TIRA from the submodule
include(${CMAKE_SOURCE_DIR}/deps/tira/cmake/FindTIRA.cmake)
add_library(TIRA INTERFACE)
target_sources(TIRA PUBLIC
    FILE_SET HEADERS
        BASE_DIRS ${CMAKE_SOURCE_DIR}/deps/tira
)

# Add Assets
add_custom_target(copy-runtime-files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets)

# Configure Executable
add_executable(raytracer main.cpp)
target_include_directories(raytracer PUBLIC ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(raytracer PRIVATE glm::glm TIRA)