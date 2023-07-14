cmake_minimum_required(VERSION 3.12)
project(sali)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(OpenMP)
find_package(TBB REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O0 -Wall)

include_directories(${TBB_INCLUDE_DIRS})
include_directories(src/core)


add_executable(example_mt src/examples/example_multithread.cpp)

target_link_libraries(example_mt PUBLIC TBB::tbb OpenMP::OpenMP_CXX )
