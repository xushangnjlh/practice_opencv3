cmake_minimum_required(VERSION 2.8)
project(dense_mapping)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -fopenmp" )

add_executable(${PROJECT_NAME} "main")
add_executable(${PROJECT_NAME} "hello")
