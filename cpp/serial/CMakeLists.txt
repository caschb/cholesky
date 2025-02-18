cmake_minimum_required(VERSION 3.16)

project(CholeskySerial LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
add_executable(cholesky main.cpp)

set(CXX_STANDARD 17)

target_link_libraries(cholesky PRIVATE Eigen3::Eigen)
target_include_directories(cholesky PRIVATE Eigen3::Eigen)
