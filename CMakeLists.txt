cmake_minimum_required(VERSION 2.8.3)
project(vi_ekf)

set(CMAKE_BUILD_TYPE Release)

message("CMAKE_C_FLAGS_RELEASE is ${CMAKE_C_FLAGS_RELEASE}")

set(CMAKE_CXX_FLAGS "-std=c++0x")

find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)

###########
## Build ##
###########

include_directories(include)
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# mavrosflight library
add_library(vi_ekf
  src/vi_ekf.cpp
  include/vi_ekf.h
  src/math_helper.cpp
  include/math_helper.h
)
