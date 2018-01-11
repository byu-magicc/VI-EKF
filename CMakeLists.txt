cmake_minimum_required(VERSION 2.8.3)
project(vi_ekf)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

###########
## Build ##
###########

include_directories(include)
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
)

# mavrosflight library
add_library(vi_ekf
  src/vi_ekf.cpp
  include/vi_ekf.h
  include/math_helper.h
  include/quat.h
)

add_executable(test
    src/test.cpp
)
target_link_libraries(test ${GTEST_LIBRARIES} pthread vi_ekf)
