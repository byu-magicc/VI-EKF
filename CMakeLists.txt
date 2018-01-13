cmake_minimum_required(VERSION 2.8.3)
project(vi_ekf)

set(CMAKE_CXX_FLAGS "-std=c++17")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  image_transport
  cv_bridge
)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp sensor_msgs image_transport cv_bridge
)

###########
## Build ##
###########

include_directories(include)
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_library(vi_ekf
  src/vi_ekf.cpp
  include/vi_ekf.h
  include/math_helper.h
  include/quat.h
)

add_library(klt_tracker
  src/klt_tracker.cpp
  include/klt_tracker.h
)

add_executable(test
    src/test.cpp
)
target_link_libraries(test ${GTEST_LIBRARIES} pthread vi_ekf)

add_executable(vi_ekf_node
  src/vi_ekf_ros.cpp
  include/vi_ekf_ros.h
)
target_link_libraries(vi_ekf_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  vi_ekf
  klt_tracker
)
