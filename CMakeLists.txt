cmake_minimum_required(VERSION 2.8.3)
project(vi_ekf)

set(CMAKE_CXX_FLAGS "-std=c++17 -mavx -mfma")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  image_transport
  cv_bridge
  geometry_msgs
  nav_msgs
  rosbag
)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp sensor_msgs image_transport cv_bridge geometry_msgs nav_msgs
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
  src/vi_ekf/vi_ekf.cpp
  src/vi_ekf/vi_ekf_helper.cpp
  src/vi_ekf/vi_ekf_feat.cpp
  src/vi_ekf/vi_ekf_log.cpp
  src/vi_ekf/vi_ekf_meas.cpp
  src/vi_ekf/vi_ekf_error.cpp
  src/vi_ekf/vi_ekf_kfr.cpp
  src/vi_ekf/vi_ekf_dyn.cpp
  include/vi_ekf.h
  src/math_helper.cpp
  include/math_helper.h
  include/quat.h
)

add_executable(vi_ekf_test
    src/test.cpp
)
target_link_libraries(vi_ekf_test ${GTEST_LIBRARIES} ${OpenCV_LIBS} pthread vi_ekf)

add_library(vi_ekf_ros
  src/vi_ekf_ros.cpp
  include/vi_ekf_ros.h
  include/eigen_helpers.h
)
target_link_libraries(vi_ekf_ros
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  vi_ekf
)

add_executable(vi_ekf_node
  src/vi_ekf_node.cpp
  )
target_link_libraries(vi_ekf_node
  ${catkin_LIBRARIES}
  vi_ekf_ros  
)

add_executable(vi_ekf_rosbag
  src/rosbag_parser.cpp
  )
target_link_libraries(vi_ekf_rosbag
  ${catkin_LIBRARIES}
  vi_ekf_ros  
)
