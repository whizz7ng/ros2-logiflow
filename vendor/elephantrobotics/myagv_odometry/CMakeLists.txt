cmake_minimum_required(VERSION 3.8)
project(myagv_odometry)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Boost REQUIRED)
find_package(nav_msgs REQUIRED)

include_directories(
  include
)

add_executable(myagv_odometry_node
               src/myAGV.cpp
               src/myAGVSub.cpp)
ament_target_dependencies(myagv_odometry_node rclcpp tf2 tf2_ros sensor_msgs geometry_msgs nav_msgs)

install(TARGETS
  myagv_odometry_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  src launch config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
