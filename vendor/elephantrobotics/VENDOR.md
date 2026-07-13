# Elephant Robotics vendor packages

These packages are based on the Elephant Robotics myAGV ROS repository
and include project-specific modifications.

## Upstream

- Repository: https://github.com/elephantrobotics/myagv_ros
- Branch: myagv_ros_2023Pi
- Commit: unknown
- Imported date: 2026-07-13

## Imported packages

- myagv_odometry
- myagv_navigation2

## Local modifications

### myagv_odometry

- myAGV serial communication and odometry handling
- command and UART diagnostic logging
- odometry and EKF launch configuration
- project-specific launch configuration

### myagv_navigation2

- Nav2 controller and velocity parameters
- mecanum/omnidirectional driving configuration
- active navigation launch configuration
- safety-filter-compatible command topic configuration

The original license is preserved in this directory.
