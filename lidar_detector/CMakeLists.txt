cmake_minimum_required(VERSION 3.8)
project(lidar_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(detection_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(lidar_detector_node
  src/voxel_grid_based_detection.cpp
)
ament_target_dependencies(lidar_detector_node
  detection_msgs
  geometry_msgs
  pcl_conversions
  rclcpp
  sensor_msgs
  std_msgs
  visualization_msgs
)
target_link_libraries(lidar_detector_node
  ${PCL_LIBRARIES} 
)

install(
  TARGETS lidar_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
