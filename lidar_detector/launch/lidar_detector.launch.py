from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
  pkg_share_dir = get_package_share_directory('lidar_detector')

  params_file_path = os.path.join(pkg_share_dir, 'config', 'lidar_params.yaml')

  lidar_detector_node = Node(
    package="lidar_detector",
    executable="lidar_detector_node",
    name="lidar_detector_node",
    output="screen",
    parameters=[
      params_file_path
    ]
  )

  return LaunchDescription([
    lidar_detector_node
  ])
