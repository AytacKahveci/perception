from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
  pkg_perception = get_package_share_directory("perception_pipeline")
  default_rviz_config_path = os.path.join(pkg_perception, 'rviz', 'config.rviz')

  kitti_publisher_node = Node(
    package="perception_pipeline",
    executable="kitti_publisher_node"
  )

  rviz_node = Node(
    package='rviz2',
    executable='rviz2',
    name='rviz2',
    arguments=['-d', default_rviz_config_path]
  )

  return LaunchDescription([
    DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path, description="Rviz config"),
    kitti_publisher_node,
    rviz_node
  ])
