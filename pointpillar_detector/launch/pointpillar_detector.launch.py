from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
  pkg_share_dir = get_package_share_directory('pointpillar_detector')
  
  params_file_path = os.path.join(pkg_share_dir, 'config', 'params.yaml')

  encoder_model_path = os.path.join(
      pkg_share_dir,
      'models',
      'pts_voxel_encoder_centerpoint.onnx'
  )
  remaining_model_path = os.path.join(
      pkg_share_dir,
      'models',
      'pts_backbone_neck_head_centerpoint.onnx'
  )

  pointpillar_detector_node = Node(
    package="pointpillar_detector",
    executable="pointpillar_detector_node",
    name="pointpillar_detector_node",
    output="screen",
    parameters=[
      params_file_path,
      {
      "encoder_model_path": encoder_model_path,
      "remaining_model_path": remaining_model_path
      }
    ]
  )
  
  return LaunchDescription([
    pointpillar_detector_node
  ])
