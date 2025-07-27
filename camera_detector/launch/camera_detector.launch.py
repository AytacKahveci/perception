from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
  pkg_share_dir = get_package_share_directory('camera_detector')

  default_model_cfg_path = os.path.join(pkg_share_dir, 'models', 'yolov3.cfg')
  default_model_weights_path = os.path.join(pkg_share_dir, 'models', 'yolov3.weights')
  default_class_names_path = os.path.join(pkg_share_dir, 'models', 'coco.names')

  model_cfg_path_arg = DeclareLaunchArgument(
    'model_cfg_path',
    default_value=default_model_cfg_path,
    description='Path to the YOLOv3 model configuration (.cfg) file.'
  )
  model_weights_path_arg = DeclareLaunchArgument(
    'model_weights_path',
    default_value=default_model_weights_path,
    description='Path to the YOLOv3 model weights (.weights) file.'
  )
  class_names_path_arg = DeclareLaunchArgument(
    'class_names_path',
    default_value=default_class_names_path,
    description='Path to the class names (.names) file.'
  )
  confidence_threshold_arg = DeclareLaunchArgument(
    'confidence_threshold',
    default_value=TextSubstitution(text='0.5'),
    description='Confidence threshold for object detection.'
  )
  nms_threshold_arg = DeclareLaunchArgument(
    'nms_threshold',
    default_value=TextSubstitution(text='0.4'),
    description='Non-Maximum Suppression (NMS) threshold.'
  )

  camera_detector_node = Node(
    package="camera_detector",
    executable="camera_detector_node",
    name="camera_detector_node",
    output="screen",
    parameters=[
      {'model_cfg_path': LaunchConfiguration('model_cfg_path')},
      {'model_weights_path': LaunchConfiguration('model_weights_path')},
      {'class_names_path': LaunchConfiguration('class_names_path')},
      {'confidence_threshold': LaunchConfiguration('confidence_threshold')},
      {'nms_threshold': LaunchConfiguration('nms_threshold')},
    ]
  )

  return LaunchDescription([
    model_cfg_path_arg,
    model_weights_path_arg,
    class_names_path_arg,
    confidence_threshold_arg,
    nms_threshold_arg,
    camera_detector_node
  ])