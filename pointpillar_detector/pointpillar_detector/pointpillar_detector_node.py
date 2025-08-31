import rclpy
from rclpy.node import Node
from pointpillar_detector.pointpillar_detector import PointPillarDetector
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray

class PointPillarDetectorNode(Node):
  def __init__(self):
    super().__init__('point_pillar_detector_node')
    self.declare_parameter("encoder_model_path", "models/pts_voxel_encoder_centerpoint.onnx")
    self.declare_parameter("remaining_model_path", "models/pts_backbone_neck_head_centerpoint.onnx")
    self.declare_parameter("voxel_size", [0.32, 0.32, 100.0])
    self.declare_parameter("max_voxels", 40000)
    self.declare_parameter("max_points_in_pillar", 32)
    self.declare_parameter("ranges", [-76.8, -76.8, -4.0, 76.8, 76.8, 6.0])
    self.declare_parameter("class_names", ["CAR", "TRUCK", "BUS", "BICYCLE", "PEDESTRIAN"])
    self.declare_parameter("iou_threshold", 0.5)
    self.declare_parameter("score_threshold", 0.3)
    
    parameters = {
      "encoder_model_path": self.get_parameter("encoder_model_path").get_parameter_value().string_value,
      "remaining_model_path": self.get_parameter("remaining_model_path").get_parameter_value().string_value,
      "voxel_size": self.get_parameter("voxel_size").get_parameter_value().double_array_value,
      "max_voxels": self.get_parameter("max_voxels").get_parameter_value().integer_value,
      "max_points_in_pillar": self.get_parameter("max_points_in_pillar").get_parameter_value().integer_value,
      "ranges": self.get_parameter("ranges").get_parameter_value().double_array_value,
      "class_names": self.get_parameter("class_names").get_parameter_value().string_array_value,
      "iou_threshold": self.get_parameter("iou_threshold").get_parameter_value().double_value,
      "score_threshold": self.get_parameter("score_threshold").get_parameter_value().double_value,
    }
    
    self.detector = PointPillarDetector(parameters, self.get_logger())
    
    self.subscription = self.create_subscription(
      PointCloud2,
      '/kitti_pcd',
      self.listener_callback,
      10)
    self.publisher_ = self.create_publisher(PointCloud2, '/pillars', 10)
    self.marker_pub = self.create_publisher(MarkerArray, '/detections', 10)
    self.once = True
    self.marker_array_msg = MarkerArray()
  
  def listener_callback(self, msg):
    if self.once:
      self.get_logger().info('Received point cloud with %d points' % msg.width)
      self.marker_array_msg = self.detector.detect(msg)
      self.once = False
    
    # Publish the markers
    self.marker_pub.publish(self.marker_array_msg)
  

def main(args=None):
  rclpy.init(args=args)
  rclpy.spin(PointPillarDetectorNode())
  rclpy.shutdown()
