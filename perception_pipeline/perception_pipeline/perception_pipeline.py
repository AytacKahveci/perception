#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import cv2
import numpy as np
import open3d as o3d

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from perception_pipeline.utils_kitti import Calibration


class Pipeline(Node):
  def __init__(self, name):
    super().__init__(name)

    qos_profile = QoSProfile(
      reliability=QoSReliabilityPolicy.RELIABLE,
      history=QoSHistoryPolicy.KEEP_LAST,
      depth=10
    )

    self.image_sub = self.create_subscription(Image, "/kitti_imgraw_L/RGB", self.image_callback, 10)
    self.pcd_sub = self.create_subscription(PointCloud2, "/kitti_pcd", self.pointcloud_callback, 10)
    self.image_pub = self.create_publisher(Image, '/img', 10)
    self.marker_pub = self.create_publisher(MarkerArray, '/pcd_markers', 10)
    self.pcd_pub = self.create_publisher(PointCloud2, '/pcd_cloud', 10)
    self.image = None
    self.image_received = False

    self.cv_bridge = CvBridge()
    self.calibration = Calibration("/media/aytac/Tank/ros_ws/src/perception_pipeline/rawdata/2011_09_26", True)

  def image_callback(self, image: Image):
    self.image = self.cv_bridge.imgmsg_to_cv2(image)
    self.image_received = True

  def pointcloud_callback(self, msg: PointCloud2):
    points_list = [[data[0], data[1], data[2]] for data in list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))]
    if not points_list:
      return

    points = np.array(points_list, dtype=np.float32)

    # Open3D PointCloud object
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    # Voxel grid filter
    cloud = cloud.voxel_down_sample(voxel_size=0.05)

    # ROI filter just for simplicity
    points_filtered = np.asarray(cloud.points)
    mask = (points_filtered[:, 0] > 0.0) & (points_filtered[:, 0] < 10.0) & \
           (points_filtered[:, 1] > -3.0) & (points_filtered[:, 1] < 3.0) & \
           (points_filtered[:, 2] > -1.5) & (points_filtered[:, 2] < 1.0)
    cloud.points = o3d.utility.Vector3dVector(points_filtered[mask])

    # Euclidean clustering
    labels = np.array(cloud.cluster_dbscan(eps=0.5, min_points=25, print_progress=False))

    marker_array = MarkerArray()
    points_clustered = np.asarray(cloud.points)
    max_label = labels.max()
    points = np.empty((0, 3), dtype=np.float32)

    for i in range(max_label + 1):
      cluster_points = points_clustered[labels == i]
      if cluster_points.size == 0:
        continue
      points = np.vstack([points, cluster_points])

      min_pt = cluster_points.min(axis=0)
      max_pt = cluster_points.max(axis=0)

      if self.image_received:
        min_point = self.calibration.project_velo_to_image(np.array([min_pt]))[0]
        max_point = self.calibration.project_velo_to_image(np.array([max_pt]))[0]

        cv2.rectangle(self.image, (int(min_point[0]), int(min_point[1])), (int(max_point[0]), int(max_point[1])), (0, 0, 255), 3)
        image_msg = self.cv_bridge.cv2_to_imgmsg(self.image, encoding="bgr8")
        self.image_pub.publish(image_msg)

      marker = Marker()
      marker.header.frame_id = msg.header.frame_id
      marker.header.stamp = self.get_clock().now().to_msg()
      marker.ns = "objects"
      marker.id = i
      marker.type = Marker.CUBE
      marker.action = Marker.ADD
      marker.pose.position.x = float((min_pt[0] + max_pt[0]) / 2)
      marker.pose.position.y = float((min_pt[1] + max_pt[1]) / 2)
      marker.pose.position.z = float((min_pt[2] + max_pt[2]) / 2)
      marker.pose.orientation.w = 1.0
      marker.scale.x = float(max_pt[0] - min_pt[0])
      marker.scale.y = float(max_pt[1] - min_pt[1])
      marker.scale.z = float(max_pt[2] - min_pt[2])
      marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)
      marker.lifetime.sec = 0
      marker_array.markers.append(marker)

    header = Header()
    header.stamp = self.get_clock().now().to_msg()
    header.frame_id = msg.header.frame_id

    fields = [
      PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
      PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
      PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]

    cloudpoint_processed_msg = point_cloud2.create_cloud(header, fields, points)
    self.pcd_pub.publish(cloudpoint_processed_msg)

    self.marker_pub.publish(marker_array)


def main():
  rclpy.init()
  pipeline = Pipeline("pipeline_node")
  rclpy.spin(pipeline)
  pipeline.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
