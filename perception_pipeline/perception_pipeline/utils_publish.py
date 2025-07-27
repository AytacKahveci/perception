#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import numpy as np

FRAME_ID = "map"

def publish_frame(frame_pub, frame_id):
  frame_msg = std_msgs.String()
  frame_msg.data = "%010d" % frame_id
  frame_pub.publish(frame_msg)


def publish_img(cam_pub, bridge, cvimg):
  imgmsg = bridge.cv2_to_imgmsg(cvimg, 'bgr8')
  imgmsg.header.frame_id = FRAME_ID
  cam_pub.publish(imgmsg)


def publish_pcd(pcd_pub, pcd):
  def create_pcdmsg(pcd, parent_frame):
    """ Creates a point cloud message.
    Args:
        pcd: pcd data of point cloud
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    References:
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
        http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html
    """
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    points = np.asarray(pcd.points)  # points: Nx3 array of xyz positions.
    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
      name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyz')]

    header = std_msgs.Header(frame_id=parent_frame)
    pcd_msg = sensor_msgs.PointCloud2(
      header=header,
      height=1,
      width=points.shape[0],
      is_dense=False,
      is_bigendian=False,
      fields=fields,
      point_step=(itemsize*3),
      row_step=(itemsize*3*points.shape[0]),
      data=data)
    return pcd_msg

  pcd_msg = create_pcdmsg(pcd, FRAME_ID)
  pcd_pub.publish(pcd_msg)


def publish_pcl(pcl_pub, point_cloud, frame_id=FRAME_ID):
  def create_pclmsg(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    References:
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
        http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html
    """
    # In a PointCloud2 message, the point cloud is stored as an byte
    # array. In order to unpack it, we also include some parameters
    # which desribes the size of each individual point.

    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes()

    # The fields specify what the bytes represents. The first 4 bytes
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [sensor_msgs.PointField(
      name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
      for i, n in enumerate('xyz')]

    # The PointCloud2 message also has a header which specifies which
    # coordinate frame it is represented in.
    header = std_msgs.Header(frame_id=parent_frame)

    return sensor_msgs.PointCloud2(
      header=header,
      height=1,
      width=points.shape[0],
      is_dense=False,
      is_bigendian=False,
      fields=fields,
      # Every point consists of three float32s.
      point_step=(itemsize * 3),
      row_step=(itemsize * 3 * points.shape[0]),
      data=data
    )

  pcd = create_pclmsg(point_cloud[:, :3], frame_id)
  pcl_pub.publish(pcd)
