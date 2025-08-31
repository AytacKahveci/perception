import matplotlib.pyplot as plt
import numpy as np
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations
from visualization_msgs.msg import Marker, MarkerArray
import onnxruntime as ort

class PointPillarDetector:
  def __init__(self, parameters : dict, logger):
    self.encoder_model_path = parameters["encoder_model_path"]
    self.remaining_model_path = parameters["remaining_model_path"]
    self.voxel_size = np.asarray(parameters["voxel_size"], dtype=np.float32)
    self.max_voxels = parameters["max_voxels"]
    self.max_points_in_pillar = parameters["max_points_in_pillar"]
    self.ranges = np.asarray(parameters["ranges"], dtype=np.float32)
    self.class_names = parameters["class_names"]
    self.iou_threshold = parameters["iou_threshold"]
    self.score_threshold = parameters["score_threshold"]
    self.num_features = 9
    self.grid = np.round((self.ranges[3:] - self.ranges[0:3]) / self.voxel_size).astype(np.int32)
    self.header = Header()
    self.logger = logger
    self.encoder_session = self.create_onnx_session(self.encoder_model_path)
    self.remaining_session = self.create_onnx_session(self.remaining_model_path)

  def detect(self, pcd: PointCloud2):
    pillars, pilliar_coords = self.create_pillars(pcd)
    encoder_output = self.infer_encoder(self.encoder_session, pillars)
    self.logger.info(f"Encoder output shape: {encoder_output[0].shape}")
    
    spatial_features = self.create_spatial_features(encoder_output[0], pilliar_coords)
    output = self.infer_remaining(self.remaining_session, spatial_features)
    self.logger.info(f"Remaining model output shape: {output[0].shape}")

    detections = self.get_detections(output, self.score_threshold)
    self.logger.info(f"Detected objects: {len(detections)}")
    filtered_detections = self.filter_nms(detections, self.iou_threshold)
    self.logger.info(f"Filtered object: {len(filtered_detections)}")

    return self.create_cube_markers_from_detections(filtered_detections, frame_id="map")

  def create_pillars(self, pcd: PointCloud2):
    ''' Create pillars

    Args:
      pcd: pointcloud

    Returns:
      pillar_features: np.array(max_voxels, max_points_in_pillar, num_features)
      pillar_coords = np.array(max_voxels, 2)
    '''
    self.header = pcd.header
    points = pc2.read_points_numpy(pcd, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    print("Point cloud shape:", points.shape)
    point_pillars = {}
    for point in points:
      x_ind = np.floor((point[0] - self.ranges[0]) / self.voxel_size[0]).astype(np.int32)
      y_ind = np.floor((point[1] - self.ranges[1]) / self.voxel_size[1]).astype(np.int32)
      

      if 0 <= x_ind < self.grid[0] and 0 <= y_ind < self.grid[1]:
        pillar_key = (x_ind, y_ind)
        if pillar_key not in point_pillars:
          point_pillars[pillar_key] = []
        
        point_pillars[pillar_key].append(point)
        
    pillar_features = np.zeros((self.max_voxels, self.max_points_in_pillar, self.num_features), dtype=np.float32)
    pillar_coords = np.zeros((self.max_voxels, 2), dtype=np.int32)
    num_points_in_pillar = np.zeros(self.max_voxels, dtype=np.int32)

    pillar_count = 0
    for pillar_key, points in point_pillars.items():
      if pillar_count >= self.max_voxels:
        break

      points_in_pillar = np.asarray(points, dtype=np.float32)
      num_points = points_in_pillar.shape[0]
      if num_points > self.max_points_in_pillar:
        # Randomly sample
        indexes = np.random.choice(num_points, self.max_points_in_pillar, replace=False)  
        points_in_pillar = points_in_pillar[indexes]
        num_points = self.max_points_in_pillar
      
      x, y, z, r = points_in_pillar[:, 0], points_in_pillar[:, 1], points_in_pillar[:, 2], points_in_pillar[:, 3]
      x_mean = np.mean(x)
      y_mean = np.mean(y)
      z_mean = np.mean(z)
      
      x_c = x - x_mean
      y_c = y - y_mean
      z_c = z - z_mean

      x_p = x - (pillar_key[0] * self.voxel_size[0] + self.ranges[0] + self.voxel_size[0] / 2)
      y_p = y - (pillar_key[1] * self.voxel_size[1] + self.ranges[1] + self.voxel_size[1] / 2)

      features = np.stack([x, y, z, r, x_c, y_c, z_c, x_p, y_p], axis=1)
      pillar_features[pillar_count, :num_points, :] = features
      pillar_coords[pillar_count, :] = np.asarray([pillar_key[0], pillar_key[1]], dtype=np.int32)
      num_points_in_pillar[pillar_count] = num_points

      pillar_count += 1
    return pillar_features, pillar_coords
  
  def get_pillars_cloud(self, pillars: np.array):
    ''' Get pillars cloud for visualization

    Args:
      pillars: np.array(max_voxels, max_points_in_pillar, num_features)
    '''
    fields = [
      PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
      PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
      PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
      PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    cmap = plt.get_cmap('tab20')
    all_points = []
    
    for pillar_count in range(pillars.shape[0]):
      current_pillar_points = pillars[pillar_count, :, :]

      if not np.all(current_pillar_points == 0):
        color = cmap(pillar_count % 20)
        
        intensity_value = color[0] + color[1] + color[2]
        
        intensities = np.full((current_pillar_points.shape[0], 1), intensity_value)

        points_with_intensity = np.hstack([current_pillar_points[:, :3], intensities])

        all_points.append(points_with_intensity)

    if not all_points:
      return None

    # Concatenate all points from all pillars
    combined_points = np.concatenate(all_points, axis=0)
    
    pcd = pc2.create_cloud(self.header, fields, combined_points)
    
    return pcd

  def create_spatial_features(self, encoder_output: np.array, pillar_coords: np.array):
    ''' Create spatial features from encoder output
    
    Args:
      encoder_output: np.array(max_voxels, 1, max_points_in_pillar)
      pillar_coords: np.array(max_voxels, 2)
    '''
    W = self.grid[0]
    H = self.grid[1]
    spatial_features = np.zeros((1, 32, H, W), dtype=np.float32)
    for i in range(pillar_coords.shape[0]):
      x, y = pillar_coords[i]
      # TODO: Check x, y => y, x
      spatial_features[0, :, x, y] = encoder_output[i, 0, :]
    return spatial_features
  
  def create_onnx_session(self, model_path: str):
    ''' Create ONNX session

    Args:
      model_path: path to the ONNX model

    Returns:
      ONNX Runtime session
    '''
    try:
      session = ort.InferenceSession(model_path)
      print("ONNX session created successfully.")
      print("Input names:", [input.name for input in session.get_inputs()])
      return session
    except Exception as e:
      print("Error creating ONNX session:", e)
      return None
    
  def infer_encoder(self, session : ort.InferenceSession, pillars: np.array):
    ''' Run inference on the encoder model
    
    Args:
      session: ONNX Runtime session
      pillars: np.array(max_voxels, max_points_in_pillar, num_features) '''

    input_name = session.get_inputs()[0].name
    encoder_output = session.run(None, {input_name: pillars})
    return encoder_output
  
  def infer_remaining(self, session: ort.InferenceSession, spatial_features: np.array):
    ''' Run inference on the remaining model
    
    Args:
      session: ONNX Runtime session
      spatial_features: np.array(1, C, H, W) 

    Returns:
      model output
    '''
    
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: spatial_features})
    return output

  @staticmethod
  def calc_sigma(val):
    ''' Calculate sigmoid function

    Args:
      val: value

    Returns:
      output
    '''
    return 1 / (1 + np.exp(-val))

  @staticmethod
  def calc_iou(detection1 : np.array, detection2 : np.array):
    ''' Calculate intersection of uniuon

    Args:
      detection1: np.array(x, y, z, w, l, h)
      detection2: np.array(x, y, z, w, l, h)

    Return:
      iou: float32
    '''
    xl = max(detection1[0] - detection1[3] / 2.0, detection2[0] - detection2[3] / 2.0)
    yl = max(detection1[1] - detection1[4] / 2.0, detection2[1] - detection2[4] / 2.0)
    zl = max(detection1[2] - detection1[5] / 2.0, detection2[2] - detection2[5] / 2.0)

    xr = min(detection1[0] + detection1[3] / 2.0, detection2[0] + detection2[3] / 2.0)
    yr = min(detection1[1] + detection1[4] / 2.0, detection2[1] + detection2[4] / 2.0)
    zr = min(detection1[2] + detection1[5] / 2.0, detection2[2] + detection2[5] / 2.0)

    intersection = max(0, xr - xl) * max(0, yr - yl) * max(0, zr - zl)

    vol1 = detection1[3] * detection1[4] * detection1[5]
    vol2 = detection2[3] * detection2[4] * detection2[5]
    union = vol1 + vol2 - intersection

    iou = intersection / union if union > 0 else 0
    return iou

  def filter_nms(self, detections: list, iou_threshold = 0.5):
    ''' Filter detections according to non maximum suppression

    Args:
      detections: list(dict())
      iou_threshold: Threshold for checking similarity

    Returns:
      filtered_detections: list(dict())
    '''
    if not detections:
      return []
    
    detections.sort(key=lambda det : det["score"], reverse=True)

    keep = []
    while len(detections):
      max_detection = detections.pop(0)
      keep.append(max_detection)

      detections_cpy = detections[:]
      for detection in detections_cpy:
        iou = self.calc_iou(max_detection["bbox"], detection["bbox"])
        if iou > iou_threshold:
          detections.remove(detection)
    return keep

  def get_detections(self, output: list, score_threshold = 0.3):
    ''' Get detections from the inference

    Args:
      output: list

    Return:
      detections: list(dict())
    '''
    heatmap = self.calc_sigma(output[0])
    reg = output[1]
    height = output[2]
    dim = output[3]
    rot = output[4]
    vel = output[5]

    detections = []
    for class_id, class_name in enumerate(self.class_names):
      heatmap_tmp = heatmap[0, class_id, :, :]

      peak_points = np.argwhere(heatmap_tmp > score_threshold)
      for point_x, point_y in peak_points:
        # TODO: check point_x, point_y => point_y, point_x
        score = heatmap_tmp[point_x, point_y]

        reg_val = reg[0, :, point_x, point_y]
        height_val = height[0, 0, point_x, point_y]
        dim_val = dim[0, :, point_x, point_y]
        rot_val = rot[0, :, point_x, point_y]
        vel_val = vel[0, :, point_x, point_y]

        center_x = (point_x + reg_val[0]) * self.voxel_size[0] + self.ranges[0]
        center_y = (point_y + reg_val[1]) * self.voxel_size[1] + self.ranges[1]
        center_z = height_val

        w, l, h = dim_val[0], dim_val[1], dim_val[2]

        yaw = np.arctan2(rot_val[1], rot_val[0])

        vel_x, vel_y = vel_val[0], vel_val[1]
        detections.append({
          "class_name": class_name,
          "score": score,
          "bbox": [center_x, center_y, center_z, w, l, h],
          "yaw": yaw,
          "velocity": [vel_x, vel_y]
        })
    return detections

  def create_cube_markers_from_detections(self, detections, frame_id, namespace="detections"):
    '''
    Creates a visualization_msgs/MarkerArray from a list of detection dictionaries.

    Args:
      detections: A list of dictionaries, where each dictionary represents a detection
                  with keys "class_name", "score", "bbox", "yaw", and "velocity".
      frame_id: The ROS frame ID for the markers (e.g., "base_link").
      namespace: The namespace for the markers to avoid conflicts.

    Returns:
      A visualization_msgs/MarkerArray containing a Marker for each detection.
    '''
    marker_array = MarkerArray()
    
    for i, det in enumerate(detections):
      # Create a new Marker message
      marker = Marker()
      
      # Set the header
      marker.header.frame_id = frame_id

      # Set the marker's namespace and ID. The ID must be unique.
      marker.ns = namespace
      marker.id = i
      
      # Set the marker type to a cube
      marker.type = Marker.CUBE
      
      # Set the marker action to ADD
      marker.action = Marker.ADD

      # Get the bounding box and orientation data
      center_x, center_y, center_z, w, l, h = det["bbox"]
      yaw = det["yaw"]

      # Set the position
      marker.pose.position.x = float(center_x)
      marker.pose.position.y = float(center_y)
      marker.pose.position.z = float(center_z)
      
      # Convert yaw angle (z-axis rotation) to a quaternion
      quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
      marker.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

      # Set the scale of the cube (width, length, height)
      marker.scale.x = float(l)
      marker.scale.y = float(w)
      marker.scale.z = float(h)

      # Set the color and transparency (RGBA)
      # Here we'll use a simple green color for visualization.
      # You can customize this based on class_name or score.
      if det["class_name"] in ["CAR", "TRUCK", "BUS"]:
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
      elif det["class_name"] == "PEDESTRIAN":
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
      elif det["class_name"] == "BICYCLE":
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
      marker.color.a = 0.5  # 50% transparency


      # Set the marker lifetime to a short duration to prevent old markers from persisting
      marker.lifetime.sec = 0
      marker.lifetime.nanosec = 100000000  # 0.1 seconds

      # Append the new marker to the array
      marker_array.markers.append(marker)

    return marker_array
  
