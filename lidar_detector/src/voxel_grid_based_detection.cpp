#include <algorithm>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#include "detection_msgs/msg/detection3_d.hpp"
#include "detection_msgs/msg/detection3_d_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/header.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"

#include <pcl/common/common.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

geometry_msgs::msg::Point getCentroid(const sensor_msgs::msg::PointCloud2 & pointcloud)
{
  geometry_msgs::msg::Point centroid;
  centroid.x = 0.0f;
  centroid.y = 0.0f;
  centroid.z = 0.0f;
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(pointcloud, "x"),
      iter_y(pointcloud, "y"), iter_z(pointcloud, "z");
      iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    centroid.x += *iter_x;
    centroid.y += *iter_y;
    centroid.z += *iter_z;
  }
  const size_t size = pointcloud.width * pointcloud.height;
  centroid.x = centroid.x / static_cast<float>(size);
  centroid.y = centroid.y / static_cast<float>(size);
  centroid.z = centroid.z / static_cast<float>(size);
  return centroid;
}

class LidarDetectorNode : public rclcpp::Node
{
public:
  LidarDetectorNode() : Node("lidar_detector_node")
  {
    min_x_ = this->declare_parameter("min_x", 0.0);
    min_y_ = this->declare_parameter("min_y", -10.0);
    min_z_ = this->declare_parameter("min_z", -2.0);
    max_x_ = this->declare_parameter("max_x", 20.0);
    max_y_ = this->declare_parameter("max_y", 10.0);
    max_z_ = this->declare_parameter("max_z", 1.0);
    ground_plane_distance_threshold_ = this->declare_parameter("ground_plane_distance_threshold", 0.2);
    cluster_tolerance_ = this->declare_parameter("cluster_tolerance", 0.5);
    min_cluster_size_ = this->declare_parameter("min_cluster_size", 10);
    max_cluster_size_ = this->declare_parameter("max_cluster_size", 25000);
    voxel_leaf_size_ = this->declare_parameter("voxel_leaf_size", 0.5f);
    min_points_per_voxel_ = this->declare_parameter("min_points_per_voxel", 3);
    min_voxel_cluster_size_for_filtering_ = this->declare_parameter("min_voxel_cluster_size_for_filtering", 150);
    max_voxel_cluster_for_output_ = this->declare_parameter("max_voxel_cluster_for_output", 800);
    max_points_per_voxel_in_large_cluster_ = this->declare_parameter("max_points_per_voxel_in_large_cluster", 10);
  
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/kitti_pcd", 10, std::bind(&LidarDetectorNode::pointCloudCallback, this, std::placeholders::_1));

    detection_pub_ = this->create_publisher<detection_msgs::msg::Detection3DArray>("/lidar/detections", 10);
    detection_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/lidar/detection_markers", 10);
    filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/filtered_point_cloud", 10);
    ground_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/ground_point_cloud", 10);
    non_ground_pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/non_ground_point_cloud", 10);

    callback_handle_ = this->add_on_set_parameters_callback(
        std::bind(&LidarDetectorNode::onParameterEventCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "LidarDetectorNode initialized. Subscribing to /lidar/point_cloud, publishing to /lidar/detections");
  }

  void cluster(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_msg,
               detection_msgs::msg::Detection3DArray& detection_array_msg)
  {
    int point_step = pointcloud_msg->point_step;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*pointcloud_msg, *pointcloud);

    // ROI Filter
    /* pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_roi(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PassThrough<pcl::PointXYZ> passthrough_filter;
    passthrough_filter.setInputCloud(pointcloud);
    passthrough_filter.setFilterFieldName("x");
    passthrough_filter.setFilterLimits(min_x_, max_x_);
    passthrough_filter.filter(*cloud_filtered_roi);

    passthrough_filter.setInputCloud(cloud_filtered_roi);
    passthrough_filter.setFilterFieldName("y");
    passthrough_filter.setFilterLimits(min_y_, max_y_);
    passthrough_filter.filter(*cloud_filtered_roi);

    passthrough_filter.setInputCloud(cloud_filtered_roi);
    passthrough_filter.setFilterFieldName("z");
    passthrough_filter.setFilterLimits(min_z_, max_z_);
    passthrough_filter.filter(*cloud_filtered_roi);

    sensor_msgs::msg::PointCloud2 filtered_roi_output;
    pcl::toROSMsg(*cloud_filtered_roi, filtered_roi_output);
    filtered_roi_output.header = pointcloud_msg->header;
    filtered_pub_->publish(filtered_roi_output);
    */
    // Ground plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(ground_plane_distance_threshold_);
    seg.setInputCloud(pointcloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) 
    {
      RCLCPP_WARN(this->get_logger(), "Could not estimate a planar model for the given dataset.");
      return;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>());
    extract.setInputCloud(pointcloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_ground);

    sensor_msgs::msg::PointCloud2 ground_output;
    pcl::toROSMsg(*cloud_ground, ground_output);
    ground_output.header = pointcloud_msg->header;
    ground_pc_pub_->publish(ground_output);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_non_ground(new pcl::PointCloud<pcl::PointXYZ>());
    extract.setNegative(true);
    extract.filter(*cloud_non_ground);

    sensor_msgs::msg::PointCloud2 non_ground_output;
    pcl::toROSMsg(*cloud_non_ground, non_ground_output);
    non_ground_output.header = pointcloud_msg->header;
    non_ground_pc_pub_->publish(non_ground_output);


    // Voxel grid filtering
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_map(new pcl::PointCloud<pcl::PointXYZ>);
    const float z_axis_voxel_size = 100000.0f;
    
    voxel_grid_.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, z_axis_voxel_size);
    voxel_grid_.setMinimumPointsNumberPerVoxel(min_points_per_voxel_);
    voxel_grid_.setInputCloud(cloud_non_ground);
    voxel_grid_.setSaveLeafLayout(true);
    voxel_grid_.filter(*voxel_map);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_2d(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : voxel_map->points)
    {
      pcl::PointXYZ point2d;
      point2d.x = point.x;
      point2d.y = point.y;
      point2d.z = 0.0;
      pointcloud_2d->points.push_back(point2d);
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(pointcloud_2d);

    // Euclidean clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(1);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(pointcloud_2d);
    ec.extract(cluster_indices);

    std::unordered_map</* voxel grid index */ int, /* cluster index */ int> voxel_to_cluster_map;
    std::vector<sensor_msgs::msg::PointCloud2> temporary_clusters;  // no check about cluster size
    std::vector<size_t> clusters_data_size;
    temporary_clusters.resize(cluster_indices.size());
    for (size_t cluster_idx = 0; cluster_idx < cluster_indices.size(); ++cluster_idx) 
    {
      const auto & cluster = cluster_indices.at(cluster_idx);
      auto & temporary_cluster = temporary_clusters.at(cluster_idx);
      for (const auto & point_idx : cluster.indices) 
      {
        voxel_to_cluster_map[point_idx] = cluster_idx;
      }
      temporary_cluster.height = pointcloud_msg->height;
      temporary_cluster.fields = pointcloud_msg->fields;
      temporary_cluster.point_step = point_step;
      temporary_cluster.data.resize(cluster.indices.size() * point_step);
      clusters_data_size.push_back(0);
    }

    // Precompute which clusters are large enough based on the voxel threshold.
    // This avoids repeatedly checking the size during per-point processing.
    std::vector<bool> is_large_cluster(cluster_indices.size(), false);
    std::vector<bool> is_extreme_large_cluster(cluster_indices.size(), false);

    for (size_t cluster_idx = 0; cluster_idx < cluster_indices.size(); ++cluster_idx) {
      const int cluster_size = static_cast<int>(cluster_indices[cluster_idx].indices.size());
      is_large_cluster[cluster_idx] = cluster_size > min_voxel_cluster_size_for_filtering_;
      is_extreme_large_cluster[cluster_idx] = cluster_size > max_voxel_cluster_for_output_;
    }

    // Initialize a map to track how many points each voxel has per cluster.
    // Key: cluster index -> (Key: voxel index -> value: point count)
    std::unordered_map<int, std::unordered_map<int, int>> point_counts_per_voxel_per_cluster;
    std::vector<size_t> random_indices(pointcloud->points.size());
    static std::default_random_engine rng(42);
    std::iota(random_indices.begin(), random_indices.end(), 0);
    std::shuffle(random_indices.begin(), random_indices.end(), rng);
    for (size_t i = 0; i < random_indices.size(); ++i) {
      const size_t random_index = random_indices[i];
      const auto & point = pointcloud->points.at(random_index);
      // for (size_t i = 0; i < pointcloud->points.size(); ++i) {
      // const auto & point = pointcloud->points.at(i);
      const int voxel_index =
        voxel_grid_.getCentroidIndexAt(voxel_grid_.getGridCoordinates(point.x, point.y, point.z));
      auto voxel_to_cluster_map_it = voxel_to_cluster_map.find(voxel_index);
      if (voxel_to_cluster_map_it != voxel_to_cluster_map.end()) {
        // Track point count per voxel per cluster
        int cluster_idx = voxel_to_cluster_map_it->second;
        if (is_extreme_large_cluster[cluster_idx]) {
          continue;
        }
        if (is_large_cluster[cluster_idx]) {
          int & voxel_point_count = point_counts_per_voxel_per_cluster[cluster_idx][voxel_index];
          if (voxel_point_count >= max_points_per_voxel_in_large_cluster_) {
            continue;  // Skip adding this point
          }
          voxel_point_count++;
        }

        auto & cluster_data_size = clusters_data_size.at(voxel_to_cluster_map_it->second);
        std::memcpy(
          &temporary_clusters.at(voxel_to_cluster_map_it->second).data[cluster_data_size],
          &pointcloud_msg->data[random_index * point_step], point_step);
        cluster_data_size += point_step;
        if (cluster_data_size == temporary_clusters.at(voxel_to_cluster_map_it->second).data.size()) {
          temporary_clusters.at(voxel_to_cluster_map_it->second)
            .data.resize(temporary_clusters.at(voxel_to_cluster_map_it->second).data.size() * 2);
        }
      }
    }

    {
      detection_array_msg.detections.clear();
      detection_array_msg.header.frame_id = pointcloud_msg->header.frame_id;
      detection_array_msg.header.stamp = this->now();
      
      visualization_msgs::msg::MarkerArray detection_markers;
      int marker_id = 0;

      for (size_t i = 0; i < temporary_clusters.size(); ++i) 
      {
        auto & i_cluster_data_size = clusters_data_size.at(i);
        int cluster_size = static_cast<int>(i_cluster_data_size / point_step);
        if (cluster_size < min_cluster_size_) {
          // Cluster size is below the minimum threshold; skip without messaging.
          continue;
        }
        auto & cluster = temporary_clusters.at(i);
        cluster.width = i_cluster_data_size / point_step / pointcloud_msg->height;
        const auto centroid = getCentroid(cluster);

        detection_msgs::msg::Detection3D detection_msg;
        detection_msg.class_name = "object";
        detection_msg.score = 1.0;

        detection_msg.pose.pose.position.x = centroid.x;
        detection_msg.pose.pose.position.y = centroid.y;
        detection_msg.pose.pose.position.z = centroid.z;
        detection_msg.pose.pose.orientation.x = 0.0;
        detection_msg.pose.pose.orientation.y = 0.0;
        detection_msg.pose.pose.orientation.z = 0.0;
        detection_msg.pose.pose.orientation.w = 1.0;

        detection_msg.dimensions.x = 1;
        detection_msg.dimensions.y = 1;
        detection_msg.dimensions.z = 1;

        //if (detection_msg.dimensions.x > 0.01 && detection_msg.dimensions.y > 0.01 && detection_msg.dimensions.z > 0.01)
        {
          detection_array_msg.detections.push_back(detection_msg);

          visualization_msgs::msg::Marker detection_marker;
          detection_marker.header.frame_id = pointcloud_msg->header.frame_id;
          detection_marker.header.stamp = this->now();
          detection_marker.ns = "object";
          detection_marker.id = marker_id;
          detection_marker.type = visualization_msgs::msg::Marker::CUBE;
          detection_marker.action = visualization_msgs::msg::Marker::ADD;
          detection_marker.pose = detection_msg.pose.pose;
          detection_marker.scale.x = 0.3;
          detection_marker.scale.y = 0.3;
          detection_marker.scale.z = 0.3;
          detection_marker.color.r = 1.0;
          detection_marker.color.g = 0.0;
          detection_marker.color.b = 0.0;
          detection_marker.color.a = 1.0;
          detection_marker.lifetime.sec = 0;
          detection_markers.markers.push_back(detection_marker);
          marker_id++;
        }
      }
      
      detection_pub_->publish(detection_array_msg);
      detection_markers_pub_->publish(detection_markers);
    }

    RCLCPP_DEBUG(this->get_logger(), "Published %zu 3D detections.", detection_array_msg.detections.size());
  }

private:
  rcl_interfaces::msg::SetParametersResult onParameterEventCallback(const std::vector<rclcpp::Parameter> &parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "Parameter updated successfully";

    for (const auto &param : parameters) 
    {
      if (param.get_name() == "min_x") 
      {
        min_x_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Parameter 'min_x' updated to: %f", min_x_);
      } 
      else if (param.get_name() == "max_x") 
      {
        max_x_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Parameter 'max_x' updated to: %f", max_x_);
      } 
      else if (param.get_name() == "min_y") 
      {
        min_y_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Parameter 'min_y' updated to: %f", min_y_);
      } 
      else if (param.get_name() == "max_y") 
      {
        max_y_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Parameter 'max_y' updated to: %f", max_y_);
      } 
      else if (param.get_name() == "min_z") 
      {
        min_z_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Parameter 'min_z' updated to: %f", min_z_);
      } 
      else if (param.get_name() == "max_z") 
      {
        max_z_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Parameter 'max_z' updated to: %f", max_z_);
      } 
      else if (param.get_name() == "ground_plane_distance_threshold") 
      {
        ground_plane_distance_threshold_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Parameter 'ground_plane_distance_threshold' updated to: %f", ground_plane_distance_threshold_);
      }
      else if (param.get_name() == "cluster_tolerance") 
      {
        cluster_tolerance_ = param.as_double();
        RCLCPP_INFO(this->get_logger(), "Parameter 'cluster_tolerance' updated to: %f", cluster_tolerance_);
      }
      else if (param.get_name() == "min_cluster_size")
      {
        min_cluster_size_ = param.as_int();
        RCLCPP_INFO(this->get_logger(), "Parameter 'min_cluster_size' updated to: %d", min_cluster_size_);
      }
      else if (param.get_name() == "max_cluster_size") 
      {
        max_cluster_size_ = param.as_int();
        RCLCPP_INFO(this->get_logger(), "Parameter 'max_cluster_size' updated to: %d", max_cluster_size_);
      } 
      else 
      {
        RCLCPP_WARN(this->get_logger(), "Unknown parameter received: %s", param.get_name().c_str());
      }
    }
    return result;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    RCLCPP_DEBUG(this->get_logger(), "Received PointCloud2 message. Processing...");

    detection_msgs::msg::Detection3DArray objects;
    cluster(msg, objects);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

  rclcpp::Publisher<detection_msgs::msg::Detection3DArray>::SharedPtr detection_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr detection_markers_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pc_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr non_ground_pc_pub_;

  double min_x_;
  double min_y_;
  double min_z_;
  double max_x_;
  double max_y_;
  double max_z_;
  double ground_plane_distance_threshold_;
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  float voxel_leaf_size_;
  int min_points_per_voxel_;
  int min_voxel_cluster_size_for_filtering_;
  int max_voxel_cluster_for_output_;
  int max_points_per_voxel_in_large_cluster_;

  OnSetParametersCallbackHandle::SharedPtr callback_handle_;

  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarDetectorNode>());
  rclcpp::shutdown();

  return 0;
}
