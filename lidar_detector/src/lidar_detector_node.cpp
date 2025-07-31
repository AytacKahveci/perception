#include <string>
#include <memory>

#include "detection_msgs/msg/detection3_d.hpp"
#include "detection_msgs/msg/detection3_d_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

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


class LidarDetectorNode : public rclcpp::Node
{
public:
  LidarDetectorNode() : Node("lidar_detector_node")
  {
    this->declare_parameter("min_x", 0.0);
    this->declare_parameter("min_y", -10.0);
    this->declare_parameter("min_z", -2.0);
    this->declare_parameter("max_x", 20.0);
    this->declare_parameter("max_y", 10.0);
    this->declare_parameter("max_z", 1.0);
    this->declare_parameter("ground_plane_distance_threshold", 0.2);
    this->declare_parameter("cluster_tolerance", 0.5);
    this->declare_parameter("min_cluster_size", 10);
    this->declare_parameter("max_cluster_size", 25000);

    this->get_parameter("min_x", min_x_);
    this->get_parameter("min_y", min_y_);
    this->get_parameter("min_z", min_z_);
    this->get_parameter("max_x", max_x_);
    this->get_parameter("max_y", max_y_);
    this->get_parameter("max_z", max_z_);

    this->get_parameter("ground_plane_distance_threshold", ground_plane_distance_threshold_);
    this->get_parameter("cluster_tolerance", cluster_tolerance_);
    this->get_parameter("min_cluster_size", min_cluster_size_);
    this->get_parameter("max_cluster_size", max_cluster_size_);
    
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PCLPointCloud2 cloud2;
    pcl_conversions::moveToPCL(*msg, cloud2);
    pcl::fromPCLPointCloud2(cloud2, *cloud);

    // ROI Filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_roi(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PassThrough<pcl::PointXYZ> passthrough_filter;
    passthrough_filter.setInputCloud(cloud);
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
    pcl::PCLPointCloud2 cloud_filtered_roi_pcd2;
    pcl::toPCLPointCloud2(*cloud_filtered_roi, cloud_filtered_roi_pcd2);
    pcl_conversions::fromPCL(cloud_filtered_roi_pcd2, filtered_roi_output);
    filtered_roi_output.header = msg->header;
    filtered_pub_->publish(filtered_roi_output);

    // Ground plane segmentation
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(ground_plane_distance_threshold_);
    seg.setInputCloud(cloud_filtered_roi);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) 
    {
      RCLCPP_WARN(this->get_logger(), "Could not estimate a planar model for the given dataset.");
      return;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered_roi);
    extract.setIndices(inliers);
    extract.setNegative(false);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>());
    extract.filter(*cloud_ground);

    sensor_msgs::msg::PointCloud2 ground_output;
    pcl::PCLPointCloud2 cloud_ground_pcd2;
    pcl::toPCLPointCloud2(*cloud_ground, cloud_ground_pcd2);
    pcl_conversions::fromPCL(cloud_ground_pcd2, ground_output);
    ground_output.header = msg->header;
    ground_pc_pub_->publish(ground_output);

    extract.setNegative(true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_non_ground(new pcl::PointCloud<pcl::PointXYZ>());
    extract.filter(*cloud_non_ground);

    sensor_msgs::msg::PointCloud2 non_ground_output;
    pcl::PCLPointCloud2 cloud_non_ground_pcd2;
    pcl::toPCLPointCloud2(*cloud_non_ground, cloud_non_ground_pcd2);
    pcl_conversions::fromPCL(cloud_non_ground_pcd2, non_ground_output);
    non_ground_output.header = msg->header;
    non_ground_pc_pub_->publish(non_ground_output);

    // Euclidean clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_non_ground);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_non_ground);
    ec.extract(cluster_indices);

    detection_msgs::msg::Detection3DArray detection_array_msg;
    detection_array_msg.header = msg->header;

    visualization_msgs::msg::MarkerArray detection_markers;
    int marker_id = 0;

    for (const auto& indices : cluster_indices) 
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& idx : indices.indices) 
      {
        cloud_cluster->push_back((*cloud_non_ground)[idx]);
      }
      cloud_cluster->width = cloud_cluster->size();
      cloud_cluster->height = 1;
      cloud_cluster->is_dense = true;

      pcl::PointXYZ min_pt, max_pt;
      pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt);

      detection_msgs::msg::Detection3D detection_msg;
      detection_msg.class_name = "object"; 
      detection_msg.score = 1.0;

      detection_msg.pose.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
      detection_msg.pose.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
      detection_msg.pose.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
      detection_msg.pose.pose.orientation.x = 0.0;
      detection_msg.pose.pose.orientation.y = 0.0;
      detection_msg.pose.pose.orientation.z = 0.0;
      detection_msg.pose.pose.orientation.w = 1.0;

      detection_msg.dimensions.x = max_pt.x - min_pt.x;
      detection_msg.dimensions.y = max_pt.y - min_pt.y;
      detection_msg.dimensions.z = max_pt.z - min_pt.z;
      
      //if (detection_msg.dimensions.x > 0.01 && detection_msg.dimensions.y > 0.01 && detection_msg.dimensions.z > 0.01)
      {
        detection_array_msg.detections.push_back(detection_msg);

        visualization_msgs::msg::Marker detection_marker;
        detection_marker.header = msg->header;
        detection_marker.header.stamp = this->now();
        detection_marker.ns = "object";
        detection_marker.id = marker_id;
        detection_marker.type = visualization_msgs::msg::Marker::CUBE;
        detection_marker.action = visualization_msgs::msg::Marker::ADD;
        detection_marker.pose = detection_msg.pose.pose;
        detection_marker.scale.x = max_pt.x - min_pt.x;
        detection_marker.scale.y = max_pt.y - min_pt.y;
        detection_marker.scale.z = max_pt.z - min_pt.z;
        detection_marker.color.r = 1.0;
        detection_marker.color.g = 0.0;
        detection_marker.color.b = 0.0;
        detection_marker.color.a = 1.0;
        detection_marker.lifetime.sec = 0;
        detection_markers.markers.push_back(detection_marker);
      }
    }

    detection_pub_->publish(detection_array_msg);
    detection_markers_pub_->publish(detection_markers);

    RCLCPP_DEBUG(this->get_logger(), "Published %zu 3D detections.", detection_array_msg.detections.size());
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

  OnSetParametersCallbackHandle::SharedPtr callback_handle_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarDetectorNode>());
  rclcpp::shutdown();

  return 0;
}
