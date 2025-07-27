#include <fstream>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "detection_msgs/msg/detection2_d.hpp"
#include "detection_msgs/msg/detection2_d_array.hpp"

class CameraDetectorNode : public rclcpp::Node
{
public:
  CameraDetectorNode() : Node("camera_detector_node")
  {
    this->declare_parameter<std::string>("model_cfg_path", "");
    this->declare_parameter<std::string>("model_weights_path", "");
    this->declare_parameter<std::string>("class_names_path", "");
    this->declare_parameter<double>("confidence_threshold", 0.5);
    this->declare_parameter<double>("nms_threshold", 0.4);

    this->get_parameter("model_cfg_path", model_cfg_path_);
    this->get_parameter("model_weights_path", model_weights_path_);
    this->get_parameter("class_names_path", class_names_path_);
    this->get_parameter("confidence_threshold", confidence_threshold_);
    this->get_parameter("nms_threshold", nms_threshold_);

    // Load model
    try 
    {
      net_ = cv::dnn::readNetFromDarknet(model_cfg_path_, model_weights_path_);
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
      RCLCPP_INFO(this->get_logger(), "YOLOv3 model loaded successfully with CUDA backend");
    }
    catch (const cv::Exception& ex)
    {
      RCLCPP_WARN(this->get_logger(), "Couldn't load YOLOv3 model with CUDA. Falling back to CPU. Error: %s", ex.what());
      net_ = cv::dnn::readNetFromDarknet(model_cfg_path_, model_weights_path_);
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    loadClassNames(class_names_path_);

    image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/kitti_imgraw_L/RGB", 10, std::bind(&CameraDetectorNode::imageCallback, this, std::placeholders::_1));

    detection_publisher_ = this->create_publisher<detection_msgs::msg::Detection2DArray>(
        "/camera_detections", 10);

    image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/processed_image", 10);

    RCLCPP_INFO(this->get_logger(), "CameraDetectorNode initialized.");
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Publisher<detection_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;

  // Parameters
  std::string model_cfg_path_;
  std::string model_weights_path_;
  std::string class_names_path_;
  double confidence_threshold_;
  double nms_threshold_;

private:
  void loadClassNames(const std::string& filename)
  {
    std::ifstream ifs(filename.c_str());
    if (!ifs.is_open())
    {
      RCLCPP_ERROR(this->get_logger(), "Cannot open class names file: %s", filename.c_str());
      return;
    }

    std::string line;
    while (std::getline(ifs, line))
    {
      class_names_.push_back(line);
    }
    RCLCPP_INFO(this->get_logger(), "Loaded %zu class names.", class_names_.size());
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    RCLCPP_DEBUG(this->get_logger(), "Received image. Processing...");
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat frame = cv_ptr->image;
    if (frame.empty())
    {
      RCLCPP_WARN(this->get_logger(), "Empty frame received");
      return;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(), true, false);
    net_.setInput(blob);

    std::vector<cv::String> output_layers = net_.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs;
    net_.forward(outs, output_layers);

    processDetections(frame, outs, msg->header);
    
    sensor_msgs::msg::Image processed_image;
    processed_image.header = msg->header;
    cv_ptr->toImageMsg(processed_image);
    image_publisher_->publish(processed_image);
  }

  void processDetections(cv::Mat& frame, const std::vector<cv::Mat>& outs, const std_msgs::msg::Header& header)
  {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); i++)
    {
      float* data = reinterpret_cast<float*>(outs[i].data);
      for (size_t j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
      {
        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        cv::Point class_id_point;
        double confidence;
        minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
        if (confidence > confidence_threshold_) 
        {
          int center_x = (int)(data[0] * frame.cols);
          int center_y = (int)(data[1] * frame.rows);
          int width = (int)(data[2] * frame.cols);
          int height = (int)(data[3] * frame.rows);
          int left = center_x - width / 2;
          int top = center_y - height / 2;

          class_ids.push_back(class_id_point.x);
          confidences.push_back((float)confidence);
          boxes.push_back(cv::Rect(left, top, width, height));
        }
      }    
    }

    // Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indices);

    detection_msgs::msg::Detection2DArray detection_array_msg;
    detection_array_msg.header = header;

    for (size_t i = 0; i < indices.size(); ++i) 
    {
      int idx = indices[i];
      cv::Rect box = boxes[idx];

      detection_msgs::msg::Detection2D detection_msg;
      detection_msg.bbox.x_offset = box.x;
      detection_msg.bbox.y_offset = box.y;
      detection_msg.bbox.width = box.width;
      detection_msg.bbox.height = box.height;
      detection_msg.score = confidences[idx];

      if (class_ids[idx] < class_names_.size()) 
      {
        detection_msg.class_name = class_names_[class_ids[idx]];
        cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 3);
        cv::putText(frame, detection_msg.class_name, cv::Point(box.x, box.y - 5), 2, 2, cv::Scalar(255, 0, 0));
      } 
      else 
      {
        detection_msg.class_name = "unknown";
      }
      detection_array_msg.detections.push_back(detection_msg);
    }

    detection_publisher_->publish(detection_array_msg);
    RCLCPP_DEBUG(this->get_logger(), "Published %zu detections", detection_array_msg.detections.size());
  }
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraDetectorNode>());
  rclcpp::shutdown();

  return 0;
}
