/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Georgia Institute of Technology
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Georgia Institute of Technology nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/
// ROS
#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/CvBridge.h>

// TF
#include <tf/transform_listener.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl/ros/conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost
#include <boost/shared_ptr.hpp>

// STL
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <iostream>
#include <utility>
#include <float.h>
#include <math.h>
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND

using boost::shared_ptr;
typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image,
                                                        sensor_msgs::PointCloud2> MySyncPolicy;

class DataCollectNode
{
 public:
  DataCollectNode(ros::NodeHandle &n) :
      n_(n), n_private_("~"),
      image_sub_(n, "color_image_topic", 1),
      depth_sub_(n, "depth_image_topic", 1),
      cloud_sub_(n, "point_cloud_topic", 1),
      sync_(MySyncPolicy(15), image_sub_, depth_sub_, cloud_sub_),
      camera_initialized_(false), save_count_(0)
  {
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());
    // Get parameters from the server
    n_private_.param("display_wait_ms", display_wait_ms_, 3);
    n_private_.param("save_all", save_all_, false);

    std::string output_path_def = "~";
    n_private_.param("img_output_path", base_output_path_, output_path_def);

    std::string cam_info_topic_def = "/kinect_head/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_,
                     cam_info_topic_def);
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_,
                     default_workspace_frame);
    n_private_.param("max_depth", max_depth_, 4.0);

    // Setup ros node connections
    sync_.registerCallback(&DataCollectNode::sensorCallback,
                           this);
  }

  void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg,
                      const sensor_msgs::ImageConstPtr& depth_msg,
                      const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    if (!camera_initialized_)
    {
      cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
          cam_info_topic_, n_, ros::Duration(5.0));
      camera_initialized_ = true;
      ROS_INFO_STREAM("Cam info: " << cam_info_);
    }
    // Convert images to OpenCV format
    cv::Mat color_frame(bridge_.imgMsgToCv(img_msg));
    cv::Mat depth_frame(bridge_.imgMsgToCv(depth_msg));

    // Swap kinect color channel order
    cv::cvtColor(color_frame, color_frame, CV_RGB2BGR);

    // Convert nans to zeros
    for (int r = 0; r < depth_frame.rows; ++r)
    {
      float* depth_row = depth_frame.ptr<float>(r);
      for (int c = 0; c < depth_frame.cols; ++c)
      {
        float cur_d = depth_row[c];
        if (isnan(cur_d))
        {
          depth_row[c] = 0.0;
        }
      }
    }

    // color_frame, depth_frame
    std::stringstream color_out;
    std::stringstream depth_out;
    color_out << base_output_path_ << "/color" << save_count_ << ".png";
    depth_out << base_output_path_ << "/depth" << save_count_ << ".png";
    cv::Mat depth_save_img(depth_frame.size(), CV_16UC1);
    depth_frame.convertTo(depth_save_img, CV_16UC1, 65535/max_depth_);
    cv::imshow("color", color_frame);
    cv::imshow("depth", depth_save_img);
    char c = cv::waitKey(display_wait_ms_);

    ROS_INFO_STREAM("Writting image number " << save_count_);
    if (c == 's' || save_all_)
    {
      cv::imwrite(color_out.str(), color_frame);
      cv::imwrite(depth_out.str(), depth_save_img);
      // save point cloud to disk
      std::stringstream cloud_out;
      cloud_out << base_output_path_ << "/cloud" << save_count_ << ".pcd";
      pcl::io::savePCDFile(cloud_out.str(), *cloud_msg);
      save_count_++;
    }
  }

  /**
   * Executive control function for launching the node.
   */
  void spin()
  {
    while(n_.ok())
    {
      ros::spinOnce();
    }
  }

 protected:
  ros::NodeHandle n_;
  ros::NodeHandle n_private_;
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
  message_filters::Synchronizer<MySyncPolicy> sync_;
  sensor_msgs::CameraInfo cam_info_;
  sensor_msgs::CvBridge bridge_;
  shared_ptr<tf::TransformListener> tf_;
  int display_wait_ms_;
  bool save_all_;
  std::string base_output_path_;
  std::string cam_info_topic_;
  std::string workspace_frame_;
  bool camera_initialized_;
  int save_count_;
  double max_depth_;
};

int main(int argc, char ** argv)
{
  int seed = time(NULL);
  srand(seed);
  ros::init(argc, argv, "data_node");
  ros::NodeHandle n;
  DataCollectNode data_node(n);
  data_node.spin();
  return 0;
}

