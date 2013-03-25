/*
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
#include <ros/package.h>

#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/JointState.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// TF
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>

// Boost
#include <boost/shared_ptr.hpp>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp> 
#include <opencv2/nonfree/features2d.hpp>


// STL
#include <vector>
#include <deque>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <float.h>
#include <math.h>
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND
#include <sstream>
#include <sys/time.h>


typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
        sensor_msgs::Image,
        sensor_msgs::PointCloud2> MySyncPolicy;

typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;

using boost::shared_ptr;

class GPUTest
{
  public:
    GPUTest(ros::NodeHandle &n) :
      n_(n), n_private_("~"),
      image_sub_(n_, "color_image_topic", 1),
      depth_sub_(n_, "depth_image_topic", 1),
      mask_sub_(n_, "mask_image_topic", 1),
      cloud_sub_(n_, "point_cloud_topic", 1),
      // sync_(MySyncPolicy(15), image_sub_, depth_sub_, mask_sub_, cloud_sub_),
      sync_(MySyncPolicy(15), image_sub_, depth_sub_, cloud_sub_),
      it_(n_), tf_(), camera_initialized_(false)
  {
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());

    n_private_.param("display_wait_ms", display_wait_ms_, 3);
    n_private_.param("num_downsmaples", num_downsamples_, 2);

    std::string default_optical_frame = "/head_mount_kinect_rgb_optical_frame";
    n_private_.param("optical_frame", optical_frame_, default_optical_frame);
    std::string cam_info_topic_def = "/kinect_head/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_, cam_info_topic_def);
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_, default_workspace_frame);

    std::string default_left_tool_frame = "/l_gripper_tool_frame";
    n_private_.param("l_tool_frame", left_tool_frame_, default_left_tool_frame);
    std::string default_right_tool_frame = "/r_gripper_tool_frame";
    n_private_.param("r_tool_frame", right_tool_frame_, default_right_tool_frame);


    // Setup ros node connections
    sync_.registerCallback(&GPUTest::sensorCallback, this);


    ROS_INFO("[GripperSeg] Node Initialization Complete");
  }


    void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg,
        const sensor_msgs::ImageConstPtr& depth_msg,
        const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      // Store camera information only once
      if (!camera_initialized_)
      {
        cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic_, n_, ros::Duration(3.0));
        camera_initialized_ = true;
        ROS_INFO("[GripperCollector]Initialization: Camera Info Done");

      }

      timespec t1, t2;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);


      cv::Mat color_frame, depth_frame, self_mask;
      cv_bridge::CvImagePtr color_cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
      cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg);

      color_frame = color_cv_ptr->image;
      depth_frame = depth_cv_ptr->image;

      // cv_bridge::CvImagePtr mask_cv_ptr = cv_bridge::toCvCopy(mask_msg);
      // self_mask = mask_cv_ptr->image;
      //      XYZPointCloud cloud;
      //      pcl::fromROSMsg(*cloud_msg, cloud);
      //      tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
      //          cloud.header.stamp, ros::Duration(33e-3));
      //      pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);
      //        cloud_ = cloud;
      //      cur_camera_header_ = img_msg->header;


      // Everything in GPU Mode
      // cvmat to gpumat
      cv::gpu::GpuMat color_frame_gpu;
      color_frame_gpu.upload(color_frame);
      cv::gpu::SURF_GPU surf;
      cv::gpu::GpuMat mask;
      cv::gpu::GpuMat keypoints_gpu;

      surf(color_frame_gpu, mask, keypoints_gpu);
      std::vector<cv::KeyPoint> keypoints;
      surf.downloadKeypoints(keypoints_gpu, keypoints);

      printf("%d", (int)keypoints.size());

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
      printf("SCB: "); printTime(t1, t2);
      cv::imshow("Result", color_frame);
      cv::waitKey();
    }

    void printTime(timespec start, timespec end)
    {
      //ROS_INFO("Time: %.6fms", (end.tv_nsec - start.tv_nsec)/1e6);
      ROS_INFO ("Time: %.6fms\n", (end.tv_sec - start.tv_sec)*1e3 + (end.tv_nsec - start.tv_nsec)/1e6);
    }


  protected:
    ros::NodeHandle n_;
    ros::NodeHandle n_private_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Synchronizer<MySyncPolicy> sync_;
    image_transport::ImageTransport it_;
    sensor_msgs::CameraInfo cam_info_;
    shared_ptr<tf::TransformListener> tf_;

    // frames
    cv::Mat cur_workspace_mask_;
    std_msgs::Header cur_camera_header_;
    XYZPointCloud cloud_;
    int display_wait_ms_;
    int num_downsamples_;
    std::string workspace_frame_;
    std::string optical_frame_;
    std::string left_tool_frame_;
    std::string right_tool_frame_;
    std::string tool_frame_;
    std::string task_frame_;
    bool camera_initialized_;
    bool desire_points_initialized_;
    std::string cam_info_topic_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_seg");

  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
  //my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Debug]);
  ros::NodeHandle n;
  GPUTest vsa(n);
  ros::spin();
  return 0;
}
