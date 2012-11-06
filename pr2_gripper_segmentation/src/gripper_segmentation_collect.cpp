/*********************************************************************
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

// cpl_visual_features
#include <cpl_visual_features/helpers.h>
#include <cpl_visual_features/features/shape_context.h>

// others
#include <pr2_controllers_msgs/JointTrajectoryControllerState.h>
#include <pr2_gripper_segmentation/GripperPose.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
        sensor_msgs::Image,
        sensor_msgs::Image,
        sensor_msgs::PointCloud2> MySyncPolicy;
typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;

using sensor_msgs::JointState;
using boost::shared_ptr;
using geometry_msgs::TwistStamped;
using geometry_msgs::PoseStamped;
using geometry_msgs::QuaternionStamped;
using geometry_msgs::Point;
using cpl_visual_features::downSample;

class GripperSegmentationCollector
{
  public:
    GripperSegmentationCollector(ros::NodeHandle &n) :
      n_(n), n_private_("~"),
      image_sub_(n_, "color_image_topic", 1),
      depth_sub_(n_, "depth_image_topic", 1),
      mask_sub_(n_, "mask_image_topic", 1),
      cloud_sub_(n_, "point_cloud_topic", 1),
      sync_(MySyncPolicy(15), image_sub_, depth_sub_, mask_sub_, cloud_sub_),
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
    sync_.registerCallback(&GripperSegmentationCollector::sensorCallback, this);
    // n_.subscribe("/joint_states", 1, &GripperSegmentationCollector::jointStateCallback, this);
    p_client_ = n.serviceClient<pr2_gripper_segmentation::GripperPose>("pgs_pose");
    textFile.open("/u/swl33/data/myData.csv");

    float bound = 0.3;
    x_= 0.4;
    y_= -bound;
    z_= -bound;
    which_arm_ = "l";
  }

    // destructor
    ~GripperSegmentationCollector()
    {
      textFile.close();
    }


  private:

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& js)
    //void jointStateCallback(boost::shared_ptr<sensor_msgs::JointState const> js)
    {
      ROS_INFO("I am here");
      for (unsigned int i = 0; i < js->name.size(); i++)
      {
        ROS_INFO("%s", js->name[i]);
      }
    }
    unsigned int counter;
    /**
     * Called when Kinect information is avaiable. Refresh rate of about 30Hz 
     */
    void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg,
        const sensor_msgs::ImageConstPtr& depth_msg,
        const sensor_msgs::ImageConstPtr& mask_msg,
        const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      // Store camera information only once
      if (!camera_initialized_)
      {
        cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic_, n_, ros::Duration(3.0));
        camera_initialized_ = true;
        counter = 0;
        ROS_INFO("[GripperCollector]Initialization: Camera Info Done");
      }

      cv::Mat color_frame, depth_frame, self_mask;
      cv_bridge::CvImagePtr color_cv_ptr = cv_bridge::toCvCopy(img_msg);
      cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg);
      cv_bridge::CvImagePtr mask_cv_ptr = cv_bridge::toCvCopy(mask_msg);

      color_frame = color_cv_ptr->image;
      depth_frame = depth_cv_ptr->image;
      self_mask = mask_cv_ptr->image;

      // Downsample everything first
      /*
      cv::Mat color_frame_down = downSample(color_frame, num_downsamples_);
      cv::Mat depth_frame_down = downSample(depth_frame, num_downsamples_);
      cv::Mat self_mask_down = downSample(self_mask, num_downsamples_);
       */

      XYZPointCloud cloud;
      pcl::fromROSMsg(*cloud_msg, cloud);
      tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
          cloud.header.stamp, ros::Duration(0.9));
      pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);
      cur_camera_header_ = img_msg->header;
      cur_color_frame_ = color_frame;
      cur_depth_frame_ = depth_frame.clone();
      cur_point_cloud_ = cloud;


      // ====== L,R JOINT ANGLE ======
      double current_angles[14];
      get_current_joint_angles(current_angles);
      textFile << counter << ",";
      for (int i = 0; i < 14; i++)
      {
        textFile << current_angles[i] << ",";
      }
      // ====== FOWARD KINEMATICS L,R TOOLTIP POSE ======
      get_fk_tooltip_pose(current_angles);
      for (int i = 0; i < 14; i++)
      {
        textFile << current_angles[i] << ",";
      }

      textFile << "\n";

      std::string file1 = getFileName(counter, which_arm_);
      cv::imwrite(file1 + ".jpg", color_frame);
      cv::imwrite(file1 + "m.jpg", self_mask);

      cv::imshow("show", color_frame);
      cv::waitKey(3);
      counter++;
      ROS_INFO("[%u][%f %f %f] Sampled", counter, x_, y_,z_);

      z_+= 0.10;
      if (z_ > 0.00)
      {
        z_ = -0.3;
        y_+= 0.10;
        if (y_ > 0.3)
        {
          y_ = -0.3;
          x_ += 0.1;
          if (x_ > 0.6)
          {
            if (which_arm_.compare("r") == 0)
              ros::shutdown();
            else
              x_ = 0.4;
          }
        }
      }
      pr2_gripper_segmentation::GripperPose psrv;
      PoseStamped p;
      p.header.frame_id = workspace_frame_;
      p.pose.position.x = x_;
      p.pose.position.y = y_;
      p.pose.position.z = z_;
      p.pose.orientation.w = 1;

      psrv.request.arm = which_arm_;
      psrv.request.p = p;
      if (p_client_.call(psrv)){}
      else {ROS_WARN("Service Fail");}


    }

    std::string getFileName(unsigned int counter, std::string which_arm)
    {
      std::ostringstream os, os2;
      os << "/u/swl33/data/";
      if (counter / 100 == 0)
        os << "0";
      if (counter / 10 == 0)
        os << "0";
      os <<  counter;
      os << which_arm;
      return os.str();
    }

    void get_fk_tooltip_pose(double cur[14])
    {
      PoseStamped p, pl, pr;
      p.pose.orientation.w = 1;

      p.header.frame_id = "/l_gripper_tool_frame";
      tf_->transformPose(workspace_frame_, p, pl);
      cur[0]=pl.pose.position.x;
      cur[1]=pl.pose.position.y;
      cur[2]=pl.pose.position.z;
      cur[3]=pl.pose.orientation.x;
      cur[4]=pl.pose.orientation.y;
      cur[5]=pl.pose.orientation.z;
      cur[6]=pl.pose.orientation.w;

      p.header.frame_id = "/r_gripper_tool_frame";
      tf_->transformPose(workspace_frame_, p, pr);
      cur[7]=pr.pose.position.x;
      cur[8]=pr.pose.position.y;
      cur[9]=pr.pose.position.z;
      cur[10]=pr.pose.orientation.x;
      cur[11]=pr.pose.orientation.y;
      cur[12]=pr.pose.orientation.z;
      cur[13]=pr.pose.orientation.w;
    }

    //figure out where the arm is now  
    void get_current_joint_angles(double current_angles[14]){
      int i;
      //get a single message from the topic 'l_arm_controller/state'
      pr2_controllers_msgs::JointTrajectoryControllerStateConstPtr l_state_msg = ros::topic::waitForMessage<pr2_controllers_msgs::JointTrajectoryControllerState>("l_arm_controller/state");
      //extract the joint angles from it
      for(i=0; i<7; i++){
        current_angles[i] = l_state_msg->actual.positions[i];
      }
      //get a single message from the topic 'r_arm_controller/state'
      pr2_controllers_msgs::JointTrajectoryControllerStateConstPtr r_state_msg = ros::topic::waitForMessage<pr2_controllers_msgs::JointTrajectoryControllerState>("r_arm_controller/state");
      for(i=7; i<14; i++){
        current_angles[i] = r_state_msg->actual.positions[i-7];
      }
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
    cv::Mat cur_color_frame_;
    cv::Mat cur_orig_color_frame_;
    cv::Mat cur_depth_frame_;
    cv::Mat cur_workspace_mask_;
    std_msgs::Header cur_camera_header_;
    XYZPointCloud cur_point_cloud_;
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

    std::ofstream textFile;
    ros::ServiceClient p_client_;

    float x_, y_, z_;
    std::string which_arm_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_seg");

  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
  //my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Debug]);
  ros::NodeHandle n;
  GripperSegmentationCollector vsa(n);
  ros::spin();
  return 0;
}
