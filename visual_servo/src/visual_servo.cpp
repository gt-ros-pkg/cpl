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
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

// TF
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>

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

// Others
#include <visual_servo/VisualServoTwist.h>

#define DEBUG_MODE 0
#define JACOBIAN_TYPE_INV 1
#define JACOBIAN_TYPE_PSEUDO 2
#define JACOBIAN_TYPE_AVG 3

typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;
typedef struct {
  // pixels
  cv::Point image;
  // meters, note that xyz for this is different from the other one
  pcl::PointXYZ camera;
  // the 3d location in workspace frame
  pcl::PointXYZ workspace;
  pcl::PointXYZ workspace_angular;
} VSXYZ;

// using boost::shared_ptr;
using geometry_msgs::TwistStamped;
using geometry_msgs::PointStamped;
using visual_servo::VisualServoTwist;

/**
 * We are taping robot hand with three blue painter's tape 
 * and use those three features to do the image-based visual servoing
 * Uses Kinect as image sensor and provide a twist computation service
 */
class VisualServo
{
public:
  VisualServo(int jacobian_type) :
  n_private_("~"), desired_jacobian_set_(false)
  {
    // user must specify the jacobian type
    jacobian_type_ = jacobian_type;
    std::string default_optical_frame = "/openni_rgb_optical_frame";
    n_private_.param("optical_frame", optical_frame_, default_optical_frame);
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_, default_workspace_frame);

    // others
    n_private_.param("gain_vel", gain_vel_, 1.0);
    n_private_.param("gain_rot", gain_rot_, 1.0);
    // n_private_.param("jacobian_type", jacobian_type_, JACOBIAN_TYPE_INV);
    
    ros::Time now = ros::Time(0);
    try 
    {
      tf::TransformListener listener;
      listener.waitForTransform(workspace_frame_, optical_frame_,  now, ros::Duration(2.0));
      listener.lookupTransform(workspace_frame_, optical_frame_,  now, transform);
    }
    catch (tf::TransformException e)
    {
      // return 0 value in case of error so the arm stops
      ROS_WARN_STREAM(e.what());
      ros::shutdown();
    }
  }

  
  visual_servo::VisualServoTwist computeTwist(std::vector<VSXYZ> desired_locations, std::vector<VSXYZ> hand_features)
  {
    visual_servo::VisualServoTwist srv;
    
    cv::Mat twist = computeTwistCamera(desired_locations, hand_features);
    cv::Mat temp = twist.clone(); 
    /*
    tf::StampedTransform transform; 
    ros::Time now = ros::Time(0);
    try 
    {
      tf::TransformListener listener;
      listener.waitForTransform(workspace_frame_, optical_frame_,  now, ros::Duration(2.0));
      listener.lookupTransform(workspace_frame_, optical_frame_,  now, transform);
    }
    catch (tf::TransformException e)
    {
      // return 0 value in case of error so the arm stops
      ROS_WARN_STREAM(e.what());
      return srv;
    }
    */
    
    // have to transform twist in camera frame (openni_rgb_optical_frame) to torso frame (torso_lift_link)
    tf::Vector3 twist_rot(temp.at<float>(3), temp.at<float>(4), temp.at<float>(5));
    tf::Vector3 twist_vel(temp.at<float>(0), temp.at<float>(1), temp.at<float>(2));  

    // twist transformation from optical frame to workspace frame
    tf::Vector3 out_rot = transform.getBasis() * twist_rot;
    tf::Vector3 out_vel = transform.getBasis() * twist_vel + transform.getOrigin().cross(out_rot);
    
    // multiple the velocity and rotation by gain defined in the parameter
    srv.request.twist.twist.linear.x  = out_vel.x()*gain_vel_;
    srv.request.twist.twist.linear.y  = out_vel.y()*gain_vel_;
    srv.request.twist.twist.linear.z  = out_vel.z()*gain_vel_;
    srv.request.twist.twist.angular.x = out_rot.x()*gain_rot_;
    srv.request.twist.twist.angular.y = out_rot.y()*gain_rot_;
    srv.request.twist.twist.angular.z = out_rot.z()*gain_rot_;
    
    return srv;
  }
  
  void printMatrix(cv::Mat_<double> in)
  {
    for (int i = 0; i < in.rows; i++) {
      for (int j = 0; j < in.cols; j++) {
        printf("%+.5f\t", in(i,j)); 
      }
      printf("\n");
    }
  }
  bool setDesiredInteractionMatrix(std::vector<VSXYZ> &pts) 
  {
    try 
    {
      cv::Mat im = getMeterInteractionMatrix(pts);
      if (countNonZero(im) > 0)
      {
        desired_jacobian_ = im;
        desired_jacobian_set_ = true;
        return true;
      }
    }
    catch(ros::Exception e)
    {
    }
    return false;
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
  // shared_ptr<tf::TransformListener> tf_;
  std::string workspace_frame_;
  std::string optical_frame_;
  tf::StampedTransform transform;
 
  // others
  int jacobian_type_;
  double gain_vel_;
  double gain_rot_;
  double term_threshold_;
  bool desired_jacobian_set_;
  cv::Mat desired_jacobian_;

  // compute twist in camera frame 
  cv::Mat computeTwistCamera(std::vector<VSXYZ> desired, std::vector<VSXYZ> pts)
  {
    cv::Mat error_mat;
    cv::Mat im;
    float rmse_e = 0;

    // return 0 twist if number of features does not equal to that of desired features
    if (pts.size() != desired.size())
      return cv::Mat::zeros(6, 1, CV_32F);

    // for all three features,
    for (unsigned int i = 0; i < desired.size(); i++) 
    {
      cv::Mat error = cv::Mat::zeros(2, 1, CV_32F);
      error.at<float>(0,0) = (pts.at(i)).camera.x - (desired.at(i)).camera.x;
      error.at<float>(1,0) = (pts.at(i)).camera.y - (desired.at(i)).camera.y;
      error_mat.push_back(error);

      // RMS of error for termination condition
      rmse_e += pow(error.at<float>(0,0),2) + pow(error.at<float>(1,0),2);
    }
    
    // printf("Error in camera:\t"); printMatrix(error_mat.t());

    im = getMeterInteractionMatrix(pts);
    
    // if we can't compute interaction matrix, just make all twists 0
    if (countNonZero(im) == 0) return cv::Mat::zeros(6, 1, CV_32F);
    
    // inverting the matrix, 3 approaches
    cv::Mat iim;
    switch (jacobian_type_) 
    {
      case JACOBIAN_TYPE_INV:
        iim = (im).inv();
        break;
      case JACOBIAN_TYPE_PSEUDO:
        iim = pseudoInverse(im);
        break;
      default: // JACOBIAN_TYPE_AVG
        // We use specific way shown on visual servo by Chaumette 2006
        
        if (!desired_jacobian_set_) return cv::Mat::zeros(6, 1, CV_32F);
        
        cv::Mat temp = desired_jacobian_ + im;
        iim = 0.5 * pseudoInverse(temp);
    }

    // Gain Matrix K (different from camera intrinsics)
    cv::Mat gain = cv::Mat::eye(6,6, CV_32F);

    // K x IIM x ERROR = TWIST
    return gain*iim*error_mat;
  }
  
  cv::Mat pseudoInverse(cv::Mat im) 
  {
    return (im.t() * im).inv()*im.t();
  }
  
 
  /**
   * get the interaction matrix
   * @param depth_frame  Need the depth information from Kinect for Z
   * @param pts          Vector of feature points
   * @return             Return the computed interaction Matrix (6 by 6)
   */
  cv::Mat getMeterInteractionMatrix(std::vector<VSXYZ> &pts) 
  {
    int size = (int)pts.size();
    // interaction matrix, image jacobian
    cv::Mat L = cv::Mat::zeros(size*2, 6,CV_32F);
    //if (pts.size() != 3) 
    //  return L;
    for (int i = 0; i < size; i++) {
      pcl::PointXYZ xyz= pts.at(i).camera;
      float x = xyz.x;
      float y = xyz.y;
      float Z = xyz.z;
      if (Z < 0.01 || Z > 0.95)
      {
        ROS_ERROR("Incorrect Z (%f). Cannot Compute Jacobian", Z);
        return cv::Mat::zeros(size*2, 6, CV_32F);
      }
      // float z = cur_point_cloud_.at(y,x).z;
      int l = i * 2;
      if (isnan(Z)) return cv::Mat::zeros(6,6, CV_32F);
      L.at<float>(l,0) = -1/Z;   L.at<float>(l+1,0) = 0;
      L.at<float>(l,1) = 0;      L.at<float>(l+1,1) = -1/Z;
      L.at<float>(l,2) = x/Z;    L.at<float>(l+1,2) = y/Z;
      L.at<float>(l,3) = x*y;    L.at<float>(l+1,3) = (1 + pow(y,2));
      L.at<float>(l,4) = -(1+pow(x,2));  L.at<float>(l+1,4) = -x*y;
      L.at<float>(l,5) = y;      L.at<float>(l+1,5) = -x;
    }
    return L;
  }
};
