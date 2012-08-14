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
#include <geometry_msgs/PoseStamped.h>
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

// Boost
#include <boost/shared_ptr.hpp>

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
using geometry_msgs::PoseStamped;
using visual_servo::VisualServoTwist;
using boost::shared_ptr;
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
    std::string default_optical_frame = "/head_mount_kinect_rgb_optical_frame";
    n_private_.param("optical_frame", optical_frame_, default_optical_frame);
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_, default_workspace_frame);

    // others
    n_private_.param("gain_vel", gain_vel_, 1.0);
    n_private_.param("gain_rot", gain_rot_, 1.0);
    // n_private_.param("jacobian_type", jacobian_type_, JACOBIAN_TYPE_INV);

    try
    {
      ros::Time now = ros::Time(0);
      listener_.waitForTransform(workspace_frame_, optical_frame_,  now, ros::Duration(2.0));
    }
    catch (tf::TransformException e)
    {
      ROS_WARN_STREAM(e.what());
    }
  }

    void setCamInfo(sensor_msgs::CameraInfo cam_info)
    {
      cam_info_ = cam_info;
    }

    visual_servo::VisualServoTwist computeTwist(std::vector<PoseStamped> goal, std::vector<PoseStamped> gripper, int mode)
    {
      switch (mode)
      {
      case 1:
        return PBVSTwist(goal, gripper);

      default:
        return IBVSTwist(goal, gripper);
      }
    }

    visual_servo::VisualServoTwist computeTwist(std::vector<VSXYZ> goal, std::vector<VSXYZ> gripper)
    {
      return IBVSTwist(goal, gripper);
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

    /**************************** 
     * Projection & Conversions
     ****************************/
    std::vector<VSXYZ> Point3DToVSXYZ(std::vector<pcl::PointXYZ> in, shared_ptr<tf::TransformListener> tf_)
    {
      std::vector<VSXYZ> ret;
      for (unsigned int i = 0; i < in.size(); i++)
      {
        ret.push_back(point3DToVSXYZ(in.at(i), tf_));
      }
      return ret;
    }

    VSXYZ point3DToVSXYZ(pcl::PointXYZ in, shared_ptr<tf::TransformListener> tf_)
    {
      // [X,Y,Z] -> [u,v] -> [x,y,z]
      VSXYZ ret;

      // 3D point in world frame
      pcl::PointXYZ in_c = in;
      cv::Point img = projectPointIntoImage(in_c, workspace_frame_, optical_frame_, tf_);
      cv::Mat temp = projectImagePointToPoint(img);

      float depth = sqrt(pow(in_c.z,2) + pow(temp.at<float>(0,0),2) + pow(temp.at<float>(1,0),2));
      pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), depth);
      ret.image = img;
      // for simpler simulator, this will be the same
      ret.camera = _2d;
      ret.workspace= in;
      return ret;
    }

    std::vector<VSXYZ> CVPointToVSXYZ(XYZPointCloud cloud, cv::Mat depth_frame, std::vector<cv::Point> in) 
    {
      std::vector<VSXYZ> ret;
      for (unsigned int i = 0; i < in.size(); i++)
      {
        ret.push_back(CVPointToVSXYZ(cloud, depth_frame, in.at(i)));
      }
      return ret;
    }

    VSXYZ CVPointToVSXYZ(XYZPointCloud cloud, cv::Mat depth_frame, cv::Point in) 
    {
      // [u,v] -> [x,y,z] (from camera intrinsics) & [X, Y, Z] (from PointCloud)
      VSXYZ ret;
      // pixel to meter value (using inverse of camera intrinsic) 
      cv::Mat temp = projectImagePointToPoint(in);

      // getZValue averages out z-value in a window to reduce noises
      pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), getZValue(depth_frame, in.x, in.y));
      pcl::PointXYZ _3d = cloud.at(in.x, in.y);

      ret.image = in;
      ret.camera = _2d;
      ret.workspace= _3d;
      return ret;
    }

    void printVSXYZ(VSXYZ i)
    {
      printf("Im: %+.3d %+.3d\tCam: %+.3f %+.3f %+.3f\twork: %+.3f %+.3f %+.3f\n",\
          i.image.x, i.image.y, i.camera.x, i.camera.y, i.camera.z, i.workspace.x, i.workspace.y, i.workspace.z);
    }

  protected:
    ros::NodeHandle n_;
    ros::NodeHandle n_private_;
    // shared_ptr<tf::TransformListener> tf_;
    std::string workspace_frame_;
    std::string optical_frame_;
    tf::StampedTransform transform;

    tf::TransformListener listener_;
    // others
    int jacobian_type_;
    double gain_vel_;
    double gain_rot_;
    double term_threshold_;
    bool desired_jacobian_set_;
    cv::Mat desired_jacobian_;
    cv::Mat K;
    sensor_msgs::CameraInfo cam_info_;

    visual_servo::VisualServoTwist PBVSTwist(std::vector<PoseStamped> desired, std::vector<PoseStamped> pts)
    {
      cv::Mat error_mat;
      cv::Mat im;

      // return 0 twist if number of features does not equal to that of desired features
      if (pts.size() != desired.size())
        return visual_servo::VisualServoTwist();
      //return cv::Mat::zeros(6, 1, CV_32F);

      // for all three features,
      for (unsigned int i = 0; i < desired.size(); i++)
      {
        cv::Mat error = cv::Mat::zeros(2, 1, CV_32F);
        error.at<float>(0,0) = (pts.at(i)).pose.position.x -
          (desired.at(i)).pose.position.x;
        error.at<float>(1,0) = (pts.at(i)).pose.position.y -
          (desired.at(i)).pose.position.y;
        error_mat.push_back(error);
      }

      im = getMeterInteractionMatrix(pts);
      std::string of = pts.at(0).header.frame_id;
      return computeTwist(error_mat, im, of);
    }



    // compute twist in camera frame 
    visual_servo::VisualServoTwist IBVSTwist(std::vector<VSXYZ> desired, std::vector<VSXYZ> pts)
    {
      cv::Mat error_mat;
      cv::Mat im;

      // return 0 twist if number of features does not equal to that of desired features
      if (pts.size() != desired.size())
        return visual_servo::VisualServoTwist();

      // for all three features,
      for (unsigned int i = 0; i < desired.size(); i++) 
      {
        cv::Mat error = cv::Mat::zeros(2, 1, CV_32F);
        error.at<float>(0,0) = (pts.at(i)).camera.x - (desired.at(i)).camera.x;
        error.at<float>(1,0) = (pts.at(i)).camera.y - (desired.at(i)).camera.y;
        error_mat.push_back(error);
      }

      im = getMeterInteractionMatrix(pts);
      return computeTwist(error_mat, im, optical_frame_);
    }

    // override function to support PoseStamped
    visual_servo::VisualServoTwist IBVSTwist(std::vector<PoseStamped> desired, std::vector<PoseStamped> pts)
    {
      cv::Mat error_mat;
      cv::Mat im;

      // return 0 twist if number of features does not equal to that of desired features
      if (pts.size() != desired.size())
        return visual_servo::VisualServoTwist();
      //return cv::Mat::zeros(6, 1, CV_32F);

      // for all three features,
      for (unsigned int i = 0; i < desired.size(); i++)
      {
        cv::Mat error = cv::Mat::zeros(2, 1, CV_32F);
        error.at<float>(0,0) = (pts.at(i)).pose.position.x -
          (desired.at(i)).pose.position.x;
        error.at<float>(1,0) = (pts.at(i)).pose.position.y -
          (desired.at(i)).pose.position.y;
        error_mat.push_back(error);
      }

      im = getMeterInteractionMatrix(pts);
      std::string of = pts.at(0).header.frame_id;
      return computeTwist(error_mat, im, of);
    }

    visual_servo::VisualServoTwist computeTwist(cv::Mat error_mat, cv::Mat im, std::string optical_frame)
    {
      // if we can't compute interaction matrix, just make all twists 0
      if (countNonZero(im) == 0) return visual_servo::VisualServoTwist();

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

          if (!desired_jacobian_set_)
            return visual_servo::VisualServoTwist();

          cv::Mat temp = desired_jacobian_ + im;
          iim = 0.5 * pseudoInverse(temp);
      }

      // Gain Matrix K (different from camera intrinsics)
      cv::Mat gain = cv::Mat::eye(6,6, CV_32F);

      // K x IIM x ERROR = TWIST
      cv::Mat temp = gain*iim*error_mat;

      ros::Time now = ros::Time(0);
      listener_.lookupTransform(workspace_frame_, optical_frame,  now, transform);

      // have to transform twist in camera frame (openni_rgb_optical_frame) to torso frame (torso_lift_link)
      tf::Vector3 twist_rot(temp.at<float>(3), temp.at<float>(4), temp.at<float>(5));
      tf::Vector3 twist_vel(temp.at<float>(0), temp.at<float>(1), temp.at<float>(2));  

      // twist transformation from optical frame to workspace frame
      tf::Vector3 out_rot = transform.getBasis() * twist_rot;
      tf::Vector3 out_vel = transform.getBasis() * twist_vel + transform.getOrigin().cross(out_rot);

      // multiple the velocity and rotation by gain defined in the parameter
      visual_servo::VisualServoTwist srv;
      srv.request.twist.twist.linear.x  = out_vel.x()*gain_vel_;
      srv.request.twist.twist.linear.y  = out_vel.y()*gain_vel_;
      srv.request.twist.twist.linear.z  = out_vel.z()*gain_vel_;
      srv.request.twist.twist.angular.x = out_rot.x()*gain_rot_;
      srv.request.twist.twist.angular.y = out_rot.y()*gain_rot_;
      srv.request.twist.twist.angular.z = out_rot.z()*gain_rot_;
      return srv;
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
    cv::Mat getMeterInteractionMatrix(std::vector<PoseStamped> &pts)
    {
      int size = (int)pts.size();
      // interaction matrix, image jacobian
      cv::Mat L = cv::Mat::zeros(size*2, 6,CV_32F);
      //if (pts.size() != 3) 
      //  return L;
      for (int i = 0; i < size; i++) {
        PoseStamped ps = pts.at(i);
        float x = ps.pose.position.x;
        float y = ps.pose.position.y;
        float Z = ps.pose.position.z;
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

    /**************************** 
     * Projection & Conversions
     ****************************/
    /**
     * transforms a point in pixels to meter using the inverse of 
     * image intrinsic K
     * @param in a point to be transformed
     * @return returns meter value of the point in cv::Mat
     */ 
    cv::Mat projectImagePointToPoint(cv::Point in) 
    {
      if (K.rows == 0 || K.cols == 0) {

        // Camera intrinsic matrix
        K  = cv::Mat(cv::Size(3,3), CV_64F, &(cam_info_.K));
        K.convertTo(K, CV_32F);
      }

      cv::Mat k_inv = K.inv();

      cv::Mat mIn  = cv::Mat(3,1,CV_32F);
      mIn.at<float>(0,0) = in.x; 
      mIn.at<float>(1,0) = in.y; 
      mIn.at<float>(2,0) = 1; 
      return k_inv * mIn;
    }

    /**
     * transforms a cv::Mat in pixels to meter using the inverse 
     * Image Intrinsic K
     * @param in  cv::Mat input to be transformed
     * @return    returns meter in cv::Mat
     */ 
    cv::Mat projectImageMatToPoint(cv::Mat in)
    {
      cv::Point p(in.at<float>(0,0), in.at<float>(1,0));
      return projectImagePointToPoint(p);
    }
    cv::Point projectPointIntoImage(pcl::PointXYZ cur_point_pcl,
        std::string point_frame, std::string target_frame, shared_ptr<tf::TransformListener> tf_)
    {
      PointStamped cur_point;
      cur_point.header.frame_id = point_frame;
      cur_point.point.x = cur_point_pcl.x;
      cur_point.point.y = cur_point_pcl.y;
      cur_point.point.z = cur_point_pcl.z;
      return projectPointIntoImage(cur_point, target_frame, tf_);
    }

    cv::Point projectPointIntoImage(PointStamped cur_point,
        std::string target_frame, shared_ptr<tf::TransformListener> tf_)
    {
      if (K.rows == 0 || K.cols == 0) {
        // Camera intrinsic matrix
        K  = cv::Mat(cv::Size(3,3), CV_64F, &(cam_info_.K));
        K.convertTo(K, CV_32F);
      }
      cv::Point img_loc;
      try
      {
        // Transform point into the camera frame
        PointStamped image_frame_loc_m;
        tf_->transformPoint(target_frame, cur_point, image_frame_loc_m);
        // Project point onto the image
        img_loc.x = static_cast<int>((K.at<float>(0,0)*image_frame_loc_m.point.x +
              K.at<float>(0,2)*image_frame_loc_m.point.z) /
            image_frame_loc_m.point.z);
        img_loc.y = static_cast<int>((K.at<float>(1,1)*image_frame_loc_m.point.y +
              K.at<float>(1,2)*image_frame_loc_m.point.z) /
            image_frame_loc_m.point.z);
      }
      catch (tf::TransformException e)
      {
        ROS_ERROR_STREAM(e.what());
      }
      return img_loc;
    }

    float getZValue(cv::Mat depth_frame, int x, int y)
    {
      int window_size = 3;
      float value = 0;
      int size = 0; 
      for (int i = 0; i < window_size; i++) 
      {
        for (int j = 0; j < window_size; j++) 
        {
          // depth camera has x and y flipped. depth_frame.at(y,x)
          float temp = depth_frame.at<float>(y-(int)(window_size/2)+j, x-(int)(window_size/2)+i);
          // printf("[%d %d] %f\n", x-(int)(window_size/2)+i, y-(int)(window_size/2)+j, temp);
          if (!isnan(temp) && temp > 0 && temp < 2.0) 
          {
            size++;
            value += temp;
          }
          else
          {
          }
        }
      }
      if (size == 0)
        return -1;
      return value/size;
    }

    // DEBUG PURPOSE
    void printMatrix(cv::Mat_<double> in)
    {
      for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
          printf("%+.5f\t", in(i,j)); 
        }
        printf("\n");
      }
    }
};
