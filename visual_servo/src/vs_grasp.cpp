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

// Others
#include <visual_servo/VisualServoTwist.h>
#include <visual_servo/VisualServoPose.h>
#include "visual_servo.cpp"

#define DEBUG_MODE 0

#define INIT        0
#define POSE_CONTR  1
#define VS_CONTR    2
#define TERM        3

#define fmod(a,b) a - (float)((int)(a/b)*b)

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,sensor_msgs::PointCloud2> MySyncPolicy;
typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;
/*
typedef struct {
  // pixels
  cv::Point image;
  // meters, note that xyz for this is different from the other one
  pcl::PointXYZ camera;
  // the 3d location in workspace frame
  pcl::PointXYZ workspace;
  pcl::PointXYZ workspace_angular;
} VSXYZ;
*/
using boost::shared_ptr;
using geometry_msgs::TwistStamped;
using geometry_msgs::PoseStamped;
using visual_servo::VisualServoTwist;
using visual_servo::VisualServoPose;

/**
 * We are taping robot hand with three blue painter's tape 
 * and use those three features to do the image-based visual servoing
 * Uses Kinect as image sensor and provide a twist computation service
 */
class VisualServoNode
{
public:
  VisualServoNode(ros::NodeHandle &n) :
  n_(n), n_private_("~"),
  image_sub_(n, "color_image_topic", 1),
  depth_sub_(n, "depth_image_topic", 1),
  cloud_sub_(n, "point_cloud_topic", 1),
  sync_(MySyncPolicy(15), image_sub_, depth_sub_, cloud_sub_),
  it_(n), tf_(), have_depth_data_(false), camera_initialized_(false),
  desire_points_initialized_(false), PHASE(INIT) 
  {
    vs_ = shared_ptr<VisualServo>(new VisualServo(JACOBIAN_TYPE_PSEUDO));
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());
    n_private_.param("display_wait_ms", display_wait_ms_, 3);
    std::string default_optical_frame = "/head_mount_kinect_rgb_optical_frame";
    n_private_.param("optical_frame", optical_frame_, default_optical_frame);
 
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_, default_workspace_frame);
    n_private_.param("num_downsamples", num_downsamples_, 2);
    std::string cam_info_topic_def = "/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_, cam_info_topic_def);
    
    n_private_.param("crop_min_x", crop_min_x_, 0);
    n_private_.param("crop_max_x", crop_max_x_, 640);
    n_private_.param("crop_min_y", crop_min_y_, 0);
    n_private_.param("crop_max_y", crop_max_y_, 480);
    n_private_.param("min_workspace_x", min_workspace_x_, -1.0);
    n_private_.param("min_workspace_y", min_workspace_y_, -1.2);
    n_private_.param("min_workspace_z", min_workspace_z_, -0.8);
    n_private_.param("max_workspace_x", max_workspace_x_, 1.75);
    n_private_.param("max_workspace_y", max_workspace_y_, 1.2);
    n_private_.param("max_workspace_z", max_workspace_z_, 0.6);
    
    // color segmentation parameters
    n_private_.param("target_hue_value", target_hue_value_, 10);
    n_private_.param("target_hue_threshold", target_hue_threshold_, 20);
    n_private_.param("tape_hue_value", tape_hue_value_, 180);
    n_private_.param("tape_hue_threshold", tape_hue_threshold_, 50);
    n_private_.param("default_sat_bot_value", default_sat_bot_value_, 40);
    n_private_.param("default_sat_top_value", default_sat_top_value_, 40);
    n_private_.param("default_val_value", default_val_value_, 200);
    n_private_.param("min_contour_size", min_contour_size_, 10.0);
    
    // others
    n_private_.param("gain_vel", gain_vel_, 1.0);
    n_private_.param("gain_rot", gain_rot_, 1.0);
    n_private_.param("jacobian_type", jacobian_type_, JACOBIAN_TYPE_INV);
    
    n_private_.param("term_threshold", term_threshold_, 0.05);
  
    // Setup ros node connections
    sync_.registerCallback(&VisualServoNode::sensorCallback, this);
  }

  /**
   * Called when Kinect information is avaiable. Refresh rate of about 100Hz 
   */
  void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg, 
                      const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    // Store camera information only once
    if (!camera_initialized_)
    {
      cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic_, n_, ros::Duration(2.0));
      camera_initialized_ = true;
    }

    // Preparing the image
    //cv::Mat color_frame(bridge_.imgMsgToCv(img_msg));
    //cv::Mat depth_frame(bridge_.imgMsgToCv(depth_msg));
    cv::Mat color_frame, depth_frame;
    cv_bridge::CvImagePtr color_cv_ptr = cv_bridge::toCvCopy(img_msg);
    cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg);
    
    color_frame = color_cv_ptr->image;
    depth_frame = depth_cv_ptr->image;

    // Swap kinect color channel order
    cv::cvtColor(color_frame, color_frame, CV_RGB2BGR);

    XYZPointCloud cloud; 
    pcl::fromROSMsg(*cloud_msg, cloud);
    tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
        cloud.header.stamp, ros::Duration(0.5));
    pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);
    prev_camera_header_ = cur_camera_header_;
    cur_camera_header_ = img_msg->header;

    cv::Mat workspace_mask(color_frame.rows, color_frame.cols, CV_8UC1,
                           cv::Scalar(255));

    // Black out pixels in color and depth images outside of workspace
    // As well as outside of the crop window
    for (int r = 0; r < color_frame.rows; ++r)
    {
      uchar* workspace_row = workspace_mask.ptr<uchar>(r);
      for (int c = 0; c < color_frame.cols; ++c)
      {
        // NOTE: Cloud is accessed by at(column, row)
        pcl::PointXYZ cur_pt = cloud.at(c, r);
        if (cur_pt.x < min_workspace_x_ || cur_pt.x > max_workspace_x_ ||
            cur_pt.y < min_workspace_y_ || cur_pt.y > max_workspace_y_ ||
            cur_pt.z < min_workspace_z_ || cur_pt.z > max_workspace_z_ ||
            r < crop_min_y_ || c < crop_min_x_ || r > crop_max_y_ ||
            c > crop_max_x_ )
        {
          workspace_row[c] = 0;
        }
      }
    }

    // focus only on the tabletop setting. do not care about anything far or too close
    color_frame.copyTo(cur_color_frame_, workspace_mask);
    cur_orig_color_frame_ = color_frame.clone();
    cur_depth_frame_ = depth_frame.clone();
    cur_point_cloud_ = cloud;


    switch(PHASE)
    {
      case INIT:
        {
          if (initializeDesired())
          {
            initializeService();
            PHASE++;
          }
        }
        break;

      case POSE_CONTR:
        {
          visual_servo::VisualServoPose p_srv;
          VSXYZ d = desired_.front();
          p_srv.request.p.header.stamp = ros::Time::now();
          p_srv.request.p.header.frame_id = "torso_lift_link";
          p_srv.request.p.pose.position.x = d.workspace.x;
          p_srv.request.p.pose.position.y = d.workspace.y;
          p_srv.request.p.pose.position.z = d.workspace.z + 0.5;

          if (p_client_.call(p_srv))
          {
            ros::Duration(2.0).sleep();
            // on success
            int code = p_srv.response.result;
            if (0 == code)
              PHASE++;
          }
          else
          {
          }
        }
        break;

      case VS_CONTR:
        {
          // compute the twist if everything is good to go
          visual_servo::VisualServoTwist v_srv = getTwist();

          // calling the service provider to move
          if (v_client_.call(v_srv))
          {
            // on success
          }
          else
          {
            // on failure
            // ROS_WARN("Service FAILED...");
          }
        }
        break;

      default:
        {
          ROS_INFO("Reached TERM state");
          ros::shutdown();
        }
        break;
    }
  }   
  
  bool initializeDesired()
  {
    
    cv::Mat mask_t = colorSegment(cur_orig_color_frame_.clone(), target_hue_value_, target_hue_threshold_);

    cv::Mat element_t = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
    //cv::morphologyEx(mask_t, mask_t, cv::MORPH_CLOSE, element_t);
    //cv::morphologyEx(mask_t, mask_t, cv::MORPH_OPEN, element_t);
    
    // eroding is good enough (since we are getting rid of false positives)
    cv::erode(mask_t, mask_t, element_t);

    // find the largest red
    cv::Mat in = mask_t;
    std::vector<std::vector<cv::Point> > contours; contours.clear();
    cv::findContours(in, contours, cv::RETR_CCOMP,CV_CHAIN_APPROX_NONE);

    /*
    std::vector<cv::Moments> moments; moments.clear();
    
    unsigned int bigInd = 0;
    for (unsigned int i = 0; i < contours.size(); i++) {
      cv::Moments m = cv::moments(contours[i]);
      if (m.m00 > min_contour_size_) {
        // first add the forth element
        moments.push_back(m);
        // find the smallest element of 4 and remove that
        if (moments.size() > max_num) {
          double small(moments.at(0).m00);
          unsigned int smallInd(0);
          for (unsigned int j = 1; j < moments.size(); j++){
            if (moments.at(j).m00 < small) {
              small = moments.at(j).m00;
              smallInd = j;
            }
            else bigInd = j;
          }
          moments.erase(moments.begin() + smallInd);
        }
      }
    }
    std::vector<cv::Moments> ms_t = moments;
    std::vector<cv::Point> ps = contours.at(bigInd);
    
    // impossible then
    if (ms_t.size() < 1 || ps.size() < 1)
    {
      ROS_WARN("No target Found");
      return srv;
    }
     */

    if (contours.size() < 1)
    {
      ROS_WARN("No target Found");
      return false;
    }

    unsigned int cont_max = contours.at(0).size();
    unsigned int cont_max_ind = 0;
    for (unsigned int i = 1; i < contours.size(); i++) 
    {
      if (contours.at(i).size() > cont_max)
      {
        cont_max = contours.at(i).size();
        cont_max_ind = i;
      }
    }
    
    std::vector<cv::Point> ps = contours.at(cont_max_ind);
    int max_y = 0; 
    int max_y_ind = 0;
    for (unsigned int i = 0; i < ps.size(); i++)
    {
      if (ps.at(i).y > max_y)
      {
        max_y = ps.at(i).y;
        max_y_ind = i;
      }
    }
    
    std::vector<cv::Point> pts_t; pts_t.clear();
    pts_t.push_back(cv::Point(ps.at(max_y_ind).x, max_y));
    
    // convert to proper internal struct 
    desired_ = PointToVSXYZ(cur_point_cloud_, cur_depth_frame_, pts_t);

    std::vector<VSXYZ> desired_vsxyz = getFeaturesFromXYZ(desired_.front());
    desired_locations_ = desired_vsxyz;

    // for jacobian avg, we need IM at desired location as well
    if (JACOBIAN_TYPE_AVG == jacobian_type_)
      return vs_->setDesiredInteractionMatrix(desired_vsxyz);
  
    return true;
  }

  void initializeService()
  {
    ROS_DEBUG("Hooking Up The Service");
    ros::NodeHandle n;
    v_client_ = n.serviceClient<visual_servo::VisualServoTwist>("vs_twist");
    p_client_ = n.serviceClient<visual_servo::VisualServoPose>("vs_pose");
  }

  std::vector<VSXYZ> getFeaturesFromXYZ(VSXYZ origin_xyz)
  {
    pcl::PointXYZ origin = origin_xyz.workspace;
    std::vector<pcl::PointXYZ> pts; pts.clear();

    origin.y += 0.025;
    origin.z += 0.20;
    pcl::PointXYZ two = origin;
    pcl::PointXYZ three = origin;
    two.y -= 0.05; 
    three.z += 0.05;
    
    pts.push_back(origin);
    pts.push_back(two);
    pts.push_back(three);
    return Point3DToVSXYZ(pts);
  }
  
  // Service method
  visual_servo::VisualServoTwist getTwist()
  {
    visual_servo::VisualServoTwist srv;
    //////////////////////
    // Hand 
    // get all the blues 
    cv::Mat tape_mask = colorSegment(cur_color_frame_.clone(), tape_hue_value_, 
        tape_hue_threshold_);
    
    // make it clearer with morphology
    cv::Mat element_b = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    cv::morphologyEx(tape_mask, tape_mask, cv::MORPH_OPEN, element_b);

    // find the three largest blues
    std::vector<cv::Moments> ms = findMoments(tape_mask, cur_color_frame_); 
    
    // order the blue tapes
    std::vector<cv::Point> pts = getMomentCoordinates(ms);
    
    // convert the features into proper form 
    std::vector<VSXYZ> features = PointToVSXYZ(cur_point_cloud_, cur_depth_frame_, pts);

#define DISPLAY 0
#ifdef DISPLAY
    // Draw the dots on image to be displayed
    for (unsigned int i = 0; i < desired_locations_.size(); i++)
    {
      cv::Point p = desired_locations_.at(i).image;
      cv::circle(cur_orig_color_frame_, p, 2, cv::Scalar(100*i, 0, 110*(2-i)), 2);
    }
    for (unsigned int i = 0; i < features.size(); i++)
    {
      cv::Point p = features.at(i).image;
      cv::circle(cur_orig_color_frame_, p, 2, cv::Scalar(100*i, 0, 110*(2-i)), 2);
    }
    cv::imshow("in", cur_orig_color_frame_); 
    cv::waitKey(display_wait_ms_);
    
    /* 
    for (unsigned int i = 0; i < desired_locations_.size(); i++)
    {
      printVSXYZ(desired_locations_.at(i));
    }
    for (unsigned int i = 0; i < features.size(); i++)
    {
      printVSXYZ(features.at(i));
    }
    
    printf("%+.5f\t%+.5f\t%+.5f\n", srv.request.twist.twist.linear.x, 
        srv.request.twist.twist.linear.y, srv.request.twist.twist.linear.z);
    */
#endif
    srv = vs_->computeTwist(desired_locations_, features);
    srv.request.error = getError(desired_locations_, features);
    return srv;
  }

  float getError(std::vector<VSXYZ> a, std::vector<VSXYZ> b)
  {
    float e(0.0);
    unsigned int size = a.size() <= b.size() ? a.size() : b.size();
    for (unsigned int i = 0; i < size; i++)
    {
      pcl::PointXYZ a_c= a.at(i).camera;
      pcl::PointXYZ b_c= b.at(i).camera;
      e += pow(a_c.x - b_c.x ,2) + pow(a_c.y - b_c.y ,2);
    }
    return e;
  }

  /**
   * Still in construction:
   * to fix the orientation of the wrist, we now take a look at how the hand is positioned
   * and use it to compute the desired positions.
   */
  std::vector<VSXYZ> setDesiredPosition()
  {
    std::vector<VSXYZ> desired; desired.clear();
    // Looking up the hand
    // Setting the Desired Location of the wrist
    // Desired location: center of the screen
    std::vector<pcl::PointXYZ> pts; pts.clear();
    pcl::PointXYZ origin = cur_point_cloud_.at(cur_color_frame_.cols/2, cur_color_frame_.rows/2);
    origin.z += 0.10;
    pcl::PointXYZ two = origin;
    pcl::PointXYZ three = origin;
    two.y -= 0.07; 
    three.x -= 0.06;
    
    pts.push_back(origin); pts.push_back(two); pts.push_back(three);
    return Point3DToVSXYZ(pts);
  }
  
  /************************************
   * PERCEPTION
   ************************************/

  /**
   * Take three biggest moments of specific color and returns 
   * the three biggest blobs or moments. This method assumes that
   * the features are in QR code like configuration
   * @param ms   All moments of color segmented
   * @return     returns vectors of cv::Point. Ordered in specific way 
   *             (1. top left, 2. top right, and 3. bottom left)
   **/
  std::vector<cv::Point> getMomentCoordinates(std::vector<cv::Moments> ms)
  {
    std::vector<cv::Point> ret;
    ret.clear();
    if (ms.size() == 3) { 
      double centroids[3][2];
      for (int i = 0; i < 3; i++) {
        cv::Moments m0 = ms.at(i);
        double x0, y0;
        x0 = m0.m10/m0.m00;
        y0 = m0.m01/m0.m00;
        centroids[i][0] = x0; 
        centroids[i][1] = y0; 
      }
      
      // find the top left corner using distance scheme
      cv::Mat vect = cv::Mat::zeros(3,2, CV_32F); 
      vect.at<float>(0,0) = centroids[0][0] - centroids[1][0];
      vect.at<float>(0,1) = centroids[0][1] - centroids[1][1];
      vect.at<float>(1,0) = centroids[0][0] - centroids[2][0];
      vect.at<float>(1,1) = centroids[0][1] - centroids[2][1];
      vect.at<float>(2,0) = centroids[1][0] - centroids[2][0];
      vect.at<float>(2,1) = centroids[1][1] - centroids[2][1];       
      
      double angle[3];
      angle[0] = abs(vect.row(0).dot(vect.row(1))); 
      angle[1] = abs(vect.row(0).dot(vect.row(2))); 
      angle[2] = abs(vect.row(1).dot(vect.row(2))); 
      
      // printMatrix(vect); 
      double min = angle[0]; 
      int one = 0;
      for (int i = 0; i < 3; i++)
      {
        // printf("[%d, %f]\n", i, angle[i]);
        if (angle[i] < min)
        {
          min = angle[i];
          one = i;
        }
      }
      
      // index of others depending on the index of the origin
      int a = one == 0 ? 1 : 0;
      int b = one == 2 ? 1 : 2; 
      // vectors of origin to a point
      double vX0, vY0, vX1, vY1, result;
      vX0 = centroids[a][0] - centroids[one][0];
      vY0 = centroids[a][1] - centroids[one][1];
      vX1 = centroids[b][0] - centroids[one][0];
      vY1 = centroids[b][1] - centroids[one][1];
      cv::Point pto(centroids[one][0], centroids[one][1]);
      cv::Point pta(centroids[a][0], centroids[a][1]);
      cv::Point ptb(centroids[b][0], centroids[b][1]);
      
      // cross-product: simplified assuming that z = 0 for both
      result = vX1*vY0 - vX0*vY1;
      ret.push_back(pto);
      if (result >= 0) {
        ret.push_back(ptb);
        ret.push_back(pta);
      }
      else {
        ret.push_back(pta);
        ret.push_back(ptb);
      }
    }
    return ret;
  } 
  
  /**
   * 
   * @param in  single channel image input
   * @param color_frame  need the original image for debugging and imshow
   * 
   * @return    returns ALL moment of specific color in the image
   **/
  std::vector<cv::Moments> findMoments(cv::Mat in, cv::Mat &color_frame, unsigned int max_num = 3) 
  {
    cv::Mat temp = in.clone();
    std::vector<std::vector<cv::Point> > contours; contours.clear();
    cv::findContours(temp, contours, cv::RETR_CCOMP,CV_CHAIN_APPROX_NONE);
    std::vector<cv::Moments> moments; moments.clear();
    
    for (unsigned int i = 0; i < contours.size(); i++) {
      cv::Moments m = cv::moments(contours[i]);
      if (m.m00 > min_contour_size_) {
        // first add the forth element
        moments.push_back(m);
        // find the smallest element of 4 and remove that
        if (moments.size() > max_num) {
          double small(moments.at(0).m00);
          unsigned int smallInd(0);
          for (unsigned int j = 1; j < moments.size(); j++){
            if (moments.at(j).m00 < small) {
              small = moments.at(j).m00;
              smallInd = j;
            }
          }
          moments.erase(moments.begin() + smallInd);
        }
      }
    }
    return moments;
  }
  
  cv::Mat colorSegment(cv::Mat color_frame, int hue, int threshold)
  {
    /*
     * Often value = 0 or 255 are very useless. 
     * The distance at those end points get very close and it is not useful
     * Same with saturation 0. Low saturation makes everything more gray scaled
     * So the default setting are below 
     */
    return colorSegment(color_frame, hue - threshold, hue + threshold,  
        default_sat_bot_value_, default_sat_top_value_, 40, default_val_value_);
  }
  
  /** 
   * Very Basic Color Segmentation done in HSV space
   * Takes in Hue value and threshold as input to compute the distance in color space
   *
   * @param color_frame   color input from image
   *
   * @return  mask from the color segmentation 
   */
  cv::Mat colorSegment(cv::Mat color_frame, int _hue_n, int _hue_p, int _sat_n, int _sat_p, int _value_n,  int _value_p)
  {
    cv::Mat temp (color_frame.clone());
    cv::cvtColor(temp, temp, CV_BGR2HSV);
    std::vector<cv::Mat> hsv;
    cv::split(temp, hsv);
   
    // so it can support hue near 0 & 360
    _hue_n = (_hue_n + 360);
    _hue_p = (_hue_p + 360);

    // masking out values that do not fall between the condition 
    cv::Mat wm(color_frame.rows, color_frame.cols, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < temp.rows; r++)
    {
      uchar* workspace_row = wm.ptr<uchar>(r);
      for (int c = 0; c < temp.cols; c++)
      {
        int hue     = 2*(int)hsv[0].at<uchar>(r, c) + 360;  
        float sat   = 0.392*(int)hsv[1].at<uchar>(r, c); // 0.392 = 100/255
        float value = 0.392*(int)hsv[2].at<uchar>(r, c);

        if (_hue_n < hue && hue < _hue_p)
          if (_sat_n < sat && sat < _sat_p)
            if (_value_n < value && value < _value_p)
              workspace_row[c] = 255;
      } 
    }
    
    /*
    // REMOVE
    printf("[hn=%d hp=%d]", _hue_n, _hue_p);
    int r = 0; int c = temp.cols-1;
    int hue     = 2*(int)hsv[0].at<uchar>(r, c);  
    float sat   = 0.392*(int)hsv[1].at<uchar>(r, c); // 0.392 = 100/255
    float value = 0.392*(int)hsv[2].at<uchar>(r, c);
    printf("[%d,%d][%d, %.1f, %.1f]\n", r, c, hue,sat,value);
    */

    // removing unwanted parts by applying mask to the original image
    return wm;
  }
 

  /**************************** 
   * Projection & Conversions
   ****************************/
  std::vector<VSXYZ> Point3DToVSXYZ(std::vector<pcl::PointXYZ> in)  
  {
    std::vector<VSXYZ> ret;
    for (unsigned int i = 0; i < in.size(); i++)
    {
      ret.push_back(convertFrom3DPointToVSXYZ(in.at(i)));
    }
    return ret;
  }

  /*
  cv::Mat pointXYZToMat(pcl::PointXYZ in)
  {
    cv::Mat ret = cv::Mat::ones(4,1,CV_32F);
    ret.at<float>(0,0) = in.x;
    ret.at<float>(1,0) = in.y;
    ret.at<float>(2,0) = in.z;
    return ret;
  }
  
  pcl::PointXYZ matToPointXYZ(cv::Mat in)
  {
    return pcl::PointXYZ(in.at<float>(0,0), in.at<float>(1,0),
    in.at<float>(2,0));
  }
  */

#ifdef SIMULATION
  // separate function for simulation since we can't utilize 
  // TF for rotation
  VSXYZ convertFrom3DPointToVSXYZ(pcl::PointXYZ in) 
  {
    // [X,Y,Z] -> [u,v] -> [x,y,z]
    VSXYZ ret;
    
    // 3D point in world frame
    pcl::PointXYZ in_c = in;

    // temporary to apply the transformation
    cv::Mat t = cv::Mat::ones(4,1, CV_32F);
    t.at<float>(0,0) = in_c.x;
    t.at<float>(1,0) = in_c.y;
    t.at<float>(2,0) = in_c.z;
    cv::Mat R = simulateGetRotationMatrix(sim_camera_x_, sim_camera_y_,
      sim_camera_z_, sim_camera_wx_, sim_camera_wy_, sim_camera_wz_);
    
    // apply the rotation
    t = R.inv() * t;
    in_c.x = t.at<float>(0,0);
    in_c.y = t.at<float>(1,0);
    in_c.z = t.at<float>(2,0);
    
    // really doesn't matter which frame they are in. ignored for now.
    cv::Point img = projectPointIntoImage(in_c, default_optical_frame, default_optical_frame);
    cv::Mat temp = projectImagePointToPoint(img);

    float depth = sqrt(pow(in_c.z,2) + pow(temp.at<float>(0,0),2) + pow(temp.at<float>(1,0),2));
    pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), depth);
    
    ret.image = img;
    // for simpler simulator, this will be the same
    ret.camera = _2d;
    ret.workspace= in;
    return ret;
  }
#else
  VSXYZ convertFrom3DPointToVSXYZ(pcl::PointXYZ in) 
  {
    // [X,Y,Z] -> [u,v] -> [x,y,z]
    VSXYZ ret;
    
    // 3D point in world frame
    pcl::PointXYZ in_c = in;
    cv::Point img = projectPointIntoImage(in_c, workspace_frame_, optical_frame_);
    cv::Mat temp = projectImagePointToPoint(img);

    float depth = sqrt(pow(in_c.z,2) + pow(temp.at<float>(0,0),2) + pow(temp.at<float>(1,0),2));
    pcl::PointXYZ _2d(temp.at<float>(0,0), temp.at<float>(1,0), depth);
    
    ret.image = img;
    // for simpler simulator, this will be the same
    ret.camera = _2d;
    ret.workspace= in;
    return ret;
  }
#endif

  std::vector<VSXYZ> PointToVSXYZ(XYZPointCloud cloud, cv::Mat depth_frame, std::vector<cv::Point> in) 
  {
    std::vector<VSXYZ> ret;
    for (unsigned int i = 0; i < in.size(); i++)
    {
      ret.push_back(convertFromPointToVSXYZ(cloud, depth_frame, in.at(i)));
    }
    return ret;
  }
  
  VSXYZ convertFromPointToVSXYZ(XYZPointCloud cloud, cv::Mat depth_frame, cv::Point in) 
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
  
  /** 
   * Transforms a point in Point Cloud to Image Frame (pixels)
   * @param cur_point_pcl  The point to be transformed in pcl::PointXYZ
   * @param point_frame    The frame that the PointCloud is in
   * @param target_frame   Image frame
   * @return               returns the pixel value of the PointCloud in image frame
   */
  cv::Point projectPointIntoImage(pcl::PointXYZ cur_point_pcl,
                                  std::string point_frame, std::string target_frame)
  {
    PointStamped cur_point;
    cur_point.header.frame_id = point_frame;
    cur_point.point.x = cur_point_pcl.x;
    cur_point.point.y = cur_point_pcl.y;
    cur_point.point.z = cur_point_pcl.z;
    return projectPointIntoImage(cur_point, target_frame);
  }
  
  cv::Point projectPointIntoImage(PointStamped cur_point,
                                  std::string target_frame)
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

  /**********************
   * HELPER METHODS
   **********************/
  void printMatrix(cv::Mat_<double> in)
  {
    for (int i = 0; i < in.rows; i++) {
      for (int j = 0; j < in.cols; j++) {
        printf("%+.5f\t", in(i,j)); 
      }
      printf("\n");
    }
  }
  
  void printVSXYZ(VSXYZ i)
  {
    printf("Im: %+.3d %+.3d\tCam: %+.3f %+.3f %+.3f\twork: %+.3f %+.3f %+.3f\n",\
     i.image.x, i.image.y, i.camera.x, i.camera.y, i.camera.z, i.workspace.x, i.workspace.y, i.workspace.z);
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
      }
    }
    if (size == 0)
      return -1;
    return value/size;
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
  image_transport::ImageTransport it_;
  sensor_msgs::CameraInfo cam_info_;
  shared_ptr<tf::TransformListener> tf_;
  cv::Mat cur_color_frame_;
  cv::Mat cur_orig_color_frame_;
  cv::Mat cur_depth_frame_;
  cv::Mat cur_workspace_mask_;
  std_msgs::Header cur_camera_header_;
  std_msgs::Header prev_camera_header_;
  XYZPointCloud cur_point_cloud_;

  bool have_depth_data_;
  int display_wait_ms_;
  int num_downsamples_;
  std::string workspace_frame_;
  std::string optical_frame_;
  bool camera_initialized_;
  bool desire_points_initialized_;
  std::string cam_info_topic_;
  int tracker_count_;

  // filtering 
  double min_workspace_x_;
  double max_workspace_x_;
  double min_workspace_y_;
  double max_workspace_y_;
  double min_workspace_z_;
  double max_workspace_z_;
  int crop_min_x_;
  int crop_max_x_;
  int crop_min_y_;
  int crop_max_y_;

  // segmenting
  int target_hue_value_;
  int target_hue_threshold_;
  int tape_hue_value_;
  int tape_hue_threshold_;
  int default_sat_bot_value_;
  int default_sat_top_value_;
  int default_val_value_;
  double min_contour_size_;

  // others    
  shared_ptr<VisualServo> vs_;
  int jacobian_type_;
  double gain_vel_;
  double gain_rot_;
  double term_threshold_;
  
  unsigned int PHASE;

  cv::Mat desired_jacobian_;
  std::vector<VSXYZ> desired_locations_;
  std::vector<VSXYZ> desired_;
  cv::Mat K;

  ros::ServiceClient v_client_;
  ros::ServiceClient p_client_;
};

int main(int argc, char ** argv)
{
  srand(time(NULL));
  ros::init(argc, argv, "visual_servo_node");
  
  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Debug]);
  
  ros::NodeHandle n;
  VisualServoNode vs_node(n);
  vs_node.spin();
}
