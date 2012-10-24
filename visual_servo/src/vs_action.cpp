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
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/Point.h>
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
#include <tf/transform_broadcaster.h>

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

// Others
#include <actionlib/server/simple_action_server.h>
#include <visual_servo/VisualServoAction.h>
#include <visual_servo/VisualServoTwist.h>
#include <visual_servo/VisualServoPose.h>
#include "visual_servo.cpp"
#include "gripper_tape.cpp"

// floating point mod
#define fmod(a,b) a - (float)((int)(a/b)*b)

#define NO_TAPE 2
#define ESTIMATED 1
#define MEASURED 0

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,sensor_msgs::PointCloud2> MySyncPolicy;
typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;

using boost::shared_ptr;
using geometry_msgs::TwistStamped;
using geometry_msgs::PoseStamped;
using geometry_msgs::QuaternionStamped;
using geometry_msgs::Point;
using visual_servo::VisualServoTwist;
using visual_servo::VisualServoPose;

class VisualServoAction
{
  public:
    VisualServoAction(ros::NodeHandle &n, std::string which_arm) :
      n_(n),
      as_(n_, "vsaction", false),
      action_name_("vsaction"),
      n_private_("~"),
      image_sub_(n_, "color_image_topic", 1),
      depth_sub_(n_, "depth_image_topic", 1),
      cloud_sub_(n_, "point_cloud_topic", 1),
      sync_(MySyncPolicy(15), image_sub_, depth_sub_, cloud_sub_),
      it_(n_), tf_(), br_(), camera_initialized_(false)
  {
    vs_ = shared_ptr<VisualServo>(new VisualServo(JACOBIAN_TYPE_PSEUDO));
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());
    br_ = shared_ptr<tf::TransformBroadcaster>(new tf::TransformBroadcaster());
    which_arm_ = which_arm;

    n_private_.param("display_wait_ms", display_wait_ms_, 3);

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
    if (which_arm_.compare("l") == 0)
      tool_frame_ = left_tool_frame_;
    else
      tool_frame_ = right_tool_frame_;
std::string default_task_frame = "/vs_goal_frame";
    n_private_.param("task_frame", task_frame_, default_task_frame);

    // color segmentation parameters
    n_private_.param("target_hue_value", target_hue_value_, 350);
    n_private_.param("target_hue_threshold", target_hue_threshold_, 48);
    n_private_.param("gripper_tape_hue_value", gripper_tape_hue_value_, 200);
    n_private_.param("gripper_tape_hue_threshold", gripper_tape_hue_threshold_, 50);
    n_private_.param("default_sat_bot_value", default_sat_bot_value_, 20);
    n_private_.param("default_sat_top_value", default_sat_top_value_, 100);
    n_private_.param("default_val_value", default_val_value_, 100);
    n_private_.param("min_contour_size", min_contour_size_, 10.0);

    // others
    n_private_.param("jacobian_type", jacobian_type_, JACOBIAN_TYPE_INV);
    n_private_.param("vs_err_term_thres", vs_err_term_threshold_, 0.00075);
    n_private_.param("pose_servo_z_offset", pose_servo_z_offset_, 0.045);
    n_private_.param("place_z_velocity", place_z_velocity_, -0.025);
    n_private_.param("gripper_tape1_offset_x", tape1_offset_x_, 0.02);
    n_private_.param("gripper_tape1_offset_y", tape1_offset_y_, -0.025);
    n_private_.param("gripper_tape1_offset_z", tape1_offset_z_, 0.07);

    n_private_.param("max_exec_time", max_exec_time_, 20.0);

    // Setup ros node connections
    sync_.registerCallback(&VisualServoAction::sensorCallback, this);
    v_client_ = n_.serviceClient<visual_servo::VisualServoTwist>("/vs_twist");

    gripper_tape_ = GripperTape();

    as_.registerGoalCallback(boost::bind(&VisualServoAction::goalCB, this));
    ROS_DEBUG("[vsaction] Initialization 0: Node init & Register Callback Done");
    as_.start();
    ROS_INFO("[vsaction] \e[0;34minit done. Action for \e[1;34m[%s]\e[0;34m-arm started", which_arm_.c_str());
  }

    // destructor
    ~VisualServoAction()
    {
    }

    void updateGoalTransform(PoseStamped p)
    {
      tf::Transform tr;
      tr.setOrigin(tf::Vector3(p.pose.position.x, p.pose.position.y, p.pose.position.z));
      tr.setRotation(tf::Quaternion(p.pose.orientation.x, p.pose.orientation.y,
            p.pose.orientation.z, p.pose.orientation.w));
      br_->sendTransform(tf::StampedTransform(tr, ros::Time::now(), workspace_frame_, task_frame_));
    }

    void goalCB()
    {
      // goal pose given is the pose in tool frame
      tcp_goal_p_ = as_.acceptNewGoal()->pose;
      ROS_INFO("[vsaction] \e[1;34mGoal Accepted: p[%.3f %.3f %.3f]a[%.3f %.3f %.3f]", tcp_goal_p_.pose.position.x, tcp_goal_p_.pose.position.y, tcp_goal_p_.pose.position.z,
tcp_goal_p_.pose.orientation.x, tcp_goal_p_.pose.orientation.y, tcp_goal_p_.pose.orientation.z
      );

      // parameter resets
      max_no_tape_ = 0;
      setTimer(max_exec_time_);

      return;
    }

    /**
     * Called when Kinect information is avaiable. Refresh rate of about 30Hz 
     */
    void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg, 
        const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      // Store camera information only once
      if (!camera_initialized_)
      {
        cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic_, n_, ros::Duration(3.0));
        camera_initialized_ = true;
        vs_->setCamInfo(cam_info_);
        ROS_INFO("[vsaction]Initialization: Camera Info Done");
      }

      if (!as_.isActive())
        return;

      cv::Mat color_frame, depth_frame;
      cv_bridge::CvImagePtr color_cv_ptr = cv_bridge::toCvCopy(img_msg);
      cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg);

      color_frame = color_cv_ptr->image;
      depth_frame = depth_cv_ptr->image;

      XYZPointCloud cloud; 
      pcl::fromROSMsg(*cloud_msg, cloud);
      tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
          cloud.header.stamp, ros::Duration(0.9));
      pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);
      cur_camera_header_ = img_msg->header;
      cur_color_frame_ = color_frame;
      cur_orig_color_frame_ = color_frame.clone();
      cur_depth_frame_ = depth_frame.clone();
      cur_point_cloud_ = cloud;

      // tape features are updated (can be interpolated too)
      int u = updateGripperFeatures();


      // exit before computing bad vs values
      if (u == NO_TAPE)
      {
        // if we couldn't find no tape for 60 frames (about 2 seconds)
        if (max_no_tape_++ > 10)
        {
          ROS_WARN("Cannot find the end-effector. Aborting");
          as_.setAborted(result_, "no_tape");
        }
        return;
      }

      try
      {
        updateGoalTransform(tcp_goal_p_);
        // goal_p_ = gripper_tape_.getTaskTape0Pose(tf_, task_frame_, workspace_frame_);
        std::vector<VSXYZ> desired_vsxyz = vs_->Point3DToVSXYZ(gripper_tape_.getTaskTapePose(tf_, task_frame_, workspace_frame_), tf_);
        cur_goal_ = desired_vsxyz;
        goal_p_ = vs_->VSXYZToPoseStamped(desired_vsxyz.front());
      }
      catch (tf::TransformException ex)
      {
      }
      /*
      // Draw Features
      for (unsigned int i = 0; i < tape_features_.size(); i++)
      {
        cv::Point p = tape_features_.at(i).image;
        cv::circle(cur_orig_color_frame_, p, 2, cv::Scalar(100*i, 0, 110*(2-i)), 2);
      }
      for (unsigned int i = 0; i < cur_goal_.size(); i++)
      {
        cv::Point p = cur_goal_.at(i).image;
        cv::putText(cur_orig_color_frame_, "x", p, 2, 0.75, cv::Scalar(100*i, 0, 110*(2-i), 1));
      }
      cv::imshow("in", cur_orig_color_frame_); 
      cv::waitKey(5);
      */

      ROS_INFO("[vsaction][g:%.3f,%.3f,%.3f][t:%.3f,%.3f,%.3f]", goal_p_.pose.position.x,
          goal_p_.pose.position.y,goal_p_.pose.position.z,
          tape_features_p_.pose.position.x,
          tape_features_p_.pose.position.y,
          tape_features_p_.pose.position.z);

      // get the VS value 
      std::vector<PoseStamped> goals, feats;
      goals.push_back(goal_p_);
      feats.push_back(tape_features_p_);

      // setting arm_controller values
      visual_servo::VisualServoTwist v_srv = vs_->getTwist(goals,feats);


      if  (isnan(v_srv.request.twist.twist.linear.z) || isnan(v_srv.request.twist.twist.linear.x))
      {
        v_srv.request.twist.twist.linear.x = 0;
        v_srv.request.twist.twist.linear.y = 0;
        v_srv.request.twist.twist.linear.z = 0;
      }
      // we can align x and y first then come down
      v_srv.request.twist.twist.linear.z /= 2;
      // zero angular velocity for now
      v_srv.request.twist.twist.angular.x = 0;
      v_srv.request.twist.twist.angular.y = 0;
      v_srv.request.twist.twist.angular.z = 0;
      v_srv.request.twist.header.frame_id = workspace_frame_;


      v_srv.request.arm = which_arm_;
      //float err = getError(goal_p_, tape_features_p_);
      //v_srv.request.error = err;
      float err = v_srv.request.error;

      // setting action values
      feedback_.error = err;
      feedback_.ee = tape_features_p_;
      feedback_.twist = v_srv.request.twist;
      feedback_.which_arm = which_arm_;
      result_.error = err;

      if (v_client_.call(v_srv)){}
      as_.publishFeedback(feedback_);

      // termination condition
      if (v_srv.request.error < vs_err_term_threshold_)
      {
        ROS_INFO("\e[1;31mSuccess! Error Estimated: %.7f", v_srv.request.error);
        as_.setSucceeded(result_);
        sendZeroVelocity();
      }
      else if (isExpired())
      {
        ROS_WARN("Failed to go to the goal in time given. Aborting");
        as_.setAborted(result_, "timeout");
      }
    }

    void setTimer(float time)
    {
      ros::Time timer = ros::Time::now() + ros::Duration(time);
      timer_ = timer;
    }
    bool isExpired()
    {
      ros::Duration d = ros::Time::now() - timer_;
      if (d.toSec() > 0)
      {
        // set alarm to zero to be safe
        setTimer(0);
        return true;
      }
      return false; 
    }

    // Detect Tapes on Gripepr and update its position
    int updateGripperFeatures()
    {
      int ret = MEASURED;
      int default_tape_num = 3;

      PoseStamped p;
      p.header.frame_id = tool_frame_;
      p.pose.orientation.w = 1;
      tf_->transformPose(workspace_frame_, p, p);
      Point fkpp = p.pose.position;

      //////////////////////
      // Hand 
      // get all the blues 
      cv::Mat tape_mask = colorSegment(cur_color_frame_.clone(), gripper_tape_hue_value_, gripper_tape_hue_threshold_);

      // make it clearer with morphology
      cv::Mat element_b = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
      cv::morphologyEx(tape_mask, tape_mask, cv::MORPH_OPEN, element_b);

      // find the three largest blues
      std::vector<cv::Moments> ms = findMoments(tape_mask, cur_color_frame_); 

      // order the blue tapes
      std::vector<cv::Point> pts = getMomentCoordinates(ms);

      // convert the features into proper form 
      tape_features_ = vs_->CVPointToVSXYZ(cur_point_cloud_, cur_depth_frame_, pts);
      if ((int)tape_features_.size() != default_tape_num)
      {
        if ((int)v_fk_diff_.size() != default_tape_num)
        {
          ROS_WARN("[vsaction]Cannot find tape in image and do not have enough historical data to interpolate");
          tape_features_p_ = PoseStamped();
          return NO_TAPE;
        }
        ret = ESTIMATED;
        std::vector<pcl::PointXYZ> estimated_pose;
        estimated_pose.resize(default_tape_num);
        tape_features_.resize(default_tape_num);
        for (unsigned int i = 0; i < v_fk_diff_.size(); i++)
        {
          estimated_pose.at(i).x = v_fk_diff_.at(i).x + fkpp.x;
          estimated_pose.at(i).y = v_fk_diff_.at(i).y + fkpp.y;
          estimated_pose.at(i).z = v_fk_diff_.at(i).z + fkpp.z;
        }
        tape_features_ = vs_->Point3DToVSXYZ(estimated_pose, tf_);
      }
      tape_features_p_ = vs_->VSXYZToPoseStamped(tape_features_.front());
      // use forward kinematics to get the orientation
      QuaternionStamped q;
      q.quaternion.w = 1;
      q.header.frame_id = tool_frame_;
      tf_->transformQuaternion(workspace_frame_, q, q);
      tape_features_p_.pose.orientation = q.quaternion;

      pcl::PointXYZ tape0 = tape_features_.at(0).workspace;
      pcl::PointXYZ tape1 = tape_features_.at(1).workspace;
      pcl::PointXYZ tape2 = tape_features_.at(2).workspace;
      // gripper_tape_.setTapeRelLoc(tape0, tape1, tape2);
      std::vector<PoseStamped> tape_locs;
      tape_locs.push_back(GripperTape::formPoseStamped(workspace_frame_, tape0, tape_features_p_.pose.orientation));
      tape_locs.push_back(GripperTape::formPoseStamped(workspace_frame_, tape1, tape_features_p_.pose.orientation));
      tape_locs.push_back(GripperTape::formPoseStamped(workspace_frame_, tape2, tape_features_p_.pose.orientation));
      gripper_tape_.setTapeLoc(tf_, tool_frame_, tape_locs);


      // we aren't going to let controllers rotate at all when occluded;
      // vision vs. forward kinematics
      if (ret == MEASURED)
      {
        if (v_fk_diff_.size() == 0)
          v_fk_diff_.resize(default_tape_num); // initialize in case it isn't
        for (unsigned int i = 0 ; i < tape_features_.size() && i < v_fk_diff_.size(); i++)
        {
          v_fk_diff_.at(i).x = tape_features_.at(i).workspace.x - fkpp.x;
          v_fk_diff_.at(i).y = tape_features_.at(i).workspace.y - fkpp.y;
          v_fk_diff_.at(i).z = tape_features_.at(i).workspace.z - fkpp.z;
        }
      }

      return ret;
    }

    float getError(PoseStamped a, PoseStamped b)
    {
      ROS_DEBUG("[vsaction] getError: [%.3f %.3f %.3f] vs [%.3f %.3f %.3f]",
      a.pose.position.x, a.pose.position.y, a.pose.position.z,
      b.pose.position.x, b.pose.position.y, b.pose.position.z);
      float e(0.0);
      /*
      e += pow(a.pose.orientation.x - b.pose.orientation.x, 2);
      e += pow(a.pose.orientation.y - b.pose.orientation.y, 2);
      e += pow(a.pose.orientation.z - b.pose.orientation.z, 2);
      e += pow(a.pose.orientation.w - b.pose.orientation.w, 2);
      e /= 2;
      */
      e += pow(a.pose.position.x - b.pose.position.x, 2);
      e += pow(a.pose.position.y - b.pose.position.y, 2);
      e += pow(a.pose.position.z - b.pose.position.z, 2);
      return e;
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

      return wm;
    }

    /**********************
     * HELPER METHODS
     **********************/

    bool sendZeroVelocity()
    {
      // assume everything is init to zero
      visual_servo::VisualServoTwist v_srv;

      // need this to mvoe the arm
      v_srv.request.error = -1;
      v_srv.request.arm = which_arm_;
      return v_client_.call(v_srv);
    }

    visual_servo::VisualServoPose formPoseService(float px, float py, float pz)
    {

      visual_servo::VisualServoPose p = VisualServoMsg::createPoseMsg(px, py, pz, -0.4582, 0, 0.8889, 0);
      p.request.arm = which_arm_;
      return p;
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

  protected:
    ros::NodeHandle n_;
    actionlib::SimpleActionServer<visual_servo::VisualServoAction> as_;
    std::string action_name_;
    shared_ptr<VisualServo> vs_;

    std::string which_arm_;
    ros::Time timer_;
    double max_exec_time_;
    unsigned int max_no_tape_;

    visual_servo::VisualServoFeedback feedback_;
    visual_servo::VisualServoResult result_;

    ros::NodeHandle n_private_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Synchronizer<MySyncPolicy> sync_;
    image_transport::ImageTransport it_;
    sensor_msgs::CameraInfo cam_info_;
    shared_ptr<tf::TransformListener> tf_;
    shared_ptr<tf::TransformBroadcaster> br_;

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

    // segmenting
    int target_hue_value_;
    int target_hue_threshold_;
    int gripper_tape_hue_value_;
    int gripper_tape_hue_threshold_;
    int default_sat_bot_value_;
    int default_sat_top_value_;
    int default_val_value_;
    double min_contour_size_;

    // Other params
    int jacobian_type_;
    double vs_err_term_threshold_;
    double pose_servo_z_offset_;
    double place_z_velocity_;
    double tape1_offset_x_;
    double tape1_offset_y_;
    double tape1_offset_z_;
    ros::ServiceClient v_client_;

    // desired location/current gripper location
    cv::Mat desired_jacobian_;
    std::vector<VSXYZ> cur_goal_;
    std::vector<VSXYZ> goal_;
    PoseStamped goal_p_;
    PoseStamped tcp_goal_p_;
    std::vector<VSXYZ> tape_features_;
    PoseStamped tape_features_p_;
    VSXYZ desired_;
    cv::Mat K;
    std::vector<pcl::PointXYZ> v_fk_diff_;
    bool is_gripper_initialized_;
    bool gripper_pose_estimated_;
    GripperTape gripper_tape_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vsaction");

  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
  //my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Debug]);
  ros::NodeHandle n;
  std::string which_arm = argv[1];
  VisualServoAction vsa(n, which_arm);
  ros::spin();
  return 0;
}
