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

// PR2_GRIPPER_SENSOR_ACTION
#include <pr2_controllers_msgs/Pr2GripperCommandAction.h>
#include <pr2_gripper_sensor_msgs/PR2GripperGrabAction.h>
#include <pr2_gripper_sensor_msgs/PR2GripperEventDetectorAction.h>
#include <pr2_gripper_sensor_msgs/PR2GripperReleaseAction.h>
#include <actionlib/client/simple_action_client.h>

// Others
#include <visual_servo/VisualServoTwist.h>
#include <visual_servo/VisualServoPose.h>
#include <std_srvs/Empty.h>
#include "visual_servo.cpp"

#include <tabletop_pushing/point_cloud_segmentation.h>

#define DEBUG_MODE 0
#define VISUAL_SERVO_TYPE 0

#define PERCEPTION_COLOR_SEGMENTATION 0
#define PERCEPTION_POINT_CLOUD 1
#define PERCEPTION PERCEPTION_POINT_CLOUD
//#define PERCEPTION PERCEPTION_COLOR_SEGMENTATION

// statemachine constants
#define INIT          0
#define SETTLE        1
#define INIT_OBJS     2
#define INIT_HAND     3
#define INIT_DESIRED  4
#define POSE_CONTR    5
#define POSE_CONTR_2  6
#define VS_CONTR_1    7
#define VS_CONTR_2    8
#define GRAB          9
#define RELOCATE      10
#define DESCEND_INIT  11
#define DESCEND       12
#define RELEASE       13
#define FINISH        14
#define TERM          15

// floating point mod
#define fmod(a,b) a - (float)((int)(a/b)*b)

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,sensor_msgs::PointCloud2> MySyncPolicy;
typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;

typedef actionlib::SimpleActionClient<pr2_gripper_sensor_msgs::PR2GripperGrabAction> GrabClient;
typedef actionlib::SimpleActionClient<pr2_gripper_sensor_msgs::PR2GripperEventDetectorAction> EventDetectorClient;
typedef actionlib::SimpleActionClient<pr2_gripper_sensor_msgs::PR2GripperReleaseAction> ReleaseClient; 
typedef actionlib::SimpleActionClient<pr2_controllers_msgs::Pr2GripperCommandAction> GripperClient; 

using boost::shared_ptr;
using geometry_msgs::TwistStamped;
using geometry_msgs::PoseStamped;
using geometry_msgs::QuaternionStamped;
using geometry_msgs::Point;
using visual_servo::VisualServoTwist;
using visual_servo::VisualServoPose;

class GripperTape
{
  /**
   * Our gripper has three blue tapes used for gripper pose perception.
   * These tapes are stored as follows: position away from gripper tip center (between l and r)
   * Relative position of tape 1 and 2 from tape 0
   * This will be replaced with better feature perception.
   **/
  public:
    GripperTape()
    {
    }

    void setTapeRelLoc(pcl::PointXYZ tape0, pcl::PointXYZ tape1, pcl::PointXYZ tape2)
    {
      tape1_loc_.x = tape1.x - tape0.x;
      tape1_loc_.y = tape1.y - tape0.y;
      tape1_loc_.z = tape1.z - tape0.z;
      tape2_loc_.x = tape2.x - tape0.x;
      tape2_loc_.y = tape2.y - tape0.y;
      tape2_loc_.z = tape2.z - tape0.z;
    }

    // this is the offset from tip center to tape0
    void setOffset(pcl::PointXYZ offset)
    {
      tape0_loc_ = offset;
    }

    std::vector<pcl::PointXYZ> getTapePoseFromXYZ(pcl::PointXYZ orig)
    {
      std::vector<pcl::PointXYZ> pts; pts.clear();
      pcl::PointXYZ zero = addPointXYZ(orig, tape0_loc_);
      pcl::PointXYZ one = addPointXYZ(zero, tape1_loc_);
      pcl::PointXYZ two = addPointXYZ(zero, tape2_loc_);

      pts.push_back(zero);
      pts.push_back(one);
      pts.push_back(two);

      return pts;
    }

  private:
    pcl::PointXYZ tape0_loc_;
    pcl::PointXYZ tape1_loc_;
    pcl::PointXYZ tape2_loc_;

    pcl::PointXYZ addPointXYZ(pcl::PointXYZ a, pcl::PointXYZ b)
    {
      pcl::PointXYZ r;
      r.x = a.x + b.x;
      r.y = a.y + b.y;
      r.z = a.z + b.z;
      return r;
    }
};

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
      desire_points_initialized_(false), PHASE(INIT), is_gripper_initialized_(false), gripper_pose_estimated_(false),
      is_detected_(false), place_detection_(false)

  {
    vs_ = shared_ptr<VisualServo>(new VisualServo(JACOBIAN_TYPE_PSEUDO));
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());

    n_private_.param("display_wait_ms", display_wait_ms_, 3);

    std::string default_optical_frame = "/head_mount_kinect_rgb_optical_frame";
    n_private_.param("optical_frame", optical_frame_, default_optical_frame);
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_, default_workspace_frame);
    std::string cam_info_topic_def = "/kinect_head/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_, cam_info_topic_def);

    // color segmentation parameters
    n_private_.param("target_hue_value", target_hue_value_, 10);
    n_private_.param("target_hue_threshold", target_hue_threshold_, 20);
    n_private_.param("gripper_tape_hue_value", gripper_tape_hue_value_, 180);
    n_private_.param("gripper_tape_hue_threshold", gripper_tape_hue_threshold_, 50);
    n_private_.param("default_sat_bot_value", default_sat_bot_value_, 40);
    n_private_.param("default_sat_top_value", default_sat_top_value_, 40);
    n_private_.param("default_val_value", default_val_value_, 200);
    n_private_.param("min_contour_size", min_contour_size_, 10.0);

    // others
    n_private_.param("jacobian_type", jacobian_type_, JACOBIAN_TYPE_INV);
    n_private_.param("vs_err_term_thres", vs_err_term_threshold_, 0.001);
    n_private_.param("pose_servo_z_offset", pose_servo_z_offset_, 0.045);
    n_private_.param("place_z_velocity", place_z_velocity_, -0.025);
    n_private_.param("gripper_tape1_offset_x", tape1_offset_x_, 0.02);
    n_private_.param("gripper_tape1_offset_y", tape1_offset_y_, -0.025);
    n_private_.param("gripper_tape1_offset_z", tape1_offset_z_, 0.07);

    // Setup ros node connections
    sync_.registerCallback(&VisualServoNode::sensorCallback, this);

#ifdef EXPERIMENT
    chatter_pub_ = n_.advertise<std_msgs::String>("chatter", 100);
    alarm_ = ros::Time(0);
#endif

    gripper_tape_ = GripperTape();
    gripper_tape_.setOffset(pcl::PointXYZ(tape1_offset_x_, tape1_offset_y_, tape1_offset_z_));
    ROS_INFO("Initialization 0: Node init & Register Callback Done");
  }

  // destructor
  ~VisualServoNode()
  {
    delete gripper_client_;
    delete grab_client_;
    delete release_client_;
    delete detector_client_;
  }

    /**
     * Called when Kinect information is avaiable. Refresh rate of about 30Hz 
     */
    void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg, 
        const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {

      ros::Time start = ros::Time::now();

      // Store camera information only once
      if (!camera_initialized_)
      {
        cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic_, n_, ros::Duration(3.0));
        camera_initialized_ = true;
        vs_->setCamInfo(cam_info_);
        ROS_INFO("Initialization: Camera Info Done");
      }

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

      // need to profile this
      gripper_pose_estimated_ = updateGripperFeatures();
      executeStatemachine();

#define DISPLAY 1
#ifdef DISPLAY
      // show a pretty imshow
      setDisplay();
#endif
#ifdef EXPERIMENT
      // for profiling purpose
      ros::Time end = ros::Time::now();
      std::stringstream ss;
      ss << "Time it took is " << (end - start).toSec();
      std_msgs::String msg;
      msg.data = ss.str();
      chatter_pub_.publish(msg);
#endif
    }

    void executeStatemachine()
    {
      if (sleepNonblock())
      {
        temp_draw_.clear();
        switch(PHASE)
        {
          case INIT:
            {
              reset();
#ifndef PROFILE
              ROS_INFO("Initializing Services and Robot Configuration");
              initializeService();

              // try to initialize the arm
              std_srvs::Empty e;
              i_client_.call(e);

              // this is a blocking close
              close();
              float temp = getTipDistance();
              close_gripper_dist_ = temp;
              ROS_INFO("Close Gripper Tip distane: %.4f", temp);
              sleep(2);
              // gripper needs to be controlled from here
              open();
#endif
              if (temp != -1)
              {
                PHASE = INIT_OBJS;
                setSleepNonblock(5.0);
              }
            }
            break;
          case INIT_OBJS:
            {
              bool t = false;
              ROS_INFO("Phase Init Obj: Getting Objects without the arm");
#if PERCEPTION==PERCEPTION_COLOR_SEGMENTATION
              t = initializeDesired(desired_);
#elif PERCEPTION==PERCEPTION_POINT_CLOUD
              t = initializeDesired(po_);
#endif
              if (t)
              {
#if PERCEPTION==PERCEPTION_POINT_CLOUD
                ROS_INFO(">> Found %d objects", (int)(po_.size()));
#endif
                if (is_gripper_initialized_)
                  PHASE = INIT_DESIRED;
                else
                PHASE = SETTLE;
              }
              else
              {
                ROS_WARN("Failed. Retrying initialization in 2 seconds");
                setSleepNonblock(2.0);
              }

            }
            break;
          case SETTLE:
            {
              ROS_INFO("Phase Settle: Move Gripper to Init Position");
              // Move Hand to Some Preset Pose
              visual_servo::VisualServoPose p_srv = formPoseService(0.62, 0.05, -0.1);
              if (p_client_.call(p_srv))
              {
                setSleepNonblock(3.0);
                PHASE = INIT_HAND;
              }
              else
              {
                ROS_WARN("Failed to put the hand in initial configuration");
              }
            }
            break;
          case INIT_HAND:
            {
              ROS_INFO("Phase Init Hand: Remembering Blue Tape Positions");

              if (tape_features_.size() == 3)
              {
                pcl::PointXYZ tape0 = tape_features_.at(0).workspace;
                pcl::PointXYZ tape1 = tape_features_.at(1).workspace;
                pcl::PointXYZ tape2 = tape_features_.at(2).workspace;

                gripper_tape_.setTapeRelLoc(tape0, tape1, tape2);
                temp_draw_.push_back(tape_features_.at(0).image);
                temp_draw_.push_back(tape_features_.at(1).image);
                temp_draw_.push_back(tape_features_.at(2).image);

                is_gripper_initialized_ = true;
                PHASE = INIT_DESIRED;
                setSleepNonblock(0.25);
              }
              else
              {
                // can't find hands
                ROS_WARN("Cannot find the hand. Please reinitialize the hand");
                PHASE = SETTLE;
              }
            }
            break;

          case INIT_DESIRED:
            {
              ROS_INFO("Phase Initialize Desired Points");
              bool t = false;
#if PERCEPTION==PERCEPTION_COLOR_SEGMENTATION
              t = setGoalForAnObject(goal_, goal_p_, desired_);
#elif PERCEPTION==PERCEPTION_POINT_CLOUD
              if (po_.size() > 0)
                t = setGoalForAnObject(goal_, goal_p_, po_.front());
              else
              {
                ROS_INFO("No more objects to be processed. Terminating");
                PHASE = TERM;
              }
#endif
              if (t)
              {
                PHASE = POSE_CONTR;
                ROS_INFO("Phase %d, Moving to next phase in 3.0 seconds", PHASE);
              }
              else
              {
                ROS_WARN("Failed. Retrying initialization in 2 seconds");
                setSleepNonblock(2.0);
              }

            }
            break;

          case POSE_CONTR:
            {
              ROS_INFO("Phase Pose Control");
              float x = goal_p_.pose.position.x;
              float y = goal_p_.pose.position.y;
              float z = goal_p_.pose.position.z + pose_servo_z_offset_;
              visual_servo::VisualServoPose p_srv = formPoseService(x, y, z);
              ROS_INFO("Move Arm to Pose [%f %f %f]", x, y, z);
              if (p_client_.call(p_srv))
              {
                // on success
                int code = p_srv.response.result;
                ROS_INFO(">> Phase %d, Pose Control: Code [%d]", POSE_CONTR, code);
                if (0 == code)
                {
                  ROS_INFO("Phase %d, Moving to next phase in 3.0 seconds", PHASE);
                  setSleepNonblock(5.0);
#ifdef VISUAL_SERVO_TYPE
                  PHASE = VS_CONTR_2;
#else
                  PHASE = VS_CONTR_1;
#endif
                  ROS_INFO("Start Visual Servoing");
                }
              }
            }
            break;

          case VS_CONTR_1:
            {
              // Servo to WayPoint before
              // Gripper landed ON object while VSing
              // This waypoint will correct X & Y first and then correct Z (going down)
              if (tape_features_.size() == goal_.size())
              {
                std::vector<cv::Point> few_pixels_up; few_pixels_up.clear();
                float offset = (tape_features_.at(0).image.y - goal_.at(0).image.y)/2;
                for (unsigned int i = 0; i < goal_.size(); i++)
                {
                  cv::Point p = goal_.at(i).image;
                  p.y += offset; // arbitrary pixel numbers & scale
                  few_pixels_up.push_back(p);
                }
                cur_goal_ = vs_->CVPointToVSXYZ(cur_point_cloud_, cur_depth_frame_,few_pixels_up);

                visual_servo::VisualServoTwist v_srv = getTwist(cur_goal_);

                // term condition
                if (v_srv.request.error < vs_err_term_threshold_)
                {
                  PHASE = VS_CONTR_2;
                }
                // calling the service provider to move
#ifndef PROFILE
                if (v_client_.call(v_srv)){}
                else{}
#endif
              }
            }
            break;

          case VS_CONTR_2:
            {
              if (!place_detection_)
              {
                place();
                place_detection_ = true;
              }

              // compute the twist if everything is good to go
#ifdef VISUAL_SERVO_TYPE
              std::vector<PoseStamped> goals, feats;
              goals.push_back(goal_p_);
              feats.push_back(tape_features_p_);
              visual_servo::VisualServoTwist v_srv = vs_->getTwist(goals,feats,0);
              v_srv.request.error = getError(goal_, tape_features_);
#else
              visual_servo::VisualServoTwist v_srv = getTwist(goal_);
#endif
              // terminal condition
              if (is_detected_ || v_srv.request.error < vs_err_term_threshold_)
              {
                // record the height at which the object wasd picked
                object_z_ = tape_features_.front().workspace.z;
                PHASE = GRAB;
                sendZeroVelocity();
              }
              else
              {
                // calling the service provider to move
                if (v_client_.call(v_srv)){}
                else
                {
                  // on failure
                  // ROS_WARN("Service FAILED...");
                }
              }
            }
            break;
          case GRAB:
            {
              // so hand doesn't move
              sendZeroVelocity();
              if(grab())
              {
                setSleepNonblock(2.0);
                float temp = getTipDistance();
                if (temp < close_gripper_dist_ + 0.01)
                {
                  ROS_WARN(">> FAILED AT GRABBING (%f < %f). TERMINATING", temp, close_gripper_dist_ + 0.01);
                  PHASE = TERM;
                }
                else
                  PHASE = RELOCATE;
              }
              else
              {
                ROS_WARN(">> FAILED AT GRABBING. TERMINATING");
                PHASE = TERM;
              }
            }
            break;
          case RELOCATE:
            {
              ROS_INFO("Phase Relocate. Move arm to new pose");
              visual_servo::VisualServoPose p_srv = formPoseService(0.5, 0.3, 0.10+object_z_);
              if (p_client_.call(p_srv))
              {
                setSleepNonblock(2.0);
                PHASE = RELEASE;
              }
            }
            break;
          case DESCEND_INIT:
            {
              ROS_INFO("Phase Descend Init: get event detector run and register a callback");
              // inits callback for sensing collision
              place();
              PHASE = DESCEND;
              ROS_INFO("Phase Descend: descend slowly until reached a ground");
            }
          case DESCEND:
            {
              if (is_detected_)
              {
                sendZeroVelocity();
                setSleepNonblock(1.5);
                PHASE = FINISH;
              }
              else
              {
                visual_servo::VisualServoTwist v_srv = VisualServoMsg::createTwistMsg(1.0,0,0,place_z_velocity_, 0,0,0);
                v_client_.call(v_srv);
              }

              // if collision is detected, 0 velocity should be commanded
            }
            break;
          case RELEASE:
            {
              release();
              setSleepNonblock(0.5);
              PHASE = DESCEND;
            }
            break;
          case FINISH:
            {
              visual_servo::VisualServoPose p_srv = formPoseService(0.5, 0.3, 0.1 + object_z_);
              if (p_client_.call(p_srv))
              {
                setSleepNonblock(2.0);
                PHASE = TERM;
              }
            }
            break;
          default:
            {
              open();
              // make the list shorter

#if PERCEPTION==PERCEPTION_POINT_CLOUD
              if (po_.size() > 0)
                po_.pop_front();
              else
              {
                ros::shutdown();
              }
#endif

              ROS_INFO("Routine Ended.");
              std::cout << "Press [Enter] if you want to do it again: ";
              while(!ros::isShuttingDown())
              {
                int c = std::cin.get();
                if (c  == '\n')
                  break;
              }

              printf("Reset the arm and repeat Pick and Place in 3 seconds\n");
              // try to initialize the arm
              std_srvs::Empty e;
              i_client_.call(e);

              reset();
              setSleepNonblock(3.0);
              PHASE = INIT_OBJS;
            }
            break;
        }
      }
    }

    float getTipDistance()
    {
      try
      {
        std::string l_frame = "/l_gripper_l_finger_tip_link";
        std::string r_frame = "/l_gripper_r_finger_tip_link";

        tf::StampedTransform transform;
        ros::Time now = ros::Time(0);
        tf::TransformListener listener;

        listener.waitForTransform(r_frame, l_frame, now, ros::Duration(2.0));
        listener.lookupTransform(l_frame, r_frame, now, transform);
        tf::Vector3 out = transform.getOrigin();

        // PointStamped p;
        //p.header.frame_id = l_frame;
        // tf_->transformPoint(r_frame, p, p);
        return out.y() < 0 ? -out.y() : out.y();
      }
      catch (tf::TransformException e)
      {
        ROS_WARN_STREAM(e.what());
      }
      return -1.0;
    }

    void reset()
    {
      goal_.clear();
      cur_goal_.clear();
      tape_features_.clear();
      is_detected_ = false;
    }

#ifdef DISPLAY
    void setDisplay()
    {
      char phase_char[10];
      sprintf(phase_char, "Phase: %d", PHASE);
      std::string phase_str = phase_char;
      cv::putText(cur_orig_color_frame_, phase_str, cv::Point(529, 18), 2, 0.60, cv::Scalar(255, 255, 255), 1);
      cv::putText(cur_orig_color_frame_, phase_str, cv::Point(531, 18), 2, 0.60, cv::Scalar(255, 255, 255), 1);
      cv::putText(cur_orig_color_frame_, phase_str, cv::Point(530, 18), 2, 0.60, cv::Scalar(40, 40, 40), 1);

      if (goal_.size() > 0)
      {
        VSXYZ d = desired_;
        cv::putText(cur_orig_color_frame_, "+", d.image, 2, 0.5, cv::Scalar(255, 0, 255), 1);
      }

      // Draw on Desired Locations
      for (unsigned int i = 0; i < goal_.size(); i++)
      {
        cv::Point p = goal_.at(i).image;
        cv::putText(cur_orig_color_frame_, "x", p, 2, 0.5, cv::Scalar(100*i, 0, 110*(2-i), 1));
      }

      // Draw on Desired Locations
      for (unsigned int i = 0; i < cur_goal_.size(); i++)
      {
        cv::Point p = cur_goal_.at(i).image;
        cv::circle(cur_orig_color_frame_, p, 3, cv::Scalar(100*i, 0, 110*(2-i)), 1);
      }

      // Draw Features
      for (unsigned int i = 0; i < tape_features_.size(); i++)
      {
        cv::Point p = tape_features_.at(i).image;
        cv::circle(cur_orig_color_frame_, p, 2, cv::Scalar(100*i, 0, 110*(2-i)), 2);
      }

      if (PHASE == INIT_DESIRED)
      {
        cv::rectangle(cur_orig_color_frame_, original_box_, cv::Scalar(0,255,255));
      }

      if (PHASE == VS_CONTR_1 || PHASE == VS_CONTR_2)
      {
        float e = getError(goal_, tape_features_);
        if (PHASE == VS_CONTR_1)
          e = getError(cur_goal_, tape_features_);
        std::stringstream stm;
        stm << "Error :" << e;
        cv::Scalar color;
        if (e > vs_err_term_threshold_)
          color = cv::Scalar(50, 50, 255);
        else
          color = cv::Scalar(50, 255, 50);
        cv::putText(cur_orig_color_frame_, stm.str(), cv::Point(5, 12), 2, 0.5, color,1);
      }

      for (unsigned int i = 0; i < temp_draw_.size(); i++)
      {
        cv::Point p = temp_draw_.at(i);
        cv::circle(cur_orig_color_frame_, p, 3, cv::Scalar(127, 255, 0), 2);
      }

      cv::imshow("in", cur_orig_color_frame_); 
      cv::waitKey(display_wait_ms_);
    }
#endif

    /**
     * Gripper Actions: These are all blocking
     **/

    //move into event_detector mode to detect object contact
    void place()
    {

      pr2_gripper_sensor_msgs::PR2GripperEventDetectorGoal place_goal;
      place_goal.command.trigger_conditions = place_goal.command.FINGER_SIDE_IMPACT_OR_SLIP_OR_ACC;
      place_goal.command.acceleration_trigger_magnitude = 2.6;  // set the contact acceleration to n m/s^2
      place_goal.command.slip_trigger_magnitude = .005;

      is_detected_ = false;
      ROS_INFO("Waiting for object placement contact...");
      detector_client_->sendGoal(place_goal,
          boost::bind(&VisualServoNode::placeDoneCB, this, _1, _2));
    }

    void placeDoneCB(const actionlib::SimpleClientGoalState& state,
      const pr2_gripper_sensor_msgs::PR2GripperEventDetectorResultConstPtr& result)
    {
      if(state == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("[Place ActionLib] Place Success");
      else
        ROS_WARN("[Place ActionLib] Place Failure");
      is_detected_ = true;
    }

    void close()
    {
      pr2_controllers_msgs::Pr2GripperCommandGoal open;
      open.command.position = 0.0;
      open.command.max_effort = -1.0;

      ROS_INFO("Sending close goal");
      gripper_client_->sendGoal(open);
      gripper_client_->waitForResult(ros::Duration(20.0));
      if(gripper_client_->getState() == actionlib::SimpleClientGoalState::SUCCEEDED){}
      else ROS_WARN("[Gripper Action Lib] Gripper Close Failed");
    }

    void open()
    {
      pr2_controllers_msgs::Pr2GripperCommandGoal open;
      open.command.position = 0.10;
      open.command.max_effort = -1.0;

      ROS_INFO("Sending open goal");
      gripper_client_->sendGoal(open);
      gripper_client_->waitForResult(ros::Duration(20.0));
      if(gripper_client_->getState() == actionlib::SimpleClientGoalState::SUCCEEDED){}
      else ROS_WARN("[Gripper Action Lib] Gripper Open Failed");
    }

    //Open the gripper, find contact on both fingers, and go into slip-servo control mode
    bool grab()
    {
      bool ret;
      pr2_gripper_sensor_msgs::PR2GripperGrabGoal grip;
      grip.command.hardness_gain = 0.02;

      ROS_INFO("Sending grab goal");
      grab_client_->sendGoal(grip);
      grab_client_->waitForResult(ros::Duration(20.0));
      ret = grab_client_->getState() == actionlib::SimpleClientGoalState::SUCCEEDED;
      if(ret)
        ROS_INFO("Successfully completed Grab");
      else
        ROS_INFO("Grab Failed");
      return ret;
    }

    // Look for side impact, finerpad slip, or contact acceleration signals and release the object once these occur
    void release()
    {
      pr2_gripper_sensor_msgs::PR2GripperReleaseGoal place;
      // set the robot to release on a figner-side impact, fingerpad slip, or acceleration impact with hand/arm
      place.command.event.trigger_conditions = place.command.event.FINGER_SIDE_IMPACT_OR_SLIP_OR_ACC;
      // set the acceleration impact to trigger on to 5 m/s^2
      place.command.event.acceleration_trigger_magnitude = 2.5;
      // set our slip-gain to release on to .005
      place.command.event.slip_trigger_magnitude = .005;

      ROS_INFO("Waiting for object placement contact...");
      release_client_->sendGoal(place,
      boost::bind(&VisualServoNode::releaseDoneCB, this, _1, _2));
    }

    void releaseDoneCB(const actionlib::SimpleClientGoalState& state,
        const pr2_gripper_sensor_msgs::PR2GripperReleaseResultConstPtr& result)
    {
      if(state == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("[Contact ActionLib] Release Success");
      else
        ROS_WARN("[Contact ActionLib] Release Failure");
      is_detected_ = true;
      // stop the gripper right away
      sendZeroVelocity();
    }


    /**
     * HELPER
     **/
    void setSleepNonblock(float time)
    {
      ros::Time alarm = ros::Time::now() + ros::Duration(time);
      alarm_ = alarm;
    }

    bool sleepNonblock()
    {
#ifdef PROFILE
      return true;
#else
      ros::Duration d = ros::Time::now() - alarm_;
      if (d.toSec() > 0)
      {
        // set alarm to zero to be safe
        alarm_ = ros::Time(0);
        return true;
      }
      return false; 
#endif
    }

    bool initializeDesired(VSXYZ &vDesire)
    {
      cv::Mat mask_t = colorSegment(cur_orig_color_frame_.clone(), target_hue_value_ - target_hue_threshold_ , target_hue_value_ + target_hue_threshold_, 50, 100, 25, 100);
      cv::Mat element_t = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
      //cv::morphologyEx(mask_t, mask_t, cv::MORPH_CLOSE, element_t);
      cv::morphologyEx(mask_t, mask_t, cv::MORPH_OPEN, element_t);
      cv::dilate(mask_t, mask_t, element_t);

      // find the largest red
      cv::Mat in = mask_t.clone();
      std::vector<std::vector<cv::Point> > contours; contours.clear();
      cv::findContours(in, contours, cv::RETR_CCOMP,CV_CHAIN_APPROX_NONE);

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
      cv::Rect r = cv::boundingRect(ps);
      original_box_ = r;

#ifdef DISPLAY
      // for prettiness
      cv::drawContours(cur_orig_color_frame_, contours, cont_max_ind, cv::Scalar(255, 255, 255));
#endif

      int size = r.width * (r.height * 0.5 + 1);
      float temp[size]; int ind = 0;
      // this is much smaller than 480 x 640
      for(int i = r.x; i < r.width + r.x; i++)
      {
        // top 50% of the rectangle area (speed trick)
        for (int j = r.y; j < r.height * 0.5 + r.y; j++)
        {
          uchar t = mask_t.at<uchar>(j,i);
          if (t > 0)
          {
            pcl::PointXYZ p = cur_point_cloud_.at(i, j);
            if (!isnan(p.z) && p.z > -1.5)
            {
              temp[ind++] = p.z;
              cv::circle(cur_orig_color_frame_, cv::Point(i, j), 1, cv::Scalar(127, 255, 0), 1);
            }
          }
        }
      }
      std::sort(temp, temp + ind);
      float max_z = temp[ind - 1];

      float thresh = max_z - 0.2*(max_z - temp[0]);

      float mean_x = 0, mean_y =0, mean_z = 0;
      int quant = 0;
      for(int i = r.x; i < r.width + r.x; i++)
      {
        for (int j = r.y; j < r.height*0.5 + r.y; j++)
        {
          pcl::PointXYZ p = cur_point_cloud_.at(i, j);
          if(!(isnan(p.x)||isnan(p.y)||isnan(p.z)))
          {
            if (p.z > thresh)
            {
              quant++;
              mean_x += p.x;
              mean_y += p.y;
              mean_z += p.z;
            }
          }
        }
      }
      pcl::PointXYZ desired(mean_x/quant, mean_y/quant, mean_z/quant);
      vDesire = vs_->point3DToVSXYZ(desired, tf_);
      return true;
    }

    bool initializeDesired(tabletop_pushing::ProtoObjects &pos)
    {
      shared_ptr<tabletop_pushing::PointCloudSegmentation> pcs_ = shared_ptr<tabletop_pushing::PointCloudSegmentation>(
        new tabletop_pushing::PointCloudSegmentation(tf_));
      pcs_->min_table_z_ = -1.0;
      pcs_->max_table_z_ = 1.0;
      pcs_->min_workspace_x_ = -1.0;
      pcs_->max_workspace_x_ = 1.75;
      pcs_->min_workspace_z_ = -1.0;
      pcs_->max_workspace_z_ = 1.0;
      pcs_->num_downsamples_ = 2;
      pcs_->table_ransac_thresh_ = 0.015;
      pcs_->table_ransac_angle_thresh_ = 30.0;
      pcs_->cluster_tolerance_ = 0.25;
      pcs_->cloud_diff_thresh_ = 0.01;
      pcs_->min_cluster_size_ = 100;
      pcs_->max_cluster_size_ = 2500;
      pcs_->voxel_down_res_ = 0.005;
      pcs_->cloud_intersect_thresh_ = 0.005;
      pcs_->hull_alpha_ = 0.1;
      pcs_->use_voxel_down_ = true;

      try
      {
        tabletop_pushing::ProtoObjects po = pcs_->findTabletopObjects(cur_point_cloud_);
        // segmeneted no object
        if(po.size() == 0)
          return false;
        pos = po;
        return true;
      }
      catch (ros::Exception e)
      {
        ROS_WARN("FindTabletopObjects failed. Try to have only one object on the table");
        return false;
      }

    }

    bool setGoalForAnObject(std::vector<VSXYZ> &goal, PoseStamped &goal_p, tabletop_pushing::ProtoObject po)
    {
      // need to get the top of an object
      pcl::PointCloud<pcl::PointXYZ> cloud = po.cloud;
      float max_z = -5000; 
      for (unsigned int i = 0; i < cloud.size(); i++)
      {
        if (cloud.at(i).z > max_z)
        {
          max_z = cloud.at(i).z;
        }
      }
      float avg_x = 0, avg_y = 0, avg_z = 0;
      int num_avg = 0;

      float abs = max_z < 0 ? -max_z : max_z;
      float threshold = max_z - abs * 0.8;

      printf("[%f vs %f ] \n", threshold, max_z);
      for (unsigned int i = 0; i < cloud.size(); i++)
      {
        pcl::PointXYZ p = cloud.at(i);
        if (!isnan(p.z) && p.z > threshold)
        {
          if (!isnan(p.x) && !isnan(p.y))
          {
            avg_x += cloud.at(i).x;
            avg_y += cloud.at(i).y;
            avg_z += cloud.at(i).z;
            num_avg++;
          }
        }
      }
      pcl::PointXYZ avg_p = pcl::PointXYZ(avg_x/num_avg, avg_y/num_avg, avg_z/num_avg);
      VSXYZ v = vs_->point3DToVSXYZ(avg_p, tf_);
      return setGoalForAnObject(goal, goal_p, v);
    }

    bool setGoalForAnObject(std::vector<VSXYZ> &goal, PoseStamped &goal_p, VSXYZ desire)
    {
      if (isnan(desire.workspace.x) || isnan(desire.workspace.y) ||
          isnan(desire.workspace.z))
      {
        ROS_ERROR("Desire Values have NaN. Unable to Proceed Further");
        return false;
      }

      std::vector<pcl::PointXYZ> temp_features = gripper_tape_.getTapePoseFromXYZ(desire.workspace);
      std::vector<VSXYZ> desired_vsxyz = vs_->Point3DToVSXYZ(temp_features, tf_);
      goal = desired_vsxyz;

      printf("%d\n", (int)goal.size());

      goal_p = vs_->VSXYZToPoseStamped(goal_.front());
      // ORIENT
      /*
      goal_p.pose.orientation.x = -0.4582;
      goal_p.pose.orientation.z = 0.8889;
      */

      // for jacobian avg, we need IM at desired location as well
      if (JACOBIAN_TYPE_AVG == jacobian_type_)
        return vs_->setDesiredInteractionMatrix(desired_vsxyz);
      return true;
    }

    void initializeService()
    {
      ROS_DEBUG(">> Hooking Up The Service");
      ros::NodeHandle n;
      v_client_ = n.serviceClient<visual_servo::VisualServoTwist>("vs_twist");
      p_client_ = n.serviceClient<visual_servo::VisualServoPose>("vs_pose");
      i_client_ = n.serviceClient<std_srvs::Empty>("vs_init");

      gripper_client_  = new GripperClient("l_gripper_sensor_controller/gripper_action",true);
      while(!gripper_client_->waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for the r_gripper_sensor_controller/event_detector action server to come up");
      }
      grab_client_  = new GrabClient("l_gripper_sensor_controller/grab",true);
      while(!grab_client_->waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for the r_gripper_sensor_controller/grab action server to come up");
      }
      release_client_  = new ReleaseClient("l_gripper_sensor_controller/release",true);
      while(!release_client_->waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for the r_gripper_sensor_controller/release action server to come up");
      }
      detector_client_ = new EventDetectorClient("l_gripper_sensor_controller/event_detector",true);
      while(!detector_client_->waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for the r_gripper_sensor_controller/event_detector action server to come up");
      }

    }
    // Service method
    visual_servo::VisualServoTwist getTwist(std::vector<VSXYZ> desire)
    {
      visual_servo::VisualServoTwist srv;
      srv = vs_->getTwist(desire, tape_features_);

      if (gripper_pose_estimated_)
      {
        float scale = 0.5;
        srv.request.twist.twist.linear.x *= scale;
        srv.request.twist.twist.linear.y *= scale;
        srv.request.twist.twist.linear.z *= scale;
        // don't let it have orientational velocity
        srv.request.twist.twist.angular.x = 0;
        srv.request.twist.twist.angular.y = 0;
        srv.request.twist.twist.angular.z = 0;
      }
      srv.request.error = getError(desire, tape_features_);
      return srv;
    }

    // Detect Tapes on Gripepr and update its position
    bool updateGripperFeatures()
    {
      bool estimated = false;
      int default_tape_num = 3;

      PoseStamped p;
      p.header.frame_id = "/l_gripper_tool_frame";
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
      if (tape_features_.size() == 0 )
      {
        if (v_fk_diff_.size() > 0)
        {
          ROS_WARN("Cannot find tape in image and do not have enough historical data to interpolate");
          tape_features_p_ = PoseStamped();
          return false;
        }
        estimated = true;
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

      // ORIENT
      /*
         QuaternionStamped p;
         p.quaternion.w = 1;
         p.header.frame_id = "/l_gripper_tool_frame";
         tf_->transformQuaternion(workspace_frame_, p, p);
         tape_features_p_.pose.orientation = p.quaternion;
       */

      // we aren't going to let controllers rotate at all when occluded;
      // vision vs. forward kinematics
      if (!estimated)
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
      return estimated;
    }

    float getError(std::vector<VSXYZ> a, std::vector<VSXYZ> b)
    {
      float e(0.0);
      unsigned int size = a.size() <= b.size() ? a.size() : b.size();

      if (size < 3)
        return 1;

      for (unsigned int i = 0; i < size; i++)
      {
        pcl::PointXYZ a_c= a.at(i).camera;
        pcl::PointXYZ b_c= b.at(i).camera;
        e += pow(a_c.x - b_c.x ,2) + pow(a_c.y - b_c.y ,2);
      }
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

    /**********************
     * HELPER METHODS
     **********************/

    bool sendZeroVelocity()
    {
      // assume everything is init to zero
      visual_servo::VisualServoTwist v_srv;

      // need this to mvoe the arm
      v_srv.request.error = 1;
      return v_client_.call(v_srv);
    }

    visual_servo::VisualServoPose formPoseService(float px, float py, float pz)
    {
      return VisualServoMsg::createPoseMsg(px, py, pz, -0.4582, 0, 0.8889, 0);
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
    XYZPointCloud cur_point_cloud_;

    // visual servo object
    shared_ptr<VisualServo> vs_;

    bool have_depth_data_;
    int display_wait_ms_;
    int num_downsamples_;
    std::string workspace_frame_;
    std::string optical_frame_;
    bool camera_initialized_;
    bool desire_points_initialized_;
    std::string cam_info_topic_;
    int tracker_count_;

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

    // State machine variable
    unsigned int PHASE;

    // desired location/current gripper location
    cv::Mat desired_jacobian_;
    std::vector<VSXYZ> cur_goal_;
    std::vector<VSXYZ> goal_;
    PoseStamped goal_p_;
    std::vector<VSXYZ> tape_features_;
    PoseStamped tape_features_p_;
    VSXYZ desired_;
    cv::Mat K;
    std::vector<pcl::PointXYZ> v_fk_diff_;
    bool is_gripper_initialized_;
    bool gripper_pose_estimated_;

    // for debugging purpose
    ros::Publisher chatter_pub_;
    ros::Time alarm_;

    // clients to services provided by vs_controller.py
    ros::ServiceClient v_client_;
    ros::ServiceClient p_client_;
    ros::ServiceClient i_client_;

    // gripper sensor action clients
    GripperClient* gripper_client_;
    GrabClient* grab_client_;
    ReleaseClient* release_client_;
    EventDetectorClient* detector_client_;

    //  drawing
    std::vector<cv::Point> temp_draw_;
    cv::Rect original_box_;

    // collision detection
    bool is_detected_;
    bool place_detection_;
    float object_z_;

    float close_gripper_dist_;
    GripperTape gripper_tape_;

    // number of objects
    tabletop_pushing::ProtoObjects po_;

};

int main(int argc, char ** argv)
{
  srand(time(NULL));
  ros::init(argc, argv, "visual_servo_node");

  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);

  ros::NodeHandle n;
  VisualServoNode vs_node(n);
  vs_node.spin();
}
