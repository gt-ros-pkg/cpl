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
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <actionlib/server/simple_action_server.h>

#include <pr2_manipulation_controllers/JTTaskControllerState.h>
#include <pr2_manipulation_controllers/JinvTeleopControllerState.h>

// TF
#include <tf/transform_listener.h>

// PCL
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <pcl16/common/common.h>
#include <pcl16/common/eigen.h>
#include <pcl16/common/centroid.h>
#include <pcl16/io/io.h>
#include <pcl16/io/pcd_io.h>
#include <pcl16_ros/transforms.h>
#include <pcl16/ros/conversions.h>
#include <pcl16/ModelCoefficients.h>
#include <pcl16/registration/transformation_estimation_svd.h>
#include <pcl16/sample_consensus/method_types.h>
#include <pcl16/sample_consensus/model_types.h>
#include <pcl16/segmentation/sac_segmentation.h>
#include <pcl16/filters/extract_indices.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost
#include <boost/shared_ptr.hpp>

// cpl_visual_features
#include <cpl_visual_features/helpers.h>
#include <cpl_visual_features/features/shape_context.h>

// tabletop_pushing
#include <tabletop_pushing/LearnPush.h>
#include <tabletop_pushing/LocateTable.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <tabletop_pushing/shape_features.h>
#include <tabletop_pushing/object_tracker_25d.h>
#include <tabletop_pushing/push_primitives.h>

// libSVM
#include <libsvm/svm.h>

// STL
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <iostream>
#include <utility>
#include <float.h>
#include <math.h>
#include <cmath>
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND

// Debugging IFDEFS
#define DISPLAY_INPUT_COLOR 1
#define DISPLAY_INPUT_DEPTH 1
#define DISPLAY_WAIT 1

using boost::shared_ptr;

using tabletop_pushing::LearnPush;
using tabletop_pushing::LocateTable;
using tabletop_pushing::PushVector;
using tabletop_pushing::PointCloudSegmentation;
using tabletop_pushing::ProtoObject;
using tabletop_pushing::ProtoObjects;
using tabletop_pushing::ShapeLocation;
using tabletop_pushing::ShapeLocations;
using tabletop_pushing::ObjectTracker25D;

using geometry_msgs::PoseStamped;
using geometry_msgs::PointStamped;
using geometry_msgs::Pose2D;
using geometry_msgs::Twist;
typedef pcl16::PointCloud<pcl16::PointXYZ> XYZPointCloud;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image,
                                                        sensor_msgs::Image,
                                                        sensor_msgs::PointCloud2> MySyncPolicy;
using cpl_visual_features::upSample;
using cpl_visual_features::downSample;
using cpl_visual_features::subPIAngle;
using cpl_visual_features::ShapeDescriptors;
using cpl_visual_features::ShapeDescriptor;

typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;
typedef tabletop_pushing::VisFeedbackPushTrackingGoal PushTrackerGoal;
typedef tabletop_pushing::VisFeedbackPushTrackingResult PushTrackerResult;
typedef tabletop_pushing::VisFeedbackPushTrackingAction PushTrackerAction;

class TabletopPushingPerceptionNode
{
 public:
  TabletopPushingPerceptionNode(ros::NodeHandle &n) :
      n_(n), n_private_("~"),
      image_sub_(n, "color_image_topic", 1),
      depth_sub_(n, "depth_image_topic", 1),
      mask_sub_(n, "mask_image_topic", 1),
      cloud_sub_(n, "point_cloud_topic", 1),
      sync_(MySyncPolicy(15), image_sub_, depth_sub_, mask_sub_, cloud_sub_),
      as_(n, "push_tracker", false),
      have_depth_data_(false),
      camera_initialized_(false), recording_input_(false), record_count_(0),
      learn_callback_count_(0), goal_out_count_(0), goal_heading_count_(0),
      frame_callback_count_(0),
      just_spun_(false), major_axis_spin_pos_scale_(0.75), object_not_moving_thresh_(0),
      object_not_moving_count_(0), object_not_moving_count_limit_(10),
      gripper_not_moving_thresh_(0), gripper_not_moving_count_(0),
      gripper_not_moving_count_limit_(10), current_file_id_(""), force_swap_(false)

  {
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());
    pcl_segmenter_ = shared_ptr<PointCloudSegmentation>(
        new PointCloudSegmentation(tf_));
    // Get parameters from the server
    n_private_.param("display_wait_ms", display_wait_ms_, 3);
    n_private_.param("use_displays", use_displays_, false);
    n_private_.param("write_input_to_disk", write_input_to_disk_, false);
    n_private_.param("write_to_disk", write_to_disk_, false);

    n_private_.param("min_workspace_x", pcl_segmenter_->min_workspace_x_, 0.0);
    n_private_.param("min_workspace_z", pcl_segmenter_->min_workspace_z_, 0.0);
    n_private_.param("max_workspace_x", pcl_segmenter_->max_workspace_x_, 0.0);
    n_private_.param("max_workspace_z", pcl_segmenter_->max_workspace_z_, 0.0);
    n_private_.param("min_table_z", pcl_segmenter_->min_table_z_, -0.5);
    n_private_.param("max_table_z", pcl_segmenter_->max_table_z_, 1.5);

    n_private_.param("mps_min_inliers", pcl_segmenter_->mps_min_inliers_, 10000);
    n_private_.param("mps_min_angle_thresh", pcl_segmenter_->mps_min_angle_thresh_, 2.0);
    n_private_.param("mps_min_dist_thresh", pcl_segmenter_->mps_min_dist_thresh_, 0.02);

    std::string default_workspace_frame = "torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_,
                     default_workspace_frame);
    std::string default_camera_frame = "head_mount_kinect_rgb_optical_frame";
    n_private_.param("camera_frame", camera_frame_, default_camera_frame);

    std::string output_path_def = "~";
    n_private_.param("img_output_path", base_output_path_, output_path_def);

    n_private_.param("start_tracking_on_push_call", start_tracking_on_push_call_, false);

    n_private_.param("num_downsamples", num_downsamples_, 2);
    pcl_segmenter_->num_downsamples_ = num_downsamples_;

    std::string cam_info_topic_def = "/kinect_head/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_, cam_info_topic_def);

    // PCL Segmentation parameters
    n_private_.param("table_ransac_thresh", pcl_segmenter_->table_ransac_thresh_, 0.01);
    n_private_.param("table_ransac_angle_thresh", pcl_segmenter_->table_ransac_angle_thresh_, 30.0);
    n_private_.param("cylinder_ransac_thresh", pcl_segmenter_->cylinder_ransac_thresh_, 0.03);
    n_private_.param("cylinder_ransac_angle_thresh", pcl_segmenter_->cylinder_ransac_angle_thresh_, 1.5);
    n_private_.param("optimize_cylinder_coefficients", pcl_segmenter_->optimize_cylinder_coefficients_, false);
    n_private_.param("sphere_ransac_thresh", pcl_segmenter_->sphere_ransac_thresh_, 0.01);
    n_private_.param("pcl_cluster_tolerance", pcl_segmenter_->cluster_tolerance_, 0.25);
    n_private_.param("pcl_difference_thresh", pcl_segmenter_->cloud_diff_thresh_, 0.01);
    n_private_.param("pcl_min_cluster_size", pcl_segmenter_->min_cluster_size_, 100);
    n_private_.param("pcl_max_cluster_size", pcl_segmenter_->max_cluster_size_, 2500);
    n_private_.param("pcl_voxel_downsample_res", pcl_segmenter_->voxel_down_res_, 0.005);
    n_private_.param("pcl_cloud_intersect_thresh", pcl_segmenter_->cloud_intersect_thresh_, 0.005);
    n_private_.param("pcl_concave_hull_alpha", pcl_segmenter_->hull_alpha_, 0.1);
    n_private_.param("use_pcl_voxel_downsample", pcl_segmenter_->use_voxel_down_, true);
    n_private_.param("icp_max_iters", pcl_segmenter_->icp_max_iters_, 100);
    n_private_.param("icp_transform_eps", pcl_segmenter_->icp_transform_eps_, 0.0);
    n_private_.param("icp_max_cor_dist", pcl_segmenter_->icp_max_cor_dist_, 1.0);
    n_private_.param("icp_ransac_thresh", pcl_segmenter_->icp_ransac_thresh_, 0.015);

    n_private_.param("push_tracker_dist_thresh", tracker_dist_thresh_, 0.05);
    n_private_.param("push_tracker_angle_thresh", tracker_angle_thresh_, 0.01);
    n_private_.param("major_axis_spin_pos_scale", major_axis_spin_pos_scale_, 0.75);
    bool use_cv_ellipse;
    n_private_.param("use_cv_ellipse", use_cv_ellipse, false);
    n_private_.param("use_mps_segmentation", use_mps_segmentation_, false);

    n_private_.param("max_object_gripper_dist", max_object_gripper_dist_, 0.10);
    n_private_.param("max_object_tool_dist", max_object_tool_dist_, 0.10);
    n_private_.param("gripper_not_moving_thresh", gripper_not_moving_thresh_, 0.005);
    n_private_.param("object_not_moving_thresh", object_not_moving_thresh_, 0.005);
    n_private_.param("gripper_not_moving_count_limit", gripper_not_moving_count_limit_, 100);
    n_private_.param("object_not_moving_count_limit", object_not_moving_count_limit_, 100);
    n_private_.param("object_not_detected_count_limit", object_not_detected_count_limit_, 5);
    n_private_.param("object_too_far_count_limit", object_too_far_count_limit_, 5);
    n_private_.param("object_not_between_count_limit", object_not_between_count_limit_, 5);
    n_private_.param("object_not_between_epsilon", object_not_between_epsilon_, 0.01);
    n_private_.param("object_not_between_tool_epsilon", object_not_between_tool_epsilon_, 0.01);
    n_private_.param("start_loc_push_time_limit", start_loc_push_time_, 5.0);
    n_private_.param("start_loc_push_dist", start_loc_push_dist_, 0.30);
    n_private_.param("use_center_pointing_shape_context", use_center_pointing_shape_context_, true);
    n_private_.param("self_mask_dilate_size", mask_dilate_size_, 5);
    n_private_.param("point_cloud_hist_res", point_cloud_hist_res_, 0.005);

    n_.param("start_loc_use_fixed_goal", start_loc_use_fixed_goal_, false);


    // Initialize classes requiring parameters
    obj_tracker_ = shared_ptr<ObjectTracker25D>(
        new ObjectTracker25D(pcl_segmenter_, num_downsamples_, use_displays_, write_to_disk_,
                             base_output_path_, camera_frame_, use_cv_ellipse, use_mps_segmentation_));

    // Setup ros node connections
    sync_.registerCallback(&TabletopPushingPerceptionNode::sensorCallback,
                           this);
    push_pose_server_ = n_.advertiseService(
        "get_learning_push_vector",
        &TabletopPushingPerceptionNode::learnPushCallback, this);
    table_location_server_ = n_.advertiseService(
        "get_table_location", &TabletopPushingPerceptionNode::getTableLocation,
        this);

    // Setup arm controller state callbacks
    jtteleop_l_arm_subscriber_ = n_.subscribe("/l_cart_transpose_push/state", 1,
                                              &TabletopPushingPerceptionNode::lArmStateCartCB,
                                              this);
    jtteleop_r_arm_subscriber_ = n_.subscribe("/r_cart_transpose_push/state", 1,
                                              &TabletopPushingPerceptionNode::rArmStateCartCB,
                                              this);
    jinv_l_arm_subscriber_  = n_.subscribe("/l_cart_jinv_push/state", 1,
                                           &TabletopPushingPerceptionNode::lArmStateVelCB,
                                           this);
    jinv_r_arm_subscriber_ = n_.subscribe("/r_cart_jinv_push/state", 1,
                                          &TabletopPushingPerceptionNode::rArmStateVelCB,
                                          this);
    // Setup push tracking action server
    as_.registerGoalCallback(
        boost::bind(&TabletopPushingPerceptionNode::pushTrackerGoalCB, this));
    as_.registerPreemptCallback(
        boost::bind(&TabletopPushingPerceptionNode::pushTrackerPreemptCB,this));
    as_.start();
  }

  void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg,
                      const sensor_msgs::ImageConstPtr& depth_msg,
                      const sensor_msgs::ImageConstPtr& mask_msg,
                      const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    if (!camera_initialized_)
    {
      cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
          cam_info_topic_, n_, ros::Duration(5.0));
      camera_initialized_ = true;
      pcl_segmenter_->cam_info_ = cam_info_;
      ROS_DEBUG_STREAM("Cam info: " << cam_info_);
    }
    // Convert images to OpenCV format
    cv::Mat color_frame;
    cv::Mat depth_frame;
    cv::Mat self_mask;
    cv_bridge::CvImagePtr color_cv_ptr = cv_bridge::toCvCopy(img_msg);
    cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg);
    cv_bridge::CvImagePtr mask_cv_ptr = cv_bridge::toCvCopy(mask_msg);
    color_frame = color_cv_ptr->image;
    depth_frame = depth_cv_ptr->image;
    self_mask = mask_cv_ptr->image;

    // Grow arm mask if requested
    if (mask_dilate_size_ > 0)
    {
      cv::Mat morph_element(mask_dilate_size_, mask_dilate_size_, CV_8UC1, cv::Scalar(255));
      cv::erode(self_mask, self_mask, morph_element);
    }

    // Transform point cloud into the correct frame and convert to PCL struct
    XYZPointCloud cloud;
    pcl16::fromROSMsg(*cloud_msg, cloud);
    tf_->waitForTransform(workspace_frame_, cloud.header.frame_id, cloud.header.stamp, ros::Duration(0.5));
    pcl16_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);

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

    XYZPointCloud cloud_self_filtered(cloud);
    pcl16::PointXYZ nan_point;
    nan_point.x = numeric_limits<float>::quiet_NaN();
    nan_point.y = numeric_limits<float>::quiet_NaN();
    nan_point.z = numeric_limits<float>::quiet_NaN();
    for (unsigned int x = 0; x < cloud.width; ++x)
    {
      for (unsigned int y = 0; y < cloud.height; ++y)
      {
        if (self_mask.at<uchar>(y,x) == 0)
        {
          cloud_self_filtered.at(x,y) = nan_point;
        }
      }
    }

    // Downsample everything first
    cv::Mat color_frame_down = downSample(color_frame, num_downsamples_);
    cv::Mat depth_frame_down = downSample(depth_frame, num_downsamples_);
    cv::Mat self_mask_down = downSample(self_mask, num_downsamples_);
    cv::Mat arm_mask_crop;
    color_frame_down.copyTo(arm_mask_crop, self_mask_down);

    // Save internally for use in the service callback
    prev_color_frame_ = cur_color_frame_.clone();
    prev_depth_frame_ = cur_depth_frame_.clone();
    prev_self_mask_ = cur_self_mask_.clone();
    prev_camera_header_ = cur_camera_header_;

    // Update the current versions
    cur_color_frame_ = color_frame_down.clone();
    cur_depth_frame_ = depth_frame_down.clone();
    cur_self_mask_ = self_mask_down.clone();
    cur_point_cloud_ = cloud;
    cur_self_filtered_cloud_ = cloud_self_filtered;
    have_depth_data_ = true;
    cur_camera_header_ = img_msg->header;
    pcl_segmenter_->cur_camera_header_ = cur_camera_header_;

    if (obj_tracker_->isInitialized() && !obj_tracker_->isPaused())
    {
      PoseStamped arm_pose;
      if (pushing_arm_ == "l")
      {
        arm_pose = l_arm_pose_;
      }
      else
      {
        arm_pose = r_arm_pose_;
      }
      PushTrackerState tracker_state = obj_tracker_->updateTracks(
          cur_color_frame_, cur_self_mask_, cur_self_filtered_cloud_, proxy_name_, arm_pose, tool_proxy_name_);
      tracker_state.proxy_name = proxy_name_;
      tracker_state.controller_name = controller_name_;
      tracker_state.behavior_primitive = behavior_primitive_;
      tracker_state.tool_proxy_name = tool_proxy_name_;

      PointStamped start_point;
      PointStamped end_point;
      start_point.header.frame_id = workspace_frame_;
      end_point.header.frame_id = workspace_frame_;
      start_point.point.x = tracker_state.x.x;
      start_point.point.y = tracker_state.x.y;
      start_point.point.z = tracker_state.z;
      end_point.point.x = tracker_goal_pose_.x;
      end_point.point.y = tracker_goal_pose_.y;
      end_point.point.z = start_point.point.z;

      displayPushVector(cur_color_frame_, start_point, end_point);
      // displayRobotGripperPoses(cur_color_frame_);
      // displayGoalHeading(cur_color_frame_, start_point, tracker_state.x.theta,
      //                    tracker_goal_pose_.theta);

      // make sure that the action hasn't been canceled
      if (as_.isActive())
      {
        as_.publishFeedback(tracker_state);
        evaluateGoalAndAbortConditions(tracker_state);
      }
    }
    else if (obj_tracker_->isInitialized() && obj_tracker_->isPaused())
    {
      obj_tracker_->pausedUpdate(cur_color_frame_);
      PointStamped start_point;
      PointStamped end_point;
      PushTrackerState tracker_state = obj_tracker_->getMostRecentState();
      start_point.header.frame_id = workspace_frame_;
      end_point.header.frame_id = workspace_frame_;
      start_point.point.x = tracker_state.x.x;
      start_point.point.y = tracker_state.x.y;
      start_point.point.z = tracker_state.z;
      end_point.point.x = tracker_goal_pose_.x;
      end_point.point.y = tracker_goal_pose_.y;
      end_point.point.z = start_point.point.z;
      displayPushVector(cur_color_frame_, start_point, end_point, "goal_vector", true);
      // displayRobotGripperPoses(cur_color_frame_);
      // displayGoalHeading(cur_color_frame_, start_point, tracker_state.x.theta,
      //                    tracker_goal_pose_.theta);
    }

    // Display junk
#ifdef DISPLAY_INPUT_COLOR
    if (use_displays_)
    {
      cv::imshow("color", cur_color_frame_);
      cv::imshow("self_mask", cur_self_mask_);
      cv::imshow("arm_mask_crop", arm_mask_crop);
    }
    // Way too much disk writing!
    if (write_input_to_disk_ && recording_input_)
    {
      std::stringstream out_name;
      if (current_file_id_.size() > 0)
      {
        std::stringstream cloud_out_name;
        out_name << base_output_path_ << current_file_id_ << "_input_" << record_count_ << ".png";
        // cloud_out_name << base_output_path_ << current_file_id_ << "_object_" << record_count_ << ".pcd";
        // ProtoObject cur_obj = obj_tracker_->getMostRecentObject();
        // pcl16::io::savePCDFile(cloud_out_name.str(), cur_obj.cloud);
      }
      else
      {
        out_name << base_output_path_ << "input" << record_count_ << ".png";
      }
      cv::imwrite(out_name.str(), cur_color_frame_);
      // std::stringstream self_out_name;
      // self_out_name << base_output_path_ << "self" << record_count_ << ".png";
      // cv::imwrite(self_out_name.str(), cur_self_mask_);
      record_count_++;
    }
#endif // DISPLAY_INPUT_COLOR
#ifdef DISPLAY_INPUT_DEPTH
    if (use_displays_)
    {
      double depth_max = 1.0;
      cv::minMaxLoc(cur_depth_frame_, NULL, &depth_max);
      cv::Mat depth_display = cur_depth_frame_.clone();
      depth_display /= depth_max;
      cv::imshow("input_depth", depth_display);
    }
#endif // DISPLAY_INPUT_DEPTH
#ifdef DISPLAY_WAIT
    if (use_displays_)
    {
      cv::waitKey(display_wait_ms_);
    }
#endif // DISPLAY_WAIT
    ++frame_callback_count_;
  }

  void evaluateGoalAndAbortConditions(PushTrackerState& tracker_state)
  {
    // Check for goal conditions
    float x_error = tracker_goal_pose_.x - tracker_state.x.x;
    float y_error = tracker_goal_pose_.y - tracker_state.x.y;
    float theta_error = subPIAngle(tracker_goal_pose_.theta - tracker_state.x.theta);

    float x_dist = fabs(x_error);
    float y_dist = fabs(y_error);
    float theta_dist = fabs(theta_error);

    if (timing_push_)
    {
      if (pushingTimeUp())
      {
        abortPushingGoal("Pushing time up");
      }
      return;
    }

    if (controller_name_ == "spin_to_heading")
    {
      if (theta_dist < tracker_angle_thresh_)
      {
        ROS_INFO_STREAM("Cur state: (" << tracker_state.x.x << ", " <<
                        tracker_state.x.y << ", " << tracker_state.x.theta << ")");
        ROS_INFO_STREAM("Desired goal: (" << tracker_goal_pose_.x << ", " <<
                        tracker_goal_pose_.y << ", " << tracker_goal_pose_.theta << ")");
        ROS_INFO_STREAM("Goal error: (" << x_dist << ", " << y_dist << ", "
                        << theta_dist << ")");
        PushTrackerResult res;
        res.aborted = false;
        as_.setSucceeded(res);
        obj_tracker_->pause();
      }
      return;
    }

    if (x_dist < tracker_dist_thresh_ && y_dist < tracker_dist_thresh_)
    {
      ROS_INFO_STREAM("Cur state: (" << tracker_state.x.x << ", " <<
                      tracker_state.x.y << ", " << tracker_state.x.theta << ")");
      ROS_INFO_STREAM("Desired goal: (" << tracker_goal_pose_.x << ", " <<
                      tracker_goal_pose_.y << ", " << tracker_goal_pose_.theta << ")");
      ROS_INFO_STREAM("Goal error: (" << x_dist << ", " << y_dist << ", "
                      << theta_dist << ")");
      PushTrackerResult res;
      res.aborted = false;
      as_.setSucceeded(res);
      obj_tracker_->pause();
      return;
    }

    if (objectNotMoving(tracker_state))
    {
      abortPushingGoal("Object is not moving");
    }
    else if (gripperNotMoving())
    {
      abortPushingGoal("Gripper is not moving");
    }
    else if (objectDisappeared(tracker_state))
    {
      abortPushingGoal("Object disappeared");
    }
    else if (controller_name_ != "tool_centroid_controller")
    {
      if (objectTooFarFromGripper(tracker_state.x))
      {
        abortPushingGoal("Object is too far from gripper.");
      }
      else if (behavior_primitive_ != "gripper_pull" &&
               objectNotBetweenGoalAndGripper(tracker_state.x))
      {
        abortPushingGoal("Object is not between gripper and goal.");
      }
    }
    else
    {
      // TODO: Fix this to be based on the tool_proxy being used
      PoseStamped tool_state;
      if (pushing_arm_ == "l")
      {
        tool_state = l_arm_pose_;
      }
      else
      {
        tool_state = r_arm_pose_;
      }

      if (objectTooFarFromTool(tracker_state.x, tool_state))
      {
        abortPushingGoal("Object is too far from tool.");
      }
      else if (objectNotBetweenGoalAndTool(tracker_state.x, tool_state))
      {
        abortPushingGoal("Object is not between tool and goal.");
      }
    }
  }

  void abortPushingGoal(std::string msg)
  {
    ROS_WARN_STREAM(msg << " Aborting.");
    PushTrackerResult res;
    res.aborted = true;
    as_.setAborted(res);
    obj_tracker_->pause();
  }

  /**
   * Service request callback method to return a location and orientation for
   * the robot to push.
   *
   * @param req The service request
   * @param res The service response
   *
   * @return true if successfull, false otherwise
   */
  bool learnPushCallback(LearnPush::Request& req, LearnPush::Response& res)
  {
    if ( have_depth_data_ )
    {
      if (!req.analyze_previous)
      {
        controller_name_ = req.controller_name;
        proxy_name_ = req.proxy_name;
        behavior_primitive_ = req.behavior_primitive;
        tool_proxy_name_ = req.tool_proxy_name;
      }

      if (req.initialize)
      {
        ROS_INFO_STREAM("Initializing");
        record_count_ = 0;
        learn_callback_count_ = 0;
        res.no_push = true;
        recording_input_ = false;
        obj_tracker_->stopTracking();
      }
      else if (req.analyze_previous || req.get_pose_only)
      {
        ROS_INFO_STREAM("Getting current object pose\n");
        res = getObjectPose();
        res.no_push = true;
        recording_input_ = false;
        obj_tracker_->stopTracking();
      }
      else // NOTE: Assume pushing as default
      {
        ROS_INFO_STREAM("Determining push start pose");
        res = getPushStartPose(req);
        recording_input_ = !res.no_objects;
        if (recording_input_)
        {
          ROS_INFO_STREAM("Starting input recording");
          ROS_INFO_STREAM("current_file_id: " << current_file_id_);
        }
        else
        {
          ROS_INFO_STREAM("Stopping input recording");
        }
        res.no_push = res.no_objects;
      }
    }
    else
    {
      ROS_ERROR_STREAM("Calling getStartPose prior to receiving sensor data.");
      recording_input_ = false;
      res.no_push = true;
      return false;
    }
    return true;
  }

  LearnPush::Response getPushStartPose(LearnPush::Request& req)
  {
    LearnPush::Response res;
    PushTrackerState cur_state;
    cur_state = startTracking(force_swap_);

    ProtoObject cur_obj = obj_tracker_->getMostRecentObject();
    if (req.learn_start_loc)
    {
      obj_tracker_->trackerDisplay(cur_color_frame_, cur_state, cur_obj);
      ROS_INFO_STREAM("Current theta: " << cur_state.x.theta);
      ROS_INFO_STREAM("Presss 's' to swap orientation: ");
      char key_press = cv::waitKey(2000);
      if (key_press == 's')
      {
        force_swap_ = !force_swap_;
        obj_tracker_->toggleSwap();
        cur_state = startTracking(force_swap_);
        obj_tracker_->trackerDisplay(cur_color_frame_, cur_state, cur_obj);
        // NOTE: Try and force redraw
        cv::waitKey(3);
        ROS_INFO_STREAM("Swapped theta: " << cur_state.x.theta);
      }
    }

    if (!start_tracking_on_push_call_)
    {
      obj_tracker_->pause();
    }

    if (cur_state.no_detection)
    {
      ROS_WARN_STREAM("No objects found");
      res.centroid.x = 0.0;
      res.centroid.y = 0.0;
      res.centroid.z = 0.0;
      res.theta = 0.0;
      res.no_objects = true;
      return res;
    }
    res.no_objects = false;
    res.centroid.x = cur_obj.centroid[0];
    res.centroid.y = cur_obj.centroid[1];
    res.centroid.z = cur_obj.centroid[2];
    res.theta = cur_state.x.theta;
    res.tool_x = cur_state.tool_x;

    // Set basic push information
    PushVector p;
    p.header.frame_id = workspace_frame_;
    bool pull_start = (req.behavior_primitive == "gripper_pull");
    bool spin_push = (req.controller_name == "spin_to_heading");

    // Choose a pushing location to test if we are learning good pushing locations
    if (req.learn_start_loc)
    {
      // Get the pushing location
      ShapeLocation chosen_loc;
      if (req.start_loc_param_path.length() > 0) // TODO: Choose start location using the learned classifier
      {
        float chosen_score = -1;
        chosen_loc = chooseLearnedPushStartLoc(cur_obj, cur_state, req.start_loc_param_path, chosen_score);
      }
      else if (start_loc_use_fixed_goal_)
      {
        chosen_loc = chooseFixedGoalPushStartLoc(cur_obj, cur_state, req.new_object,
                                                 req.num_start_loc_pushes_per_sample, req.num_start_loc_sample_locs,
                                                 req.trial_id);
      }
      else
      {
        chosen_loc = choosePushStartLoc(cur_obj, cur_state, req.new_object, req.num_start_loc_clusters);
      }
      ROS_INFO_STREAM("Chosen loc is: (" << chosen_loc.boundary_loc_.x << ", " << chosen_loc.boundary_loc_.y << ")");
      res.shape_descriptor.assign(chosen_loc.descriptor_.begin(), chosen_loc.descriptor_.end());
      float new_push_angle;
      if (spin_push)
      {
        // Set goal for spin pushing angle and goal state; then get start location as usual below
        new_push_angle = getSpinPushHeading(cur_state, chosen_loc);
        res.goal_pose.x = cur_state.x.x;
        res.goal_pose.y = cur_state.x.y;
        // NOTE: Uncomment for visualization purposes
        // res.goal_pose.x = res.centroid.x+cos(new_push_angle)*start_loc_push_dist_;
        // res.goal_pose.y = res.centroid.y+sin(new_push_angle)*start_loc_push_dist_;
        res.goal_pose.theta = req.goal_pose.theta;
      }
      else
      {
        // Set goal for pushing and then get start location as usual below
        new_push_angle = atan2(res.centroid.y - chosen_loc.boundary_loc_.y,
                               res.centroid.x - chosen_loc.boundary_loc_.x);
        res.goal_pose.x = res.centroid.x+cos(new_push_angle)*start_loc_push_dist_;
        res.goal_pose.y = res.centroid.y+sin(new_push_angle)*start_loc_push_dist_;
      }

      p.start_point.x = chosen_loc.boundary_loc_.x;
      p.start_point.y = chosen_loc.boundary_loc_.y;
      p.start_point.z = chosen_loc.boundary_loc_.z;
      p.push_angle = new_push_angle;
      p.push_dist = start_loc_push_dist_;

      // Push for a fixed amount of time
      timing_push_ = true;

      // NOTE: Write object point cloud to disk, images too for use in offline learning if we want to
      // change features in the future
      if (write_to_disk_)
      {
        std::stringstream cloud_file_name;
        cloud_file_name << base_output_path_ << req.trial_id << "_obj_cloud.pcd";
        std::stringstream color_file_name;
        color_file_name << base_output_path_ << req.trial_id << "_color.png";
        current_file_id_ = req.trial_id;
        pcl16::io::savePCDFile(cloud_file_name.str(), cur_obj.cloud);
        cv::imwrite(color_file_name.str(), cur_color_frame_);
      }
    }
    else
    {
      res.goal_pose.x = req.goal_pose.x;
      res.goal_pose.y = req.goal_pose.y;

      // Get straight line from current location to goal pose as start
      if (pull_start)
      {
        // NOTE: Want the opposite direction for pulling as pushing
        p.push_angle = atan2(res.centroid.y - res.goal_pose.y, res.centroid.x - res.goal_pose.x);
      }
      else if (spin_push)
      {
        // TODO: Figure out something here
      }
      else
      {
        p.push_angle = atan2(res.goal_pose.y - res.centroid.y, res.goal_pose.x - res.centroid.x);
      }
      // Get vector through centroid and determine start point and distance
      Eigen::Vector3f push_unit_vec(std::cos(p.push_angle), std::sin(p.push_angle), 0.0f);
      std::vector<pcl16::PointXYZ> end_points = pcl_segmenter_->lineCloudIntersectionEndPoints(
          cur_obj.cloud, push_unit_vec, cur_obj.centroid);
      p.start_point.x = end_points[0].x;
      p.start_point.y = end_points[0].y;
      p.start_point.z = end_points[0].z;
      // Get push distance
      p.push_dist = hypot(res.centroid.x - res.goal_pose.x, res.centroid.y - res.goal_pose.y);
      timing_push_ = false;
    }
    // Visualize push vector
    PointStamped obj_centroid;
    obj_centroid.header.frame_id = workspace_frame_;
    obj_centroid.point = res.centroid;
    if (spin_push)
    {
      displayGoalHeading(cur_color_frame_, obj_centroid, cur_state.x.theta, res.goal_pose.theta);
    }
    else
    {
      PointStamped start_point;
      start_point.header.frame_id = workspace_frame_;
      start_point.point = p.start_point;
      PointStamped end_point;
      end_point.header.frame_id = workspace_frame_;
      end_point.point.x = res.goal_pose.x;
      end_point.point.y = res.goal_pose.y;
      end_point.point.z = start_point.point.z;
      displayPushVector(cur_color_frame_, start_point, end_point);
      displayInitialPushVector(cur_color_frame_, start_point, end_point, obj_centroid);
    }

    // Cleanup and return
    ROS_INFO_STREAM("Chosen push start point: (" << p.start_point.x << ", " << p.start_point.y << ", " <<
                    p.start_point.z << ")");
    ROS_INFO_STREAM("Push dist: " << p.push_dist);
    ROS_INFO_STREAM("Push angle: " << p.push_angle << "\n");
    learn_callback_count_++;
    res.push = p;
    tracker_goal_pose_ = res.goal_pose;
    return res;
  }

  /**
   * Method to determine which pushing location to choose as a function of current object shape descriptors and history
   *
   * @param cur_obj Object for which we are choosing the push location
   * @param cur_state Current state information of the object
   * @param new_object Whether this object is new or has a history
   * @param num_clusters Number of push clusters to use in finding push locations
   *
   * @return The the location and descriptor of the push location
   */
  ShapeLocation choosePushStartLoc(ProtoObject& cur_obj, PushTrackerState& cur_state, bool new_object, int num_clusters)
  {
    if (new_object)
    {
      start_loc_history_.clear();
    }
    // Get shape features and associated locations
    ShapeLocations locs = tabletop_pushing::extractObjectShapeContext(cur_obj, use_center_pointing_shape_context_);
    // tabletop_pushing::extractObjectShapeContext(cur_obj, !use_center_pointing_shape_context_);

    // Choose location index from features and history
    int loc_idx = 0;
    if (start_loc_history_.size() == 0)
    {
      // TODO: Improve this to find a more unique / prototypical point?
      loc_idx = rand() % locs.size();
    }
    else
    {
      // Cluster locs based on shape similarity
      std::vector<int> cluster_ids;
      ShapeDescriptors centers;
      double min_err_change = 0.001;
      int max_iter = 1000;
      tabletop_pushing::clusterShapeFeatures(locs, num_clusters, cluster_ids, centers, min_err_change, max_iter);

      // Display the boundary locations colored by their cluster IDs
      cv::Mat boundary_disp_img(cur_color_frame_.size(), CV_32FC3, cv::Scalar(0,0,0));
      for (unsigned int i = 0; i < locs.size(); ++i)
      {
        const cv::Point2f img_idx = pcl_segmenter_->projectPointIntoImage(
            locs[i].boundary_loc_, cur_obj.cloud.header.frame_id, camera_frame_);
        boundary_disp_img.at<cv::Vec3f>(img_idx.y, img_idx.x) = pcl_segmenter_->colors_[cluster_ids[i]];
      }
      cv::imshow("Cluster colors", boundary_disp_img);

      // TODO: Easier to just keep picking random locs and choose first one with unused cluster center?
      // Find which clusters the previous choices map too, pick one other than those randomly
      std::vector<int> used_clusters;
      for (int i = 0; i < start_loc_history_.size(); ++i)
      {
        double cluster_dist = 0;
        int closest = tabletop_pushing::closestShapeFeatureCluster(start_loc_history_[i].descriptor_, centers,
                                                                   cluster_dist);
        used_clusters.push_back(closest);
      }
      bool done = false;
      int rand_cluster = -1;
      while (!done)
      {
        rand_cluster = rand() % num_clusters;
        done = true;
        for (int i = 0; i < used_clusters.size(); ++i)
        {
          if (used_clusters[i] == rand_cluster)
          {
            done = false;
          }
        }
      }
      ROS_INFO_STREAM("Chose cluster " << rand_cluster);
      // Pick random loc that has cluster id rand_cluster
      std::vector<int> loc_choices;
      for (int l = 0; l < locs.size(); ++l)
      {
        if (cluster_ids[l] == rand_cluster)
        {
          loc_choices.push_back(l);
        }
      }
      int choice_idx = rand()%loc_choices.size();
      loc_idx = loc_choices[choice_idx];
    }
    // Transform location into object frame for storage in history
    ShapeLocation s(worldPointInObjectFrame(locs[loc_idx].boundary_loc_, cur_state),
                    locs[loc_idx].descriptor_);
    start_loc_history_.push_back(s);
    return locs[loc_idx];
  }

  /**
   * Method to choose an initial pushing location at a specified percentage around the object boundary with 0 distance
   * on the boundary at the dominatnt orientation of the object.
   *
   * @param cur_obj object model of current frame
   * @param cur_state state estimate of object
   * @param new_object switch if we are initialzing on a new object
   * @param num_start_loc_pushes_per_sample The number of samples to attempt at each push locations
   * @param num_start_loc_sample_locs The number of pushing locations on the boundary to sample
   *
   * @return The location and shape descriptor on the boundary to place the hand
   */
  ShapeLocation chooseFixedGoalPushStartLoc(ProtoObject& cur_obj, PushTrackerState& cur_state, bool new_object,
                                            int num_start_loc_pushes_per_sample, int num_start_loc_sample_locs,
                                            std::string trial_id)
  {
    float hull_alpha = 0.01;
    XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha);

    int rot_idx = -1;
    if (new_object)
    {
      // Set new object model
      start_loc_obj_ = cur_obj;
      // Reset boundary traversal data
      start_loc_arc_length_percent_ = 0.0;
      start_loc_push_sample_count_ = 0;
      start_loc_history_.clear();

      // NOTE: Initial start location is the dominant orientation
      ROS_INFO_STREAM("Current state theta is: " << cur_state.x.theta);
      double min_angle_dist = FLT_MAX;
      for (int i = 0; i < hull_cloud.size(); ++i)
      {
        double theta_i = atan2(hull_cloud.at(i).y - cur_state.x.y, hull_cloud.at(i).x - cur_state.x.x);
        double angle_dist_i = fabs(subPIAngle(theta_i - cur_state.x.theta));
        if (angle_dist_i < min_angle_dist)
        {
          min_angle_dist = angle_dist_i;
          rot_idx = i;
        }
      }
    }
    else
    {
      // Increment boundary location if necessary
      if (start_loc_history_.size() % num_start_loc_pushes_per_sample == 0)
      {
        start_loc_arc_length_percent_ += 1.0/num_start_loc_sample_locs;
        ROS_INFO_STREAM("Incrementing arc length percent based on: " << num_start_loc_pushes_per_sample);
      }

      // Get initial object boundary location in the current world frame
      ROS_INFO_STREAM("init_obj_point: " << start_loc_history_[0].boundary_loc_);
      pcl16::PointXYZ init_loc_point = objectPointInWorldFrame(start_loc_history_[0].boundary_loc_, cur_state);
      ROS_INFO_STREAM("init_loc_point: " << init_loc_point);

      // Find index of closest point on current boundary to the initial pushing location
      double min_dist = FLT_MAX;
      for (int i = 0; i < hull_cloud.size(); ++i)
      {
        double dist_i = pcl_segmenter_->sqrDist(init_loc_point, hull_cloud.at(i));
        if (dist_i < min_dist)
        {
          min_dist = dist_i;
          rot_idx = i;
        }
      }
    }
    // Test hull_cloud orientation, reverse iteration if it is negative
    double pt0_theta = atan2(hull_cloud[rot_idx].y - cur_state.x.y, hull_cloud[rot_idx].x - cur_state.x.x);
    int pt1_idx = (rot_idx+1) % hull_cloud.size();
    double pt1_theta = atan2(hull_cloud[pt1_idx].y - cur_state.x.y, hull_cloud[pt1_idx].x - cur_state.x.x);
    bool reverse_data = false;
    if (subPIAngle(pt1_theta - pt0_theta) < 0)
    {
      reverse_data = true;
      ROS_INFO_STREAM("Reversing data for boundaries");
    }

    // Compute cumulative distance around the boundary at each point
    std::vector<double> boundary_dists(hull_cloud.size(), 0.0);
    double boundary_length = 0.0;
    ROS_INFO_STREAM("rot_idx is " << rot_idx);
    for (int i = 1; i <= hull_cloud.size(); ++i)
    {
      int idx0 = (rot_idx+i-1) % hull_cloud.size();
      int idx1 = (rot_idx+i) % hull_cloud.size();
      if (reverse_data)
      {
        idx0 = (hull_cloud.size()+rot_idx-i+1) % hull_cloud.size();
        idx1 = (hull_cloud.size()+rot_idx-i) % hull_cloud.size();
      }
      // NOTE: This makes boundary_dists[rot_idx] = 0.0, and we have no location at 100% the boundary_length
      boundary_dists[idx0] = boundary_length;
      double loc_dist = pcl_segmenter_->dist(hull_cloud[idx0], hull_cloud[idx1]);
      boundary_length += loc_dist;
    }

    // Find location at start_loc_arc_length_percent_ around the boundary
    double desired_boundary_dist = start_loc_arc_length_percent_*boundary_length;
    ROS_INFO_STREAM("Finding location at dist " << desired_boundary_dist << " ~= " << start_loc_arc_length_percent_*100 <<
                    "\% of " << boundary_length);
    int boundary_loc_idx;
    double min_boundary_dist_diff = FLT_MAX;
    for (int i = 0; i < hull_cloud.size(); ++i)
    {
      double boundary_dist_diff_i = fabs(desired_boundary_dist - boundary_dists[i]);
      if (boundary_dist_diff_i < min_boundary_dist_diff)
      {
        min_boundary_dist_diff = boundary_dist_diff_i;
        boundary_loc_idx = i;
      }
    }
    ROS_INFO_STREAM("Chose location at idx: " << boundary_loc_idx << " with diff " << min_boundary_dist_diff);
    // Get descriptor at the chosen location
    // ShapeLocations locs = tabletop_pushing::extractShapeContextFromSamples(hull_cloud, cur_obj,
    //                                                                         use_center_pointing_shape_context_);
    float gripper_spread = 0.05;
    pcl16::PointXYZ boundary_loc = hull_cloud[boundary_loc_idx];
    ShapeDescriptor sd = tabletop_pushing::extractLocalAndGlobalShapeFeatures(hull_cloud, cur_obj, boundary_loc,
                                                                              boundary_loc_idx, gripper_spread,
                                                                              hull_alpha, point_cloud_hist_res_);
    // Add into pushing history in object frame
    // ShapeLocation s(worldPointInObjectFrame(locs[boundary_loc_idx].boundary_loc_, cur_state),
    //                 locs[boundary_loc_idx].descriptor_);
    // start_loc_history_.push_back(s);
    ShapeLocation s_obj(worldPointInObjectFrame(boundary_loc, cur_state), sd);
    start_loc_history_.push_back(s_obj);

    // TODO: Project desired outline to show where to place object before pushing?
    cv::Mat hull_img(cur_color_frame_.size(), CV_8UC1, cv::Scalar(0));
    pcl_segmenter_->projectPointCloudIntoImage(hull_cloud, hull_img);
    hull_img*=255;
    cv::Mat hull_disp_img(hull_img.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::cvtColor(hull_img, hull_disp_img, CV_GRAY2BGR);
    cv::Point2f img_rot_idx = pcl_segmenter_->projectPointIntoImage(hull_cloud[rot_idx], hull_cloud.header.frame_id,
                                                                    camera_frame_);
    cv::circle(hull_disp_img, img_rot_idx, 4, cv::Scalar(0,255,0), 3);
    cv::Point2f img_loc_idx = pcl_segmenter_->projectPointIntoImage(boundary_loc,
                                                                    hull_cloud.header.frame_id, camera_frame_);
    cv::circle(hull_disp_img, img_loc_idx, 4, cv::Scalar(0,0, 255));
    cv::imshow("object hull", hull_disp_img);
    if (write_to_disk_)
    {
      std::stringstream hull_file_name;
      hull_file_name << base_output_path_ << trial_id << "_obj_hull.png";
      std::stringstream hull_disp_file_name;
      hull_disp_file_name << base_output_path_ << trial_id << "_obj_hull_disp.png";
      cv::imwrite(hull_file_name.str(), hull_img);
      cv::imwrite(hull_disp_file_name.str(), hull_disp_img);
    }
    ShapeLocation s_world(boundary_loc, sd);
    return s_world;
    // return locs[boundary_loc_idx];
  }

  ShapeLocation chooseLearnedPushStartLoc(ProtoObject& cur_obj, PushTrackerState& cur_state, std::string param_path,
                                          float& chosen_score)
  {
    // Get features for all of the boundary locations
    float hull_alpha = 0.01;
    XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha);
    float gripper_spread = 0.05;
    ShapeDescriptors sds = tabletop_pushing::extractLocalAndGlobalShapeFeatures(hull_cloud, cur_obj,
                                                                                gripper_spread, hull_alpha,
                                                                                point_cloud_hist_res_);
    // Set parameters for prediction
    // svm_parameter push_parameters;
    // push_parameters.svm_type = EPSILON_SVR;
    // push_parameters.kernel_type = PRECOMPUTED;
    // push_parameters.C = 2.0; // NOTE: only needed for training
    // push_parameters.p = 0.3; // NOTE: only needed for training
    // push_model.param = push_parameters;

    // TODO: Read in model SVs and coefficients
    svm_model* push_model;
    push_model = svm_load_model(param_path.c_str());

    std::vector<double> pred_push_scores;
    double best_score = FLT_MAX;
    int best_idx = -1;
    // TODO: Perform prediction at all sample locations
    for (int i = 0; i < sds.size(); ++i)
    {
      // TODO: Set the data vector
      svm_node x;
      double pred_log_score = svm_predict(push_model, &x);
      double pred_score = exp(pred_log_score);
      if (pred_score < best_score)
      {
        best_score = pred_score;
        best_idx = i;
      }
      pred_push_scores.push_back(pred_score);
    }

    // TODO: Return the location of the best score
    ShapeLocation loc;
    if (best_idx >= 0)
    {
    }
    return loc;
  }

  float getSpinPushHeading(PushTrackerState& cur_state, ShapeLocation& chosen_loc)
  {
    // Get chosen_loc angle in object frame
    pcl16::PointXYZ obj_pt = worldPointInObjectFrame(chosen_loc.boundary_loc_, cur_state);
    float phi = atan2(obj_pt.y, obj_pt.x);
    // Choose pushing direction based on angular position in object frame
    float push_angle_obj_frame;
    if ( -0.25*M_PI < phi && phi <= 0.25*M_PI)
    {
      push_angle_obj_frame = M_PI;
    }
    else if ( -0.75*M_PI < phi && phi <= -0.25*M_PI)
    {
      push_angle_obj_frame = 0.5*M_PI;
    }
    else if ( 0.25*M_PI < phi && phi <= 0.75*M_PI)
    {
      push_angle_obj_frame = -0.5*M_PI;
    }
    else if (phi <= -0.75*M_PI || phi > 0.75*M_PI)
    {
      push_angle_obj_frame = 0;
    }
    // Shift push direction into world frame
    float push_angle_world_frame = push_angle_obj_frame + cur_state.x.theta;
    ROS_INFO_STREAM("Object pose is (" << cur_state.x.x << ", " << cur_state.x.y << ", " << cur_state.x.theta << ")");
    ROS_INFO_STREAM("phi is: " << (phi));
    ROS_INFO_STREAM("push_angle_obj_frame is: " << (push_angle_obj_frame));
    ROS_INFO_STREAM("push_angle_world_frame is: " << (push_angle_world_frame));
    return push_angle_world_frame;
  }

  ShapeLocation getStartLocDescriptor(ProtoObject& cur_obj, PushTrackerState& cur_state, geometry_msgs::Point start_pt)
  {
    // Get shape features and associated locations
    ShapeLocations locs = tabletop_pushing::extractObjectShapeContext(cur_obj, use_center_pointing_shape_context_);
    // Find location closest to the chosen start point
    float min_dist = FLT_MAX;
    unsigned int loc_idx = locs.size();
    for (unsigned int i = 0; i < locs.size(); ++i)
    {
      float loc_dist = pcl_segmenter_->dist(locs[i].boundary_loc_, start_pt);
      if (loc_dist < min_dist)
      {
        min_dist = loc_dist;
        loc_idx = i;
      }
    }
    ROS_WARN_STREAM("Chose loc " << locs[loc_idx].boundary_loc_ << " with distance " << min_dist << "m");
    return locs[loc_idx];
  }

  pcl16::PointXYZ worldPointInObjectFrame(pcl16::PointXYZ world_pt, PushTrackerState& cur_state)
  {
    // Center on object frame
    pcl16::PointXYZ shifted_pt;
    shifted_pt.x = world_pt.x - cur_state.x.x;
    shifted_pt.y = world_pt.y - cur_state.x.y;
    shifted_pt.z = world_pt.z - cur_state.z;
    double ct = cos(cur_state.x.theta);
    double st = sin(cur_state.x.theta);
    // Rotate into correct frame
    pcl16::PointXYZ obj_pt;
    obj_pt.x =  ct*shifted_pt.x + st*shifted_pt.y;
    obj_pt.y = -st*shifted_pt.x + ct*shifted_pt.y;
    obj_pt.z = shifted_pt.z; // NOTE: Currently assume 2D motion
    return obj_pt;
  }

  pcl16::PointXYZ objectPointInWorldFrame(pcl16::PointXYZ obj_pt, PushTrackerState& cur_state)
  {
    // Rotate out of object frame
    pcl16::PointXYZ rotated_pt;
    double ct = cos(cur_state.x.theta);
    double st = sin(cur_state.x.theta);
    rotated_pt.x = ct*obj_pt.x - st*obj_pt.y;
    rotated_pt.y = st*obj_pt.x + ct*obj_pt.y;
    rotated_pt.z = obj_pt.z;  // NOTE: Currently assume 2D motion
    // Shift to world frame
    pcl16::PointXYZ world_pt;
    world_pt.x = rotated_pt.x + cur_state.x.x;
    world_pt.y = rotated_pt.y + cur_state.x.y;
    world_pt.z = rotated_pt.z + cur_state.z;
    return world_pt;
  }

  LearnPush::Response getObjectPose()
  {
    bool no_objects = false;
    ProtoObject cur_obj = obj_tracker_->findTargetObject(cur_color_frame_,
                                                         cur_self_filtered_cloud_,
                                                         no_objects);
    LearnPush::Response res;
    if (no_objects)
    {
      ROS_WARN_STREAM("No objects found on analysis");
      res.centroid.x = 0.0;
      res.centroid.y = 0.0;
      res.centroid.z = 0.0;
      res.theta = 0.0;
      res.no_objects = true;
      return res;
    }
    cv::RotatedRect obj_ellipse = obj_tracker_->fitObjectEllipse(cur_obj);
    res.no_objects = false;
    res.centroid.x = cur_obj.centroid[0];
    res.centroid.y = cur_obj.centroid[1];
    res.centroid.z = cur_obj.centroid[2];
    res.theta = obj_tracker_->getThetaFromEllipse(obj_ellipse);

    return res;
  }

  PushTrackerState startTracking(bool start_swap=false)
  {
    ROS_INFO_STREAM("Starting tracker");
    frame_set_count_++;
    goal_out_count_ = 0;
    goal_heading_count_ = 0;
    frame_callback_count_ = 0;
    PoseStamped arm_pose;
    if (pushing_arm_ == "l")
    {
      arm_pose = l_arm_pose_;
    }
    else
    {
      arm_pose = r_arm_pose_;
    }
    return obj_tracker_->initTracks(cur_color_frame_, cur_self_mask_, cur_self_filtered_cloud_,
                                    proxy_name_, arm_pose, tool_proxy_name_, start_swap);
  }

  void lArmStateCartCB(const pr2_manipulation_controllers::JTTaskControllerState l_arm_state)
  {
    l_arm_pose_ = l_arm_state.x;
    l_arm_vel_ = l_arm_state.xd;
  }
  void rArmStateCartCB(const pr2_manipulation_controllers::JTTaskControllerState r_arm_state)
  {
    r_arm_pose_ = r_arm_state.x;
    r_arm_vel_ = r_arm_state.xd;
  }
  void lArmStateVelCB(const pr2_manipulation_controllers::JinvTeleopControllerState l_arm_state)
  {
    l_arm_pose_ = l_arm_state.x;
    l_arm_vel_ = l_arm_state.xd;
  }
  void rArmStateVelCB(const pr2_manipulation_controllers::JinvTeleopControllerState r_arm_state)
  {
    r_arm_pose_ = r_arm_state.x;
    r_arm_vel_ = r_arm_state.xd;
  }

  void pushTrackerGoalCB()
  {
    ROS_INFO_STREAM("Accepting goal");
    shared_ptr<const PushTrackerGoal> tracker_goal = as_.acceptNewGoal();
    tracker_goal_pose_ = tracker_goal->desired_pose;
    pushing_arm_ = tracker_goal->which_arm;
    controller_name_ = tracker_goal->controller_name;
    proxy_name_ = tracker_goal->proxy_name;
    behavior_primitive_ = tracker_goal->behavior_primitive;
    tool_proxy_name_ = tracker_goal->tool_proxy_name;
    ROS_INFO_STREAM("Accepted goal of " << tracker_goal_pose_);
    gripper_not_moving_count_ = 0;
    object_not_moving_count_ = 0;
    object_not_detected_count_ = 0;
    object_too_far_count_ = 0;
    object_not_between_count_ = 0;
    push_start_time_ = ros::Time::now().toSec();

    if (obj_tracker_->isInitialized())
    {
      obj_tracker_->unpause();
    }
    else
    {
      startTracking();
    }
  }

  void pushTrackerPreemptCB()
  {
    obj_tracker_->pause();
    ROS_INFO_STREAM("Preempted push tracker");
    // set the action state to preempted
    as_.setPreempted();
  }

  bool objectNotMoving(PushTrackerState& tracker_state)
  {
    if (tracker_state.x_dot.x < object_not_moving_thresh_ &&
        tracker_state.x_dot.y < object_not_moving_thresh_)
    {
      ++object_not_moving_count_;
    }
    else
    {
      object_not_moving_count_ = 0;
    }
    return object_not_moving_count_  >= object_not_moving_count_limit_;
  }

  bool objectDisappeared(PushTrackerState& tracker_state)
  {
    if (tracker_state.no_detection)
    {
      ++object_not_detected_count_;
    }
    else
    {
      object_not_detected_count_ = 0;
    }
    return object_not_detected_count_  >= object_not_detected_count_limit_;
  }

  bool gripperNotMoving()
  {
    Twist gripper_vel;
    if ( pushing_arm_ == "l")
    {
      gripper_vel = l_arm_vel_;
    }
    else
    {
      gripper_vel = r_arm_vel_;
    }
    if (gripper_vel.linear.x < gripper_not_moving_thresh_ &&
        gripper_vel.linear.y < gripper_not_moving_thresh_)
    {
      ++gripper_not_moving_count_;
    }
    else
    {
      gripper_not_moving_count_ = 0;
    }
    return gripper_not_moving_count_  >= gripper_not_moving_count_limit_;
  }

  bool objectNotBetweenGoalAndGripper(Pose2D& obj_state)
  {
    if (pushing_arm_ == "l")
    {
      if( pointIsBetweenOthers(l_arm_pose_.pose.position, obj_state, tracker_goal_pose_,
                               object_not_between_epsilon_))
      {
        ++object_not_between_count_;
      }
      else
      {
        object_not_between_count_ = 0;
      }
    }
    else if (pushing_arm_ == "r")
    {
      if( pointIsBetweenOthers(r_arm_pose_.pose.position, obj_state, tracker_goal_pose_,
                               object_not_between_epsilon_))
      {
        ++object_not_between_count_;
      }
      else
      {
        object_not_between_count_ = 0;
      }
    }
    return object_not_between_count_ >= object_not_between_count_limit_;
  }

  bool objectNotBetweenGoalAndTool(Pose2D& obj_state, PoseStamped& tool_state)
  {
    if( pointIsBetweenOthers(tool_state.pose.position, obj_state, tracker_goal_pose_,
                             object_not_between_tool_epsilon_))
    {
      ++object_not_between_count_;
    }
    else
    {
      object_not_between_count_ = 0;
    }
    return object_not_between_count_ >= object_not_between_count_limit_;
  }

  // TODO: Make this threshold the initial distance when pushing + some epsilon
  bool objectTooFarFromGripper(Pose2D& obj_state)
  {
    geometry_msgs::Point gripper_pt;
    if (pushing_arm_ == "l")
    {
      gripper_pt = l_arm_pose_.pose.position;
    }
    else
    {
      gripper_pt = r_arm_pose_.pose.position;
    }
    float gripper_dist = hypot(gripper_pt.x-obj_state.x,gripper_pt.y-obj_state.y);
    if (gripper_dist  > max_object_gripper_dist_)
    {
      ++object_too_far_count_;
    }
    else
    {
      object_too_far_count_ = 0;
    }
    return object_too_far_count_ >= object_too_far_count_limit_;
  }

  bool objectTooFarFromTool(Pose2D& obj_state, PoseStamped& tool_state)
  {
    geometry_msgs::Point tool_pt;
    tool_pt = tool_state.pose.position;
    float tool_dist = hypot(tool_pt.x-obj_state.x, tool_pt.y-obj_state.y);
    if (tool_dist  > max_object_tool_dist_)
    {
      ++object_too_far_count_;
    }
    else
    {
      object_too_far_count_ = 0;
    }
    return object_too_far_count_ >= object_too_far_count_limit_;
  }

  bool pushingTimeUp()
  {
    if (ros::Time::now().toSec() - push_start_time_ > start_loc_push_time_)
    {
      return true;
    }
    return false;
  }

  bool pointIsBetweenOthers(geometry_msgs::Point pt, Pose2D& x1, Pose2D& x2, double epsilon=0.0)
  {
    // Project the vector pt->x2 onto the vector x1->x2
    const float a_x = x2.x - pt.x;
    const float a_y = x2.y - pt.y;
    const float b_x = x2.x - x1.x;
    const float b_y = x2.y - x1.y;
    const float a_dot_b = a_x*b_x + a_y*b_y;
    const float b_dot_b = b_x*b_x + b_y*b_y;
    const float a_onto_b = a_dot_b/b_dot_b;

    // If the (squared) distance of the projection is less than the vector from x1->x2 then it is between them
    const float d_1_x = a_onto_b*b_x;
    const float d_1_y = a_onto_b*b_y;
    const float d_1 = d_1_x*d_1_x + d_1_y*d_1_y;
    const float d_2 = b_x*b_x + b_y*b_y;

    // NOTE: Add epsilon squared distance to the projected distance to allow for small noise
    return d_1+epsilon*epsilon < d_2;
  }

  /**
   * ROS Service callback method for determining the location of a table in the
   * scene
   *
   * @param req The service request
   * @param res The service response
   *
   * @return true if successfull, false otherwise
   */
  bool getTableLocation(LocateTable::Request& req, LocateTable::Response& res)
  {
    if ( have_depth_data_ )
    {
      res.table_centroid = getTablePlane(cur_point_cloud_);
      if ((res.table_centroid.pose.position.x == 0.0 &&
           res.table_centroid.pose.position.y == 0.0 &&
           res.table_centroid.pose.position.z == 0.0) ||
          res.table_centroid.pose.position.x < 0.0)
      {
        ROS_ERROR_STREAM("No plane found, leaving");
        res.found_table = false;
        return false;
      }
      res.found_table = true;
      res.table_centroid.header.stamp = ros::Time::now();
    }
    else
    {
      ROS_ERROR_STREAM("Calling getTableLocation prior to receiving sensor data.");
      res.found_table = false;
      return false;
    }
    return true;
  }

  /**
   * Calculate the location of the dominant plane (table) in a point cloud
   *
   * @param cloud The point cloud containing a table
   *
   * @return The estimated 3D centroid of the table
   */
  PoseStamped getTablePlane(XYZPointCloud& cloud)
  {
    XYZPointCloud obj_cloud, table_cloud;
    // TODO: Comptue the hull on the first call
    Eigen::Vector4f table_centroid = pcl_segmenter_->getTablePlane(cloud,
                                                                   obj_cloud,
                                                                   table_cloud);
    PoseStamped p;
    p.pose.position.x = table_centroid[0];
    p.pose.position.y = table_centroid[1];
    p.pose.position.z = table_centroid[2];
    p.header = cloud.header;
    ROS_INFO_STREAM("Table centroid is: ("
                    << p.pose.position.x << ", "
                    << p.pose.position.y << ", "
                    << p.pose.position.z << ")");
    table_centroid_ = p;
    return p;
  }

  void displayPushVector(cv::Mat& img, PointStamped& start_point, PointStamped& end_point,
                         std::string display_name="goal_vector", bool force_no_write=false)
  {
    cv::Mat disp_img;
    img.copyTo(disp_img);

    cv::Point img_start_point = pcl_segmenter_->projectPointIntoImage(start_point);
    cv::Point img_end_point = pcl_segmenter_->projectPointIntoImage(end_point);
    cv::line(disp_img, img_start_point, img_end_point, cv::Scalar(0,0,0),3);
    cv::line(disp_img, img_start_point, img_end_point, cv::Scalar(0,255,0));
    cv::circle(disp_img, img_end_point, 4, cv::Scalar(0,0,0),3);
    cv::circle(disp_img, img_end_point, 4, cv::Scalar(0,255,0));
    if (use_displays_)
    {
      cv::imshow(display_name, disp_img);
    }
    if (write_to_disk_ && !force_no_write)
    {
      // Write to disk to create video output
      std::stringstream push_out_name;
      push_out_name << base_output_path_ << display_name << "_" << frame_set_count_ << "_"
                    << goal_out_count_++ << ".png";
      cv::imwrite(push_out_name.str(), disp_img);
    }
  }

  void displayInitialPushVector(cv::Mat& img, PointStamped& start_point, PointStamped& end_point,
                                PointStamped& centroid)
  {
    cv::Mat disp_img;
    img.copyTo(disp_img);

    cv::Point img_start_point = pcl_segmenter_->projectPointIntoImage(start_point);
    cv::Point img_end_point = pcl_segmenter_->projectPointIntoImage(end_point);
    cv::line(disp_img, img_start_point, img_end_point, cv::Scalar(0,0,0),3);
    cv::line(disp_img, img_start_point, img_end_point, cv::Scalar(0,255,0));
    cv::circle(disp_img, img_end_point, 4, cv::Scalar(0,0,0),3);
    cv::circle(disp_img, img_end_point, 4, cv::Scalar(0,255,0));
    cv::Point img_centroid_point = pcl_segmenter_->projectPointIntoImage(centroid);
    cv::circle(disp_img, img_centroid_point, 4, cv::Scalar(0,0,0),3);
    cv::circle(disp_img, img_centroid_point, 4, cv::Scalar(0,0,255));
    cv::imshow("initial_vector", disp_img);
  }


  void displayGoalHeading(cv::Mat& img, PointStamped& centroid, double theta, double goal_theta)
  {
    cv::Mat disp_img;
    img.copyTo(disp_img);

    cv::Point img_c_idx = pcl_segmenter_->projectPointIntoImage(centroid);
    cv::RotatedRect obj_ellipse = obj_tracker_->getMostRecentEllipse();
    const float x_maj_rad = (std::cos(theta)*obj_ellipse.size.height*0.5);
    const float y_maj_rad = (std::sin(theta)*obj_ellipse.size.height*0.5);
    pcl16::PointXYZ table_heading_point(centroid.point.x+x_maj_rad, centroid.point.y+y_maj_rad,
                                      centroid.point.z);
    const cv::Point2f img_maj_idx = pcl_segmenter_->projectPointIntoImage(
        table_heading_point, centroid.header.frame_id, camera_frame_);
    const float goal_x_rad = (std::cos(goal_theta)*obj_ellipse.size.height*0.5);
    const float goal_y_rad = (std::sin(goal_theta)*obj_ellipse.size.height*0.5);
    pcl16::PointXYZ goal_heading_point(centroid.point.x+goal_x_rad, centroid.point.y+goal_y_rad,
                                     centroid.point.z);
    cv::Point2f img_goal_idx = pcl_segmenter_->projectPointIntoImage(
        goal_heading_point, centroid.header.frame_id, camera_frame_);
    // TODO: Draw partially shaded ellipse showing angle error
    float img_start_angle = RAD2DEG(std::atan2(img_maj_idx.y-img_c_idx.y,
                                               img_maj_idx.x-img_c_idx.x));
    float img_end_angle = RAD2DEG(std::atan2(img_goal_idx.y-img_c_idx.y,
                                             img_goal_idx.x-img_c_idx.x));
    double img_height = std::sqrt(std::pow(img_maj_idx.x-img_c_idx.x,2) +
                                  std::pow(img_maj_idx.y-img_c_idx.y,2));
    // NOTE: Renormalize goal idx positiong to be on a circle with current heading point
    cv::Point img_goal_draw_idx(std::cos(DEG2RAD(img_end_angle))*img_height+img_c_idx.x,
                                std::sin(DEG2RAD(img_end_angle))*img_height+img_c_idx.y);
    cv::Size axes(img_height, img_height);
    // TODO: Draw backgrounds in black
    cv::ellipse(disp_img, img_c_idx, axes, img_start_angle, 0,
                RAD2DEG(subPIAngle(DEG2RAD(img_end_angle-img_start_angle))), cv::Scalar(0,0,0), 3);
    cv::line(disp_img, img_c_idx, img_maj_idx, cv::Scalar(0,0,0), 3);
    cv::line(disp_img, img_c_idx, img_goal_draw_idx, cv::Scalar(0,0,0), 3);
    cv::ellipse(disp_img, img_c_idx, axes, img_start_angle, 0,
                RAD2DEG(subPIAngle(DEG2RAD(img_end_angle-img_start_angle))), cv::Scalar(0,0,255));
    cv::line(disp_img, img_c_idx, img_maj_idx, cv::Scalar(0,0,255));
    cv::line(disp_img, img_c_idx, img_goal_draw_idx, cv::Scalar(0,0,255));
    if (use_displays_)
    {
      cv::imshow("goal_heading", disp_img);
    }
    if (write_to_disk_)
    {
      // Write to disk to create video output
      std::stringstream push_out_name;
      push_out_name << base_output_path_ << "goal_heading_" << frame_set_count_ << "_"
                    << goal_heading_count_++ << ".png";
      cv::imwrite(push_out_name.str(), disp_img);
    }
  }

  void displayRobotGripperPoses(cv::Mat& img)
  {
    cv::Mat disp_img;
    img.copyTo(disp_img);

    PointStamped l_gripper;
    l_gripper.header.frame_id = "torso_lift_link";
    l_gripper.header.stamp = ros::Time(0);
    l_gripper.point = l_arm_pose_.pose.position;
    PointStamped r_gripper;
    r_gripper.header.frame_id = "torso_lift_link";
    r_gripper.header.stamp = ros::Time(0);
    r_gripper.point = r_arm_pose_.pose.position;
    cv::Point l_gripper_img_point = pcl_segmenter_->projectPointIntoImage(l_gripper);
    cv::Point r_gripper_img_point = pcl_segmenter_->projectPointIntoImage(r_gripper);
    cv::circle(disp_img, l_gripper_img_point, 4, cv::Scalar(0,0,0),3);
    cv::circle(disp_img, l_gripper_img_point, 4, cv::Scalar(0,255,0));
    cv::circle(disp_img, r_gripper_img_point, 4, cv::Scalar(0,0,0),3);
    cv::circle(disp_img, r_gripper_img_point, 4, cv::Scalar(0,0,255));

    if (use_displays_)
    {
      cv::imshow("grippers", disp_img);
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
  message_filters::Subscriber<sensor_msgs::Image> mask_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
  message_filters::Synchronizer<MySyncPolicy> sync_;
  sensor_msgs::CameraInfo cam_info_;
  shared_ptr<tf::TransformListener> tf_;
  ros::ServiceServer push_pose_server_;
  ros::ServiceServer table_location_server_;
  ros::Subscriber jinv_l_arm_subscriber_;
  ros::Subscriber jinv_r_arm_subscriber_;
  ros::Subscriber jtteleop_l_arm_subscriber_;
  ros::Subscriber jtteleop_r_arm_subscriber_;
  actionlib::SimpleActionServer<PushTrackerAction> as_;
  cv::Mat cur_color_frame_;
  cv::Mat cur_depth_frame_;
  cv::Mat cur_self_mask_;
  cv::Mat prev_color_frame_;
  cv::Mat prev_depth_frame_;
  cv::Mat prev_self_mask_;
  std_msgs::Header cur_camera_header_;
  std_msgs::Header prev_camera_header_;
  XYZPointCloud cur_point_cloud_;
  XYZPointCloud cur_self_filtered_cloud_;
  shared_ptr<PointCloudSegmentation> pcl_segmenter_;
  bool have_depth_data_;
  int display_wait_ms_;
  bool use_displays_;
  bool write_input_to_disk_;
  bool write_to_disk_;
  std::string base_output_path_;
  int num_downsamples_;
  std::string workspace_frame_;
  std::string camera_frame_;
  PoseStamped table_centroid_;
  bool camera_initialized_;
  std::string cam_info_topic_;
  bool start_tracking_on_push_call_;
  bool recording_input_;
  int record_count_;
  int learn_callback_count_;
  int goal_out_count_;
  int goal_heading_count_;
  int frame_callback_count_;
  shared_ptr<ObjectTracker25D> obj_tracker_;
  Pose2D tracker_goal_pose_;
  std::string pushing_arm_;
  std::string proxy_name_;
  std::string controller_name_;
  std::string behavior_primitive_;
  std::string tool_proxy_name_;
  double tracker_dist_thresh_;
  double tracker_angle_thresh_;
  bool just_spun_;
  double major_axis_spin_pos_scale_;
  int frame_set_count_;
  PoseStamped l_arm_pose_;
  PoseStamped r_arm_pose_;
  Twist l_arm_vel_;
  Twist r_arm_vel_;
  bool use_mps_segmentation_;
  double max_object_gripper_dist_;
  double max_object_tool_dist_;
  double object_not_moving_thresh_;
  int object_not_moving_count_;
  int object_not_moving_count_limit_;
  double gripper_not_moving_thresh_;
  int gripper_not_moving_count_;
  int gripper_not_moving_count_limit_;
  int object_not_detected_count_;
  int object_not_detected_count_limit_;
  int object_too_far_count_;
  int object_too_far_count_limit_;
  int object_not_between_count_;
  int object_not_between_count_limit_;
  double object_not_between_epsilon_;
  double object_not_between_tool_epsilon_;
  ShapeLocations start_loc_history_;
  double start_loc_push_time_;
  double start_loc_push_dist_;
  double push_start_time_;
  bool timing_push_;
  bool use_center_pointing_shape_context_;
  bool start_loc_use_fixed_goal_;
  std::string current_file_id_;
  ProtoObject start_loc_obj_;
  double start_loc_arc_length_percent_;
  int start_loc_push_sample_count_;
  bool force_swap_;
  int mask_dilate_size_;
  double point_cloud_hist_res_;
};

int main(int argc, char ** argv)
{
  int seed = time(NULL);
  srand(seed);
  std::cout << "Rand seed is: " << seed << std::endl;
  ros::init(argc, argv, "tabletop_pushing_perception_node");
  ros::NodeHandle n;
  TabletopPushingPerceptionNode perception_node(n);
  perception_node.spin();
  return 0;
}
