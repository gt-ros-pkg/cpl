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
#include <ros/package.h>

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
#include <cpl_visual_features/comp_geometry.h>
#include <cpl_visual_features/features/shape_context.h>

// tabletop_pushing
#include <tabletop_pushing/LearnPush.h>
#include <tabletop_pushing/LocateTable.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <tabletop_pushing/shape_features.h>
#include <tabletop_pushing/object_tracker_25d.h>
#include <tabletop_pushing/push_primitives.h>
#include <tabletop_pushing/arm_obj_segmentation.h>
#include <tabletop_pushing/extern/Timer.hpp>
#include <tabletop_pushing/io_utils.h>

// libSVM
#include <libsvm/svm.h>

// STL
#include <vector>
#include <queue>
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
// #define DISPLAY_INPUT_COLOR 1
// #define DISPLAY_WAIT 1
// #define PROFILE_CB_TIME 1
// #define DEBUG_POSE_ESTIMATION 1
// #define VISUALIZE_CONTACT_PT 1
#define BUFFER_AND_WRITE 1

using boost::shared_ptr;

using namespace tabletop_pushing;

using geometry_msgs::PoseStamped;
using geometry_msgs::PointStamped;
using geometry_msgs::Pose2D;
using geometry_msgs::Twist;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image,
                                                        sensor_msgs::PointCloud2> MySyncPolicy;
// using cpl_visual_features::upSample;
// using cpl_visual_features::downSample;
using cpl_visual_features::subPIAngle;
using cpl_visual_features::lineSegmentIntersection2D;
using cpl_visual_features::ShapeDescriptors;
using cpl_visual_features::ShapeDescriptor;

typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;
typedef tabletop_pushing::VisFeedbackPushTrackingGoal PushTrackerGoal;
typedef tabletop_pushing::VisFeedbackPushTrackingResult PushTrackerResult;
typedef tabletop_pushing::VisFeedbackPushTrackingAction PushTrackerAction;

#define FOOTPRINT_XY_RES 0.001

inline int objLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/FOOTPRINT_XY_RES);
}

struct ScoredIdx
{
  double score;
  int idx;
};

class ScoredIdxComparison
{
 public:
  ScoredIdxComparison(const bool& descend=false) : descend_(descend) {}
  bool operator() (const ScoredIdx& lhs, const ScoredIdx&rhs) const
  {
    if (descend_)
    {
      return (lhs.score < rhs.score);
    }
    else
    {
      return (lhs.score > rhs.score);
    }
  }
 protected:
  bool descend_;
};

class TabletopPushingPerceptionNode
{
 public:
  TabletopPushingPerceptionNode(ros::NodeHandle &n) :
      n_(n), n_private_("~"),
      image_sub_(n, "color_image_topic", 1),
      mask_sub_(n, "mask_image_topic", 1),
      cloud_sub_(n, "point_cloud_topic", 1),
      sync_(MySyncPolicy(15), image_sub_, mask_sub_, cloud_sub_),
      as_(n, "push_tracker", false),
      have_sensor_data_(false),
      camera_initialized_(false), recording_input_(false), record_count_(0),
      learn_callback_count_(0), goal_out_count_(0), goal_heading_count_(0),
      frame_callback_count_(0), frame_set_count_(0),
      just_spun_(false), major_axis_spin_pos_scale_(0.75), object_not_moving_thresh_(0),
      object_not_moving_count_(0), object_not_moving_count_limit_(10),
      gripper_not_moving_thresh_(0), gripper_not_moving_count_(0),
      gripper_not_moving_count_limit_(10), current_file_id_(""), force_swap_(false),
      num_position_failures_(0), footprint_count_(0),
      feedback_control_count_(0), feedback_control_instance_count_(0)
  {
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());
    pcl_segmenter_ = shared_ptr<PointCloudSegmentation>(
        new PointCloudSegmentation(tf_));
    // Get parameters from the server
    n_private_.param("display_wait_ms", display_wait_ms_, 3);
    n_private_.param("use_displays", use_displays_, false);
    n_private_.param("write_input_to_disk", write_input_to_disk_, false);
    n_private_.param("write_to_disk", write_to_disk_, false);
    n_private_.param("write_dyn_learning_to_disk", write_dyn_to_disk_, false);

    n_private_.param("min_workspace_x", pcl_segmenter_->min_workspace_x_, 0.0);
    n_private_.param("min_workspace_z", pcl_segmenter_->min_workspace_z_, 0.0);
    n_private_.param("max_workspace_x", pcl_segmenter_->max_workspace_x_, 0.0);
    n_private_.param("max_workspace_z", pcl_segmenter_->max_workspace_z_, 0.0);
    n_private_.param("min_table_z", pcl_segmenter_->min_table_z_, -0.5);
    n_private_.param("max_table_z", pcl_segmenter_->max_table_z_, 1.5);

    // TODO: Tie these parameters with those in the tabletop_executive... (need to align goal and workspace names)
    n_private_.param("min_goal_x", min_goal_x_, 0.425);
    n_private_.param("max_goal_x", max_goal_x_, 0.8);
    n_private_.param("min_goal_y", min_goal_y_, -0.3);
    n_private_.param("max_goal_y", max_goal_y_, 0.3);

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
    n_private_.param("pcl_table_hull_alpha", pcl_segmenter_->hull_alpha_, 0.1);
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
    n_private_.param("gripper_not_moving_thresh", gripper_not_moving_thresh_, 0.005);
    n_private_.param("object_not_moving_thresh", object_not_moving_thresh_, 0.005);
    n_private_.param("gripper_not_moving_count_limit", gripper_not_moving_count_limit_, 100);
    n_private_.param("object_not_moving_count_limit", object_not_moving_count_limit_, 100);
    n_private_.param("object_not_detected_count_limit", object_not_detected_count_limit_, 5);
    n_private_.param("object_too_far_count_limit", object_too_far_count_limit_, 5);
    n_private_.param("object_not_between_count_limit", object_not_between_count_limit_, 5);
    n_private_.param("object_not_between_epsilon", object_not_between_epsilon_, 0.01);
    n_private_.param("start_loc_push_time_limit", start_loc_push_time_, 5.0);
    n_private_.param("start_loc_push_dist", start_loc_push_dist_, 0.30);
    n_private_.param("use_center_pointing_shape_context", use_center_pointing_shape_context_, true);
    n_private_.param("self_mask_dilate_size", mask_dilate_size_, 5);

    // Setup morphological element for arm segmentation mask
    cv::Mat tmp_morph(mask_dilate_size_, mask_dilate_size_, CV_8UC1, cv::Scalar(255));
    tmp_morph.copyTo(morph_element_);

    // Setup nan point for self filtering of point cloud
    nan_point_.x = numeric_limits<float>::quiet_NaN();
    nan_point_.y = numeric_limits<float>::quiet_NaN();
    nan_point_.z = numeric_limits<float>::quiet_NaN();

    n_private_.param("point_cloud_hist_res", point_cloud_hist_res_, 0.005);
    n_private_.param("boundary_hull_alpha", hull_alpha_, 0.01);
    n_private_.param("hull_gripper_spread", gripper_spread_, 0.05);

    n_.param("start_loc_use_fixed_goal", start_loc_use_fixed_goal_, false);
    n_.param("use_graphcut_arm_seg", use_graphcut_arm_seg_, false);


    std::string arm_color_model_name;
    n_private_.param("arm_color_model_name", arm_color_model_name, std::string(""));

#ifdef DEBUG_POSE_ESTIMATION
    pose_est_stream_.open("/u/thermans/data/new/pose_ests.txt");
#endif // DEBUG_POSE_ESTIMATION

    // Initialize classes requiring parameters
    arm_obj_segmenter_ = shared_ptr<ArmObjSegmentation>(new ArmObjSegmentation());
    if (arm_color_model_name.length() > 0)
    {
      std::stringstream arm_color_model_path;
      arm_color_model_path << ros::package::getPath("tabletop_pushing") << "/cfg/" << arm_color_model_name;
      arm_obj_segmenter_->loadArmColorModel(arm_color_model_path.str());
    }
    obj_tracker_ = shared_ptr<ObjectTracker25D>(
        new ObjectTracker25D(pcl_segmenter_, arm_obj_segmenter_, num_downsamples_, use_displays_,
                             write_to_disk_, base_output_path_, camera_frame_, use_cv_ellipse,
                             use_mps_segmentation_, use_graphcut_arm_seg_, hull_alpha_));

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
                      const sensor_msgs::ImageConstPtr& mask_msg,
                      const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
#ifdef PROFILE_CB_TIME
    long long cb_start_time = Timer::nanoTime();
#endif

    if (!camera_initialized_)
    {
      cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(
          cam_info_topic_, n_, ros::Duration(5.0));
      camera_initialized_ = true;
      have_sensor_data_ = true;
      pcl_segmenter_->cam_info_ = cam_info_;
      pcl_segmenter_->cur_camera_header_ = img_msg->header;
      ROS_DEBUG_STREAM("Cam info: " << cam_info_);
    }
    // Convert images to OpenCV format
    cv::Mat color_frame;
    cv::Mat self_mask;
    cv_bridge::CvImagePtr color_cv_ptr = cv_bridge::toCvCopy(img_msg);
    cv_bridge::CvImagePtr mask_cv_ptr = cv_bridge::toCvCopy(mask_msg);
    color_frame = color_cv_ptr->image;
    self_mask = mask_cv_ptr->image;

#ifdef PROFILE_CB_TIME
    long long grow_mask_start_time = Timer::nanoTime();
#endif
    // Grow arm mask if requested
    if (mask_dilate_size_ > 0)
    {
      cv::erode(self_mask, self_mask, morph_element_);
    }
#ifdef PROFILE_CB_TIME
    long long transform_start_time = Timer::nanoTime();
    double grow_mask_elapsed_time = (((double)(transform_start_time - grow_mask_start_time)) /
                                     Timer::NANOSECONDS_PER_SECOND);
#endif

    // Transform point cloud into the correct frame and convert to PCL struct
    pcl16::fromROSMsg(*cloud_msg, cur_point_cloud_);
    // TODO: Speed this up by not waiting...
    // (i.e. get new transform, use for whole time, assuming head is not moving)
    tf_->waitForTransform(workspace_frame_, cloud_msg->header.frame_id, cloud_msg->header.stamp,
                          ros::Duration(0.5));
    pcl16_ros::transformPointCloud(workspace_frame_, cur_point_cloud_, cur_point_cloud_, *tf_);

#ifdef PROFILE_CB_TIME
    long long filter_start_time = Timer::nanoTime();
    double transform_elapsed_time = (((double)(filter_start_time - transform_start_time)) /
                                     Timer::NANOSECONDS_PER_SECOND);
#endif

    pcl16::copyPointCloud(cur_point_cloud_, cur_self_filtered_cloud_);
    for (unsigned int r = 0; r < self_mask.rows; ++r)
    {
      for (unsigned int c = 0; c < self_mask.cols; ++c)
      {
        if (self_mask.at<uchar>(r, c) == 0)
        {
          cur_self_filtered_cloud_.at(c, r) = nan_point_;
        }
      }
    }
#ifdef PROFILE_CB_TIME
    long long downsample_start_time = Timer::nanoTime();
    double filter_elapsed_time = (((double)(downsample_start_time - filter_start_time)) /
                                   Timer::NANOSECONDS_PER_SECOND);
#endif

    // Downsample everything first
    // HACK: We have this fixed to 1, so let's not do extra memcopys
    // cv::Mat color_frame_down = downSample(color_frame, num_downsamples_);
    // cv::Mat self_mask_down = downSample(self_mask, num_downsamples_);
    cv::pyrDown(color_frame, cur_color_frame_);
    cv::pyrDown(self_mask, cur_self_mask_);

#ifdef PROFILE_CB_TIME
    long long tracker_start_time = Timer::nanoTime();
    double downsample_elapsed_time = (((double)(tracker_start_time - downsample_start_time)) /
                                      Timer::NANOSECONDS_PER_SECOND);
    double update_tracks_elapsed_time = 0.0;
    double copy_tracks_elapsed_time = 0.0;
    double display_tracks_elapsed_time = 0.0;
    double write_dyn_elapsed_time = 0.0;
    double publish_feedback_elapsed_time = 0.0;
    double evaluate_goal_elapsed_time = 0.0;
#endif

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
#ifdef PROFILE_CB_TIME
      long long update_tracks_start_time = Timer::nanoTime();
#endif

      PushTrackerState tracker_state;
      obj_tracker_->updateTracks(cur_color_frame_, cur_self_mask_, cur_self_filtered_cloud_, proxy_name_,
                                 tracker_state);

#ifdef DEBUG_POSE_ESTIMATION
      pose_est_stream_ << tracker_state.x.x << " " << tracker_state.x.y << " " << tracker_state.z << " "
                       << tracker_state.x.theta << "\n";
#endif // DEBUG_POSE_ESTIMATION

#ifdef PROFILE_CB_TIME
      long long copy_tracks_start_time = Timer::nanoTime();
      update_tracks_elapsed_time = (((double)(copy_tracks_start_time - update_tracks_start_time)) /
                                    Timer::NANOSECONDS_PER_SECOND);
#endif
#ifdef VISUALIZE_CONTACT_PT
      ProtoObject cur_contact_obj = obj_tracker_->getMostRecentObject();
      // TODO: Move this into the tracker or somewhere better
      XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_contact_obj, hull_alpha_);

      // Visualize hull_cloud;
      // NOTE: Get this point with tf for offline use
      geometry_msgs::PointStamped hand_pt_ros;
      geometry_msgs::PointStamped base_point;
      base_point.point.x = 0.0;
      base_point.point.y = 0.0;
      base_point.point.z = 0.0;
      base_point.header.frame_id = "l_gripper_tool_frame";
      tf_->transformPoint(workspace_frame_, base_point, hand_pt_ros);
      pcl16::PointXYZ hand_pt;
      hand_pt.x = hand_pt_ros.point.x;
      hand_pt.y = hand_pt_ros.point.y;
      hand_pt.z = hand_pt_ros.point.z;
      geometry_msgs::PointStamped forward_pt_ros;
      geometry_msgs::PointStamped base_point_forward;
      base_point_forward.point.x = 0.01;
      base_point_forward.point.y = 0.0;
      base_point_forward.point.z = 0.0;
      base_point_forward.header.frame_id = "l_gripper_tool_frame";
      tf_->transformPoint(workspace_frame_, base_point_forward, forward_pt_ros);
      pcl16::PointXYZ forward_pt;
      forward_pt.x = forward_pt_ros.point.x;
      forward_pt.y = forward_pt_ros.point.y;
      forward_pt.z = forward_pt_ros.point.z;

      cv::Mat hull_cloud_viz = tabletop_pushing::visualizeObjectContactLocation(hull_cloud, tracker_state,
                                                                                hand_pt, forward_pt);
      cv::imshow("obj footprint", hull_cloud_viz);
      // std::stringstream input_out_name;
      // input_out_name << base_output_path_ << "input" << footprint_count_ << ".png";
      // cv::imwrite(input_out_name.str(), cur_color_frame_);
      // std::stringstream footprint_out_name;
      // footprint_out_name << base_output_path_ << "footprint" << footprint_count_++ << ".png";
      // cv::imwrite(footprint_out_name.str(), hull_cloud_viz);

#endif // VISUALIZE_CONTACT_PT

      tracker_state.proxy_name = proxy_name_;
      tracker_state.controller_name = controller_name_;
      tracker_state.behavior_primitive = behavior_primitive_;

#ifdef PROFILE_CB_TIME
      copy_tracks_elapsed_time = (((double)(Timer::nanoTime() - copy_tracks_start_time)) /
                                        Timer::NANOSECONDS_PER_SECOND);
#endif // PROFILE_CB_TIME
#ifdef DISPLAY_WAIT
#ifdef PROFILE_CB_TIME
      long long display_tracks_start_time = Timer::nanoTime();
#endif
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
      if (controller_name_ == "rotate_to_heading")
      {
        displayGoalHeading(cur_color_frame_, start_point, tracker_state.x.theta,
                           tracker_goal_pose_.theta);
      }
#ifdef PROFILE_CB_TIME
      display_tracks_elapsed_time = (((double)(Timer::nanoTime() - display_tracks_start_time)) /
                                  Timer::NANOSECONDS_PER_SECOND);
#endif // PROFILE_CB_TIME
#endif // DISPLAY_WAIT

      // make sure that the action hasn't been canceled
      if (as_.isActive())
      {
#ifdef PROFILE_CB_TIME
        long long write_dyn_start_time = Timer::nanoTime();
#endif

        if (write_dyn_to_disk_)
        {
          // Write image and cur obj cloud
          std::stringstream image_out_name, obj_cloud_out_name;
          image_out_name << base_output_path_ << "feedback_control_input_" << feedback_control_instance_count_
                         << "_" << feedback_control_count_ << ".png";
          obj_cloud_out_name << base_output_path_ << "feedback_control_obj_" << feedback_control_instance_count_
                             << "_" << feedback_control_count_ << ".pcd";
          ProtoObject cur_obj = obj_tracker_->getMostRecentObject();

#ifdef BUFFER_AND_WRITE
          color_img_buffer_.push_back(cur_color_frame_);
          color_img_name_buffer_.push_back(image_out_name.str());
          obj_cloud_buffer_.push_back(cur_obj.cloud);
          obj_cloud_name_buffer_.push_back(obj_cloud_out_name.str());
#else // BUFFER_AND_WRITE
          cv::imwrite(image_out_name.str(), cur_color_frame_);
          pcl16::io::savePCDFile(obj_cloud_out_name.str(), cur_obj.cloud);
#endif // BUFFER_AND_WRITE

          if (feedback_control_count_ == 0)
          {
            // TODO: Look this up once and only save once...
            tf::StampedTransform workspace_to_cam_t;
            tf_->lookupTransform(camera_frame_, workspace_frame_, ros::Time(0), workspace_to_cam_t);
            std::stringstream workspace_to_cam_name, cam_info_name;
            workspace_to_cam_name << base_output_path_ << "workspace_to_cam_"
                                  << feedback_control_instance_count_ << ".txt";
            cam_info_name << base_output_path_ << "cam_info_" << feedback_control_instance_count_ << ".txt";

#ifdef BUFFER_AND_WRITE
            workspace_transform_buffer_.push_back(workspace_to_cam_t);
            workspace_transform_name_buffer_.push_back(workspace_to_cam_name.str());
            cam_info_buffer_.push_back(cam_info_);
            cam_info_name_buffer_.push_back(cam_info_name.str());
#else // BUFFER_AND_WRITE
            writeTFTransform(workspace_to_cam_t, workspace_to_cam_name.str());
            writeCameraInfo(cam_info_, cam_info_name.str());
#endif // BUFFER_AND_WRITE
          }
        }
#ifdef PROFILE_CB_TIME
        long long publish_feedback_start_time = Timer::nanoTime();
        write_dyn_elapsed_time = (((double)(publish_feedback_start_time - write_dyn_start_time)) /
                                  Timer::NANOSECONDS_PER_SECOND);
#endif

        // Put sequence / stamp id for tracker_state as unique ID for writing state & control info to disk
        tracker_state.header.seq = feedback_control_count_++;
        as_.publishFeedback(tracker_state);

#ifdef PROFILE_CB_TIME
        long long evaluate_goal_start_time = Timer::nanoTime();
        publish_feedback_elapsed_time = (((double)(evaluate_goal_start_time - publish_feedback_start_time)) /
                                      Timer::NANOSECONDS_PER_SECOND);
#endif

        evaluateGoalAndAbortConditions(tracker_state);

#ifdef PROFILE_CB_TIME
        evaluate_goal_elapsed_time = (((double)(Timer::nanoTime() - evaluate_goal_start_time)) /
                                      Timer::NANOSECONDS_PER_SECOND);
#endif
      }
    }
    else if (obj_tracker_->isInitialized() && obj_tracker_->isPaused())
    {
      obj_tracker_->pausedUpdate(cur_color_frame_);

#ifdef DISPLAY_WAIT
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
      if (controller_name_ == "rotate_to_heading")
      {
        displayGoalHeading(cur_color_frame_, start_point, tracker_state.x.theta,
                           tracker_goal_pose_.theta, true);
      }
#endif // DISPLAY_WAIT
    }

#ifdef PROFILE_CB_TIME
    double tracker_elapsed_time = (((double)(Timer::nanoTime() - tracker_start_time)) /
                                 Timer::NANOSECONDS_PER_SECOND);
#endif

    // Display junk
#ifdef DISPLAY_WAIT
#ifdef DISPLAY_INPUT_COLOR
    if (use_displays_)
    {
      cv::imshow("color", cur_color_frame_);
      // cv::imshow("self_mask", cur_self_mask_);
    }
    // Way too much disk writing!
    if (write_input_to_disk_ && recording_input_)
    {
      std::stringstream out_name;
      if (current_file_id_.size() > 0)
      {
        std::stringstream cloud_out_name;
        out_name << base_output_path_ << current_file_id_ << "_input_" << record_count_ << ".png";
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
    if (use_displays_)
    {
      cv::waitKey(display_wait_ms_);
    }
#endif // DISPLAY_WAIT

    ++frame_callback_count_;

#ifdef PROFILE_CB_TIME
    double cb_elapsed_time = (((double)(Timer::nanoTime() - cb_start_time)) /
                            Timer::NANOSECONDS_PER_SECOND);
    if (obj_tracker_->isInitialized() && !obj_tracker_->isPaused())
    {
      ROS_INFO_STREAM("cb_elapsed_time " << cb_elapsed_time);
      ROS_INFO_STREAM("\t grow_mask_elapsed_time " << grow_mask_elapsed_time);
      ROS_INFO_STREAM("\t transform_elapsed_time " << transform_elapsed_time);
      ROS_INFO_STREAM("\t filter_elapsed_time " << filter_elapsed_time);
      ROS_INFO_STREAM("\t downsample_elapsed_time " << downsample_elapsed_time);
      ROS_INFO_STREAM("\t tracker_elapsed_time " << tracker_elapsed_time);
      ROS_INFO_STREAM("\t\t update_tracks_elapsed_time " << update_tracks_elapsed_time);
      ROS_INFO_STREAM("\t\t copy_tracks_elapsed_time " << copy_tracks_elapsed_time);
      ROS_INFO_STREAM("\t\t display_tracks_elapsed_time " << display_tracks_elapsed_time);
      ROS_INFO_STREAM("\t\t write_dyn_elapsed_time " << publish_feedback_elapsed_time);
      ROS_INFO_STREAM("\t\t publish_feedback_elapsed_time " << publish_feedback_elapsed_time);
      ROS_INFO_STREAM("\t\t evaluate_goal_elapsed_time " << evaluate_goal_elapsed_time);
    }
#endif
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

    if (controller_name_ == "rotate_to_heading")
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
#ifdef BUFFER_AND_WRITE
        writeBuffersToDisk();
#endif // BUFFER_AND_WRITE
      }
      return;
    }
    // TODO: Switch to radius?
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
#ifdef BUFFER_AND_WRITE
      writeBuffersToDisk();
#endif // BUFFER_AND_WRITE
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
    else
    {
      if (objectTooFarFromGripper(tracker_state.x))
      {
        abortPushingGoal("Object is too far from gripper.");
      }
      else if (behavior_primitive_ != "gripper_pull" && objectNotBetweenGoalAndGripper(tracker_state.x))
      {
        abortPushingGoal("Object is not between gripper and goal.");
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
#ifdef BUFFER_AND_WRITE
    writeBuffersToDisk();
#endif // BUFFER_AND_WRITE
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
    if ( have_sensor_data_ )
    {
      if (!req.analyze_previous)
      {
        controller_name_ = req.controller_name;
        proxy_name_ = req.proxy_name;
        behavior_primitive_ = req.behavior_primitive;
      }

      if (req.initialize)
      {
        num_position_failures_ = 0;
        ROS_INFO_STREAM("Initializing");
        record_count_ = 0;
        learn_callback_count_ = 0;
        res.no_push = true;
        ROS_DEBUG_STREAM("Stopping input recording");
        recording_input_ = false;
        obj_tracker_->stopTracking();
      }
      else if (req.analyze_previous || req.get_pose_only)
      {
        ROS_INFO_STREAM("Getting current object pose");
        getObjectPose(res);
        res.no_push = true;
        ROS_DEBUG_STREAM("Stopping input recording");
        recording_input_ = false;
        ProtoObject cur_obj = obj_tracker_->getMostRecentObject();
        if (cur_obj.cloud.header.frame_id.size() == 0)
        {
          ROS_INFO_STREAM("cur_obj.cloud.header.frame_id is blank, setting to workspace_frame_");
          cur_obj.cloud.header.frame_id = workspace_frame_;
        }
        PushTrackerState cur_state;
        cur_state.x.x = res.centroid.x;
        cur_state.x.y = res.centroid.y;
        cur_state.x.theta = res.theta;
        cur_state.z = res.centroid.z;
        if (req.analyze_previous)
        {
          // Display different color to signal swap available
          obj_tracker_->trackerDisplay(cur_color_frame_, cur_state, cur_obj, true);
          ROS_INFO_STREAM("Current theta: " << cur_state.x.theta);
          if (use_displays_)
          {
            ROS_INFO_STREAM("Presss 's' to swap orientation: ");
            char key_press = cv::waitKey(2000);
            if (key_press == 's')
            {
              // TODO: Is this correct?
              force_swap_ = !obj_tracker_->getSwapState();
              startTracking(cur_state, force_swap_);
              ROS_INFO_STREAM("Swapped theta: " << cur_state.x.theta);
              res.theta = cur_state.x.theta;
              obj_tracker_->stopTracking();
              obj_tracker_->trackerDisplay(cur_color_frame_, cur_state, cur_obj);
              // NOTE: Try and force redraw
              cv::waitKey(3);
            }
          }
        }
        else
        {
          ROS_INFO_STREAM("Calling tracker display");
          obj_tracker_->trackerDisplay(cur_color_frame_, cur_state, cur_obj);
          obj_tracker_->stopTracking();
        }
        ROS_INFO_STREAM("Done getting current pose\n");
      }
      else // NOTE: Assume pushing as default
      {
        ROS_INFO_STREAM("Determining push start pose");
        res = getPushStartPose(req);
        recording_input_ = !res.no_objects;
        if (recording_input_)
        {
          ROS_DEBUG_STREAM("Starting input recording");
          ROS_DEBUG_STREAM("current_file_id: " << current_file_id_);
        }
        else
        {
          ROS_DEBUG_STREAM("Stopping input recording");
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
    startTracking(cur_state, force_swap_);
    ProtoObject cur_obj = obj_tracker_->getMostRecentObject();
    if (req.learn_start_loc)
    {
      obj_tracker_->trackerDisplay(cur_color_frame_, cur_state, cur_obj, true);
      if (use_displays_)
      {
        ROS_INFO_STREAM("Current theta: " << cur_state.x.theta);
        ROS_INFO_STREAM("Presss 's' to swap orientation: ");
        char key_press = cv::waitKey(2000);
        if (key_press == 's')
        {
          force_swap_ = !force_swap_;
          obj_tracker_->toggleSwap();
          startTracking(cur_state, force_swap_);
          obj_tracker_->trackerDisplay(cur_color_frame_, cur_state, cur_obj);
          // NOTE: Try and force redraw
          cv::waitKey(3);
          ROS_INFO_STREAM("Swapped theta: " << cur_state.x.theta);
        }
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

    // Set basic push information
    PushVector p;
    p.header.frame_id = workspace_frame_;
    bool pull_start = (req.behavior_primitive == "gripper_pull");
    bool rotate_push = (req.controller_name == "rotate_to_heading");

    // Choose a pushing location to test if we are learning good pushing locations
    if (req.learn_start_loc)
    {
      // Get the pushing location
      ShapeLocation chosen_loc;
      float predicted_score = -1;
      if (req.start_loc_param_path.length() > 0) // Choose start location using the learned classifier
      {
        ROS_INFO_STREAM("Finding learned push start loc");
        ROS_INFO_STREAM("Using param path "<< req.start_loc_param_path);

        // HACK: We set the name to "rand" if we are testing with rand
        if (req.start_loc_param_path.compare("rand") == 0)
        {
          chosen_loc = chooseRandomPushStartLoc(cur_obj, cur_state, rotate_push);
        }
        else
        {
          chosen_loc = chooseLearnedPushStartLoc(cur_obj, cur_state, req.start_loc_param_path, predicted_score,
                                                 req.previous_position_worked, rotate_push);
        }
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
      res.predicted_score = predicted_score;
      float new_push_angle;
      if (rotate_push)
      {
        // Set goal for rotate pushing angle and goal state; then get start location as usual below
        new_push_angle = getRotatePushHeading(cur_state, chosen_loc, cur_obj);
        res.goal_pose.x = cur_state.x.x;
        res.goal_pose.y = cur_state.x.y;
        // NOTE: Uncomment for visualization purposes
        // res.goal_pose.x = res.centroid.x+cos(new_push_angle)*start_loc_push_dist_;
        // res.goal_pose.y = res.centroid.y+sin(new_push_angle)*start_loc_push_dist_;
        res.goal_pose.theta = subPIAngle(cur_state.x.theta+M_PI);
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
      else if (rotate_push)
      {
        // TODO: Figure out something here
      }
      else
      {
        p.push_angle = atan2(res.goal_pose.y - res.centroid.y, res.goal_pose.x - res.centroid.x);
      }
      // Get vector through centroid and determine start point and distance
      Eigen::Vector3f push_unit_vec(std::cos(p.push_angle), std::sin(p.push_angle), 0.0f);
      std::vector<pcl16::PointXYZ> end_points;
      pcl_segmenter_->lineCloudIntersectionEndPoints(cur_obj.cloud, push_unit_vec, cur_obj.centroid, end_points);
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
    if (rotate_push)
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
      if (use_displays_)
      {
        displayInitialPushVector(cur_color_frame_, start_point, end_point, obj_centroid);
      }
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
    XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha_);

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
    pcl16::PointXYZ boundary_loc = hull_cloud[boundary_loc_idx];
    ShapeDescriptor sd = tabletop_pushing::extractLocalAndGlobalShapeFeatures(hull_cloud, cur_obj, boundary_loc,
                                                                              boundary_loc_idx, gripper_spread_,
                                                                              hull_alpha_, point_cloud_hist_res_);
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
                                          float& chosen_score, bool previous_position_worked,
                                          bool rotate_push)
  {
    // Get features for all of the boundary locations
    // TODO: Set these values somewhere else
    int local_length = 36;
    int global_length = 60;
    XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha_);
    ShapeDescriptors sds = tabletop_pushing::extractLocalAndGlobalShapeFeatures(hull_cloud, cur_obj,
                                                                                gripper_spread_, hull_alpha_,
                                                                                point_cloud_hist_res_);
    // Read in model SVs and coefficients
    svm_model* push_model;
    push_model = svm_load_model(param_path.c_str());
    // Remove trailing .model
    param_path.erase(param_path.size()-6, 6);
    std::stringstream train_feat_path;
    train_feat_path << param_path << "-feats.txt";

    // TODO: Get these parameters from disk
    double gamma_local = 2.5;
    double gamma_global = 2.0;
    double mixture_weight = 0.7;
    if (rotate_push)
    {
      gamma_local = 0.05;
      gamma_global = 2.5;
      mixture_weight = 0.7;
    }
    cv::Mat K = tabletop_pushing::computeChi2Kernel(sds, train_feat_path.str(), local_length, global_length,
                                                    gamma_local, gamma_global, mixture_weight);

    std::vector<double> pred_push_scores;
    std::priority_queue<ScoredIdx, std::vector<ScoredIdx>, ScoredIdxComparison> pq(
        (ScoredIdxComparison(rotate_push)) );
    XYZPointCloud hull_cloud_obj;
    hull_cloud_obj.width = hull_cloud.size();
    hull_cloud_obj.height = 1;
    hull_cloud_obj.is_dense = false;
    hull_cloud_obj.resize(hull_cloud_obj.width*hull_cloud_obj.height);

    // Perform prediction at all sample locations
    for (int i = 0; i < K.cols; ++i)
    {
      svm_node* x = new svm_node[K.rows+1];
      x[0].value = 0;
      x[0].index = 0;
      for (int j = 0; j < K.rows; ++j)
      {
        x[j+1].value = K.at<double>(j, i);
        x[j+1].index = 0; // unused
      }
      // Perform prediction and convert into appropriate space
      double raw_pred_score = svm_predict(push_model, x);
      double pred_score;
      if (rotate_push)
      {
        pred_score = exp(raw_pred_score);
      }
      else
      {
        pred_score = exp(raw_pred_score);
      }
      if (isnan(pred_score) || isinf(pred_score))
      {
        ROS_WARN_STREAM("Sample " << i <<  " has pred score: " << pred_score << "\traw pred score: " << raw_pred_score);
      }
      if (isinf(pred_score))
      {
        pred_score = raw_pred_score;
      }
      ScoredIdx scored_idx;
      scored_idx.score = pred_score;
      scored_idx.idx = i;
      pq.push(scored_idx);
      pred_push_scores.push_back(pred_score);
      // Center cloud at (0,0) but leave the orientation
      hull_cloud_obj[i].x = hull_cloud[i].x - cur_state.x.x;
      hull_cloud_obj[i].y = hull_cloud[i].y - cur_state.x.y;
      delete x;
    }

    // Free SVM struct
    // svm_free_and_destroy_model(&push_model);

    // Write SVM scores & descriptors to disk?
    writePredictedScoreToDisk(hull_cloud, sds, pred_push_scores);

    ScoredIdx best_scored = pq.top();
    // if (!previous_position_worked)
    // {
    //   // TODO: Replace this with location history not simple count
    //   num_position_failures_++;
    //   ROS_INFO_STREAM("Ignoring top: " << num_position_failures_ << " positions");
    //   for (int p = 0; p < num_position_failures_; ++p)
    //   {
    //     pq.pop();
    //   }
    // }
    // else
    // {
    //   num_position_failures_ = 0;
    // }

    // Ensure goal pose is on the table
    while (pq.size() > 0)
    {
      ScoredIdx chosen = pq.top();
      pq.pop();

      // Return the location of the best score
      ShapeLocation loc;
      loc.boundary_loc_ = hull_cloud[chosen.idx];
      loc.descriptor_ = sds[chosen.idx];
      float new_push_angle;
      Pose2D goal_pose =  generateStartLocLearningGoalPose(cur_state, cur_obj, loc, new_push_angle, rotate_push);
      if (rotate_push || goalPoseValid(goal_pose))
      {
        ROS_INFO_STREAM("Chose push location " << chosen.idx << " with score " << chosen.score);
        pcl16::PointXYZ selected(hull_cloud_obj[chosen.idx].x, hull_cloud_obj[chosen.idx].y, 0.0);
        displayLearnedPushLocScores(pred_push_scores, hull_cloud_obj, selected, rotate_push);
        chosen_score = chosen.score;
        return loc;
      }
    }
    // No points wokred
    ShapeLocation best_loc;
    best_loc.boundary_loc_ = hull_cloud[best_scored.idx];
    best_loc.descriptor_ = sds[best_scored.idx];
    ROS_INFO_STREAM("Chose default push location " << best_scored.idx << " with score " << best_scored.score);
    chosen_score = best_scored.score;
    return best_loc;
  }

  ShapeLocation chooseRandomPushStartLoc(ProtoObject& cur_obj, PushTrackerState& cur_state, bool rotate_push)
  {
    // Get features for all of the boundary locations
    // TODO: Set these values somewhere else
    XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha_);
    ShapeDescriptors sds = tabletop_pushing::extractLocalAndGlobalShapeFeatures(hull_cloud, cur_obj,
                                                                                gripper_spread_, hull_alpha_,
                                                                                point_cloud_hist_res_);
    std::vector<int> available_indices;
    for (int i = 0; i < hull_cloud.size(); ++i)
    {
      available_indices.push_back(i);
    }
    // Ensure goal pose is on the table
    while (available_indices.size() > 0)
    {
      int rand_idx = rand()%available_indices.size();
      int chosen_idx = available_indices[rand_idx];

      // Return the location of the best score
      ShapeLocation loc;
      loc.boundary_loc_ = hull_cloud[chosen_idx];
      loc.descriptor_ = sds[chosen_idx];
      float new_push_angle;
      Pose2D goal_pose =  generateStartLocLearningGoalPose(cur_state, cur_obj, loc, new_push_angle, rotate_push);
      if (rotate_push || goalPoseValid(goal_pose))
      {
        // TODO: Display boundary with 0 scores
        ROS_INFO_STREAM("Choosing random idx: " << chosen_idx);
        return loc;
      }
      available_indices.erase(available_indices.begin()+rand_idx);
    }
    // No points wokred
    int chosen_idx = rand()%hull_cloud.size();
    ShapeLocation chosen_loc;
    chosen_loc.boundary_loc_ = hull_cloud[chosen_idx];
    chosen_loc.descriptor_ = sds[chosen_idx];
    return chosen_loc;
  }

  std::vector<pcl16::PointXYZ> findAxisAlignedBoundingBox(PushTrackerState& cur_state, ProtoObject& cur_obj)
  {
    // Get cloud in object frame
    XYZPointCloud object_frame_cloud;
    worldPointsInObjectFrame(cur_obj.cloud, cur_state, object_frame_cloud);
    // Find min max x and y
    double min_x = FLT_MAX;
    double max_x = -FLT_MAX;
    double min_y = FLT_MAX;
    double max_y = -FLT_MAX;
    for (int i = 0; i < object_frame_cloud.size(); ++i)
    {
      if (object_frame_cloud[i].x < min_x)
      {
        min_x = object_frame_cloud[i].x;
      }
      if (object_frame_cloud[i].x > max_x)
      {
        max_x = object_frame_cloud[i].x;
      }
      if (object_frame_cloud[i].y < min_y)
      {
        min_y = object_frame_cloud[i].y;
      }
      if (object_frame_cloud[i].y > max_y)
      {
        max_y = object_frame_cloud[i].y;
      }
    }
    std::vector<pcl16::PointXYZ> vertices;
    vertices.push_back(objectPointInWorldFrame(pcl16::PointXYZ(max_x, max_y, 0.0), cur_state));
    vertices.push_back(objectPointInWorldFrame(pcl16::PointXYZ(max_x, min_y, 0.0), cur_state));
    vertices.push_back(objectPointInWorldFrame(pcl16::PointXYZ(min_x, min_y, 0.0), cur_state));
    vertices.push_back(objectPointInWorldFrame(pcl16::PointXYZ(min_x, max_y, 0.0), cur_state));
    return vertices;
  }

  float getRotatePushHeading(PushTrackerState& cur_state, ShapeLocation& chosen_loc, ProtoObject& cur_obj)
  {
    // cv::RotatedRect bounding_box = obj_tracker_->findFootprintBox(cur_obj);
    // cv::Point2f vertices[4];
    // bounding_box.points(vertices);

    std::vector<pcl16::PointXYZ> vertices = findAxisAlignedBoundingBox(cur_state, cur_obj);
    double min_dist = FLT_MAX;
    int chosen_idx = 0;
    float push_angle_world_frame = 0.0;

    cv::Point2f box_center;
    box_center.x = (vertices[0].x+vertices[2].x)*0.5;
    box_center.y = (vertices[0].y+vertices[2].y)*0.5;
    for (int i = 0; i < 4; ++i)
    {
      int j = (i+1)%4;
      double line_dist = cpl_visual_features::pointLineDistance2D(chosen_loc.boundary_loc_, vertices[i], vertices[j]);
      if (line_dist < min_dist)
      {
        min_dist = line_dist;
        chosen_idx = i;
        // NOTE: Push the direction from the chosen side's midpoint through the Box's center
        push_angle_world_frame = atan2(box_center.y-(vertices[i].y + vertices[j].y)*0.5,
                                       box_center.x-(vertices[i].x + vertices[j].x)*0.5);
      }
    }

    cv::Mat display_frame;
    cur_color_frame_.copyTo(display_frame);
    for (int i = 0; i < 4; ++i)
    {
      int j = (i+1)%4;

      PointStamped a_world;
      a_world.header.frame_id = cur_obj.cloud.header.frame_id;
      a_world.point.x = vertices[i].x;
      a_world.point.y = vertices[i].y;
      a_world.point.z = vertices[i].z;
      PointStamped b_world;
      b_world.header.frame_id = cur_obj.cloud.header.frame_id;
      b_world.point.x = vertices[j].x;
      b_world.point.y = vertices[j].y;
      b_world.point.z = vertices[j].z;
      cv::Point a_img = pcl_segmenter_->projectPointIntoImage(a_world);
      cv::Point b_img = pcl_segmenter_->projectPointIntoImage(b_world);
      cv::line(display_frame, a_img, b_img, cv::Scalar(0,0,0),3);
      if (i == chosen_idx)
      {
        // ROS_INFO_STREAM("chosen_idx is: " << chosen_idx);
        cv::line(display_frame, a_img, b_img, cv::Scalar(0,255,255),1);
      }
      else
      {
        cv::line(display_frame, a_img, b_img, cv::Scalar(0,255,0),1);
      }
    }
    cv::imshow("footprint_box", display_frame);
    // TODO: write to disk?
    return push_angle_world_frame;
  }

  Pose2D generateStartLocLearningGoalPose(PushTrackerState& cur_state, ProtoObject& cur_obj,
                                          ShapeLocation& chosen_loc, float& new_push_angle, bool rotate_push=false)
  {
    Pose2D goal_pose;
    if (rotate_push)
    {
      // Set goal for rotate pushing angle and goal state; then get start location as usual below
      new_push_angle = getRotatePushHeading(cur_state, chosen_loc, cur_obj);
      goal_pose.x = cur_state.x.x;
      goal_pose.y = cur_state.x.y;
      // NOTE: Uncomment for visualization purposes
      // res.goal_pose.x = cur_state.x.x+cos(new_push_angle)*start_loc_push_dist_;
      // res.goal_pose.y = cur_state.x.y+sin(new_push_angle)*start_loc_push_dist_;
      // TODO: Figure out +/- here
      goal_pose.theta = cur_state.x.theta+M_PI;
    }
    else
    {
      // Set goal for pushing and then get start location as usual below
      new_push_angle = atan2(cur_state.x.y - chosen_loc.boundary_loc_.y,
                             cur_state.x.x - chosen_loc.boundary_loc_.x);
      goal_pose.x = cur_state.x.x+cos(new_push_angle)*start_loc_push_dist_;
      goal_pose.y = cur_state.x.y+sin(new_push_angle)*start_loc_push_dist_;
    }
    return goal_pose;
  }

  bool goalPoseValid(Pose2D goal_pose) const
  {
    return (goal_pose.x < max_goal_x_ && goal_pose.x > min_goal_x_ &&
            goal_pose.y < max_goal_y_ && goal_pose.y > min_goal_y_);
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

  void worldPointsInObjectFrame(XYZPointCloud& world_cloud, PushTrackerState& cur_state, XYZPointCloud& object_cloud)
  {
    object_cloud.width = world_cloud.size();
    object_cloud.height = 1;
    object_cloud.is_dense = false;
    object_cloud.resize(object_cloud.width);
    for (int i = 0; i < world_cloud.size(); ++i)
    {
      object_cloud[i] = worldPointInObjectFrame(world_cloud[i], cur_state);
    }
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

  void getObjectPose(LearnPush::Response& res)
  {
    bool no_objects = false;
    ProtoObject cur_obj = obj_tracker_->findTargetObject(cur_color_frame_, cur_point_cloud_, no_objects, true);
    if (no_objects)
    {
      ROS_WARN_STREAM("No objects found on analysis");
      res.centroid.x = 0.0;
      res.centroid.y = 0.0;
      res.centroid.z = 0.0;
      res.theta = 0.0;
      res.no_objects = true;
      return;
    }
    cv::RotatedRect obj_ellipse;
    obj_tracker_->fitObjectEllipse(cur_obj, obj_ellipse);
    res.no_objects = false;
    res.centroid.x = cur_obj.centroid[0];
    res.centroid.y = cur_obj.centroid[1];
    res.centroid.z = cur_obj.centroid[2];
    res.theta = obj_tracker_->getThetaFromEllipse(obj_ellipse);
    if(obj_tracker_->getSwapState())
    {
      if(res.theta > 0.0)
        res.theta += - M_PI;
      else
        res.theta += M_PI;
    }
  }

  void startTracking(PushTrackerState& state, bool start_swap=false)
  {
    ROS_INFO_STREAM("Starting tracker");
    frame_set_count_++;
    goal_out_count_ = 0;
    goal_heading_count_ = 0;
    frame_callback_count_ = 0;
    obj_tracker_->initTracks(cur_color_frame_, cur_self_mask_, cur_self_filtered_cloud_, proxy_name_, state,
                             start_swap);
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
    shared_ptr<const PushTrackerGoal> tracker_goal = as_.acceptNewGoal();
    tracker_goal_pose_ = tracker_goal->desired_pose;
    pushing_arm_ = tracker_goal->which_arm;
    controller_name_ = tracker_goal->controller_name;
    proxy_name_ = tracker_goal->proxy_name;
    behavior_primitive_ = tracker_goal->behavior_primitive;
    ROS_INFO_STREAM("Accepted goal of " << tracker_goal_pose_);
    gripper_not_moving_count_ = 0;
    object_not_moving_count_ = 0;
    object_not_detected_count_ = 0;
    object_too_far_count_ = 0;
    object_not_between_count_ = 0;
    feedback_control_count_ = 0;
    feedback_control_instance_count_++;
    push_start_time_ = ros::Time::now().toSec();

    if (obj_tracker_->isInitialized())
    {
      obj_tracker_->unpause();
    }
    else
    {
      PushTrackerState state;
      startTracking(state);
    }
  }

  void pushTrackerPreemptCB()
  {
    obj_tracker_->pause();
    ROS_INFO_STREAM("Preempted push tracker");
    // set the action state to preempted
    as_.setPreempted();
#ifdef BUFFER_AND_WRITE
    writeBuffersToDisk();
#endif // BUFFER_AND_WRITE
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
    if ( have_sensor_data_ )
    {
      getTablePlane(cur_point_cloud_, res.table_centroid);
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
  void getTablePlane(XYZPointCloud& cloud, PoseStamped& p)
  {
    XYZPointCloud obj_cloud, table_cloud;
    // TODO: Comptue the hull on the first call
    Eigen::Vector4f table_centroid;
    pcl_segmenter_->getTablePlane(cloud, obj_cloud, table_cloud, table_centroid, false, true);
    p.pose.position.x = table_centroid[0];
    p.pose.position.y = table_centroid[1];
    p.pose.position.z = table_centroid[2];
    p.header = cloud.header;
    ROS_INFO_STREAM("Table centroid is: ("
                    << p.pose.position.x << ", "
                    << p.pose.position.y << ", "
                    << p.pose.position.z << ")");
    table_centroid_ = p;
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


  void displayGoalHeading(cv::Mat& img, PointStamped& centroid, double theta, double goal_theta,
                          bool force_no_write=false)
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
    if (write_to_disk_ && !force_no_write)
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

  void displayLearnedPushLocScores(std::vector<double>& push_scores, XYZPointCloud& locs, pcl16::PointXYZ selected,
                                   bool rotate_push)
  {
    double max_y = 0.2;
    double min_y = -0.2;
    double max_x = 0.2;
    double min_x = -0.2;
    int rows = ceil((max_y-min_y)/FOOTPRINT_XY_RES);
    int cols = ceil((max_x-min_x)/FOOTPRINT_XY_RES);
    cv::Mat footprint(rows, cols, CV_8UC3, cv::Scalar(255,255,255));

    for (int i = 0; i < push_scores.size(); ++i)
    {
      int img_y = rows-objLocToIdx(locs[i].x, min_x, max_x);
      int img_x = cols-objLocToIdx(locs[i].y, min_y, max_y);
      double score;
      if(rotate_push)
      {
        score = push_scores[i]/M_PI;
      }
      else
      {
        score = -log(push_scores[i])/10;
      }
      // ROS_INFO_STREAM("loc (" << x << ", " << y << ") : " << push_scores[i] << "\t" << score);
      cv::Scalar color(0, score*255, (1-score)*255);
      cv::circle(footprint, cv::Point(img_x, img_y), 1, color, 3);
      cv::circle(footprint, cv::Point(img_x, img_y), 2, color, 3);
      cv::circle(footprint, cv::Point(img_x, img_y), 3, color, 3);
    }

    // TOOD: Set out_file_path correctly
    std::stringstream score_file_name;
    score_file_name << base_output_path_ << "score_footprint" << "_" << frame_set_count_ << ".png";
    cv::imwrite(score_file_name.str(), footprint);

    // Highlight selected pushing locationx
    cv::Point selected_img(cols-objLocToIdx(selected.y, min_y, max_y), rows-objLocToIdx(selected.x, min_x, max_x));
    cv::circle(footprint, selected_img, 5, cv::Scalar(0,0,0), 2);
    cv::imshow("Push score", footprint);

    std::stringstream score_selected_file_name;
    score_selected_file_name << base_output_path_ << "score_footprint_selected" << "_" << frame_set_count_ << ".png";
    cv::imwrite(score_selected_file_name.str(), footprint);
  }

  void writePredictedScoreToDisk(XYZPointCloud& hull_cloud, ShapeDescriptors& sds, std::vector<double>& pred_scores)
  {
    // Write SVM scores & descriptors to disk?
    std::stringstream svm_file_name;
    svm_file_name << base_output_path_ << "predicted_svm_scores" << "_" << frame_set_count_ << ".txt";
    // TODO: write loc, score, descriptor
    std::ofstream svm_file_stream;
    svm_file_stream.open(svm_file_name.str().c_str());
    for (int i = 0; i < hull_cloud.size(); ++i)
    {
      svm_file_stream << hull_cloud[i].x << " " << hull_cloud[i].y << " " << pred_scores[i];
      for (int j = 0; j < sds[i].size(); ++j)
      {
        svm_file_stream << " " << sds[i][j];
      }
      svm_file_stream << "\n";
    }
    svm_file_stream.close();
  }

#ifdef BUFFER_AND_WRITE
  void writeBuffersToDisk()
  {
    for (int i = 0; i < color_img_buffer_.size(); ++i)
    {
      cv::imwrite(color_img_name_buffer_[i], color_img_buffer_[i]);
      pcl16::io::savePCDFileBinary<pcl16::PointXYZ>(obj_cloud_name_buffer_[i], obj_cloud_buffer_[i]);
    }
    for (int i = 0; i < workspace_transform_buffer_.size(); ++i)
    {
      writeTFTransform(workspace_transform_buffer_[i], workspace_transform_name_buffer_[i]);
      writeCameraInfo(cam_info_buffer_[i], cam_info_name_buffer_[i]);
    }
    // Clean up
    color_img_buffer_.clear();
    color_img_name_buffer_.clear();
    obj_cloud_buffer_.clear();
    obj_cloud_name_buffer_.clear();
    workspace_transform_buffer_.clear();
    workspace_transform_name_buffer_.clear();
    cam_info_buffer_.clear();
    cam_info_name_buffer_.clear();
  }
#endif // BUFFER_AND_WRITE

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
  cv::Mat cur_self_mask_;
  cv::Mat morph_element_;
  XYZPointCloud cur_point_cloud_;
  XYZPointCloud cur_self_filtered_cloud_;
  shared_ptr<PointCloudSegmentation> pcl_segmenter_;
  bool have_sensor_data_;
  int display_wait_ms_;
  bool use_displays_;
  bool write_input_to_disk_;
  bool write_to_disk_;
  bool write_dyn_to_disk_;
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
  int num_position_failures_;
  double min_goal_x_;
  double max_goal_x_;
  double min_goal_y_;
  double max_goal_y_;
  shared_ptr<ArmObjSegmentation> arm_obj_segmenter_;
  bool use_graphcut_arm_seg_;
  double hull_alpha_;
  double gripper_spread_;
  int footprint_count_;
  int feedback_control_count_;
  int feedback_control_instance_count_;
  pcl16::PointXYZ nan_point_;
#ifdef DEBUG_POSE_ESTIMATION
  std::ofstream pose_est_stream_;
#endif
#ifdef BUFFER_AND_WRITE
  std::vector<cv::Mat> color_img_buffer_;
  std::vector<XYZPointCloud> obj_cloud_buffer_;
  std::vector<tf::StampedTransform> workspace_transform_buffer_;
  std::vector<sensor_msgs::CameraInfo> cam_info_buffer_;
  std::vector<std::string> color_img_name_buffer_;
  std::vector<std::string> obj_cloud_name_buffer_;
  std::vector<std::string> workspace_transform_name_buffer_;
  std::vector<std::string> cam_info_name_buffer_;
#endif

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
