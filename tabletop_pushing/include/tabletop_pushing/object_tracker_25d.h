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
#ifndef object_tracker_25d_h_DEFINED
#define object_tracker_25d_h_DEFINED 1

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/PoseStamped.h>

// PCL
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <pcl16/registration/icp.h>
#include <pcl16/registration/transformation_estimation_svd.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

// Boost
#include <boost/shared_ptr.hpp>

// tabletop_pushing
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <tabletop_pushing/arm_obj_segmentation.h>
#include <tabletop_pushing/extern/gmm/gmm.h>

// STL
#include <string>

// Functional IFDEFS
#define USE_ORB 1

namespace tabletop_pushing
{
class ObjectFeaturePointModel
{
 public:
  cv::Mat descriptors;
  pcl16::PointCloud<pcl16::PointXYZ> locations;
  std::vector<cv::KeyPoint> keypoints;
  std::vector<int> bad_locs;
};

class ObjectTracker25D
{
 public:
  ObjectTracker25D(boost::shared_ptr<PointCloudSegmentation> segmenter,
                   boost::shared_ptr<ArmObjSegmentation> arm_segmenter,
                   int num_downsamples = 0,
                   bool use_displays=false, bool write_to_disk=false,
                   std::string base_output_path="", std::string camera_frame="",
                   bool use_cv_ellipse = false, bool use_mps_segmentation=false,
                   bool use_graphcut_arm_seg_=false,
                   double hull_alpha = 0.01,
                   int feature_close_size = 3,
                   float feature_point_ransac_inlier_thresh = 0.01,
                   int feature_point_max_ransac_iters = 100,
                   int brief_descriptor_byte_size = 16,
                   float feature_point_ratio_test_thresh=0.75,
                   double segment_search_radius = 0.3,
                   double feature_point_ransac_inlier_percent_thresh = 0.85);

  ProtoObject findTargetObject(cv::Mat& in_frame, pcl16::PointCloud<pcl16::PointXYZ>& cloud,
                               bool& no_objects, bool init=false);

  ProtoObject findTargetObjectGC(cv::Mat& in_frame, XYZPointCloud& cloud, cv::Mat& depth_frame,
                                 cv::Mat self_mask, bool& no_objects, bool init=false);

  void computeState(ProtoObject& cur_obj, pcl16::PointCloud<pcl16::PointXYZ>& cloud,
                    std::string proxy_name, cv::Mat& in_frame,
                    tabletop_pushing::VisFeedbackPushTrackingFeedback& state, bool init_state=false);

  void fitObjectEllipse(ProtoObject& obj, cv::RotatedRect& ellipse);

  void fitHullEllipse(XYZPointCloud& hull_cloud, cv::RotatedRect& obj_ellipse);

  void findFootprintEllipse(ProtoObject& obj, cv::RotatedRect& ellipse);

  void findFootprintBox(ProtoObject& obj, cv::RotatedRect& ellipse);

  void fit2DMassEllipse(ProtoObject& obj, cv::RotatedRect& ellipse);

  void updateStateEllipse(ProtoObject& obj, cv::RotatedRect& ellipse,
                          tabletop_pushing::VisFeedbackPushTrackingFeedback& state, bool init_state);

  void estimateTransformFromStateChange(tabletop_pushing::VisFeedbackPushTrackingFeedback& state,
                                        tabletop_pushing::VisFeedbackPushTrackingFeedback& previous_state,
                                        Eigen::Matrix4f& transform);

  void extractFeaturePointModel(cv::Mat& frame, pcl16::PointCloud<pcl16::PointXYZ>& cloud, ProtoObject& obj,
                                ObjectFeaturePointModel& model);

  bool estimateFeaturePointTransform(ObjectFeaturePointModel& source_model, ObjectFeaturePointModel& target_model,
                                     pcl16::Correspondences correspondences, Eigen::Matrix4f& transform);

  double getThetaFromEllipse(cv::RotatedRect& obj_ellipse);

  void initTracks(cv::Mat& in_frame, cv::Mat& self_mask, pcl16::PointCloud<pcl16::PointXYZ>& cloud,
                  std::string proxy_name, tabletop_pushing::VisFeedbackPushTrackingFeedback& state,
                  bool start_swap=false);

  void updateTracks(cv::Mat& in_frame, cv::Mat& self_mask, pcl16::PointCloud<pcl16::PointXYZ>& cloud,
                    std::string proxy_name, tabletop_pushing::VisFeedbackPushTrackingFeedback& state);

  void pausedUpdate(cv::Mat in_frame);

  //
  // Getters & Setters
  //
  bool isInitialized() const
  {
    return initialized_;
  }

  void stopTracking()
  {
    initialized_ = false;
    paused_ = false;
  }

  void setNumDownsamples(int num_downsamples)
  {
    num_downsamples_ = num_downsamples;
    upscale_ = std::pow(2,num_downsamples_);
  }

  tabletop_pushing::VisFeedbackPushTrackingFeedback getMostRecentState() const
  {
    return previous_state_;
  }

  ProtoObject getMostRecentObject() const
  {
    return previous_obj_;
  }

  void pause()
  {
    paused_ = true;
  }

  void unpause()
  {
    paused_ = false;
  }

  bool isPaused() const
  {
    return paused_;
  }

  bool getSwapState() const
  {
    return swap_orientation_;
  }

  void toggleSwap()
  {
    swap_orientation_ = !swap_orientation_;
  }

  //
  // I/O Functions
  //

  void trackerDisplay(cv::Mat& in_frame, ProtoObject& cur_obj, cv::RotatedRect& obj_ellipse);

  void trackerBoxDisplay(cv::Mat& in_frame, ProtoObject& cur_obj, cv::RotatedRect& obj_ellipse);

  void trackerDisplay(cv::Mat& in_frame, tabletop_pushing::VisFeedbackPushTrackingFeedback& state, ProtoObject& obj,
                      bool other_color=false);

 protected:
  void updateHeading(tabletop_pushing::VisFeedbackPushTrackingFeedback& state, bool init_state);
  ProtoObject matchToTargetObject(ProtoObjects& objects, cv::Mat& in_frame, bool init=false);
  cv::Mat getTableMask(XYZPointCloud& cloud, XYZPointCloud& table_cloud, cv::Size mask_size,
                       XYZPointCloud& obj_cloud);
  GMM buildColorModel(XYZPointCloud& cloud, cv::Mat& frame, int num_clusters);
  boost::shared_ptr<PointCloudSegmentation> pcl_segmenter_;
  boost::shared_ptr<ArmObjSegmentation> arm_segmenter_;
  int num_downsamples_;
  bool initialized_;
  int frame_count_;
  int upscale_;
  double previous_time_;
  ProtoObject previous_obj_;
  tabletop_pushing::VisFeedbackPushTrackingFeedback previous_state_;
  tabletop_pushing::VisFeedbackPushTrackingFeedback initial_state_;
  tabletop_pushing::VisFeedbackPushTrackingFeedback previous_centroid_state_;
  bool use_displays_;
  bool write_to_disk_;
  std::string base_output_path_;
  int record_count_;
  bool swap_orientation_;
  bool paused_;
  int frame_set_count_;
  std::string camera_frame_;
  bool use_cv_ellipse_fit_;
  bool use_mps_segmentation_;
  bool obj_saved_;
  GMM obj_color_model_;
  GMM table_color_model_;
  bool have_obj_color_model_;
  bool have_table_color_model_;
  bool use_graphcut_arm_seg_;
  double hull_alpha_;
  XYZPointCloud previous_hull_cloud_;
  ObjectFeaturePointModel obj_feature_point_model_;
  cv::Mat feature_point_morph_element_;
  cv::GoodFeaturesToTrackDetector feature_detector_;
#ifdef USE_ORB
  cv::ORB feature_extractor_;
#else
  cv::BriefDescriptorExtractor feature_extractor_;
#endif
  cv::BFMatcher matcher_;
  cv::Mat init_frame_;
  Eigen::Matrix4f previous_transform_;
  // pcl16::IterativeClosestPoint<pcl16::PointXYZ, pcl16::PointXYZ> feature_point_icp_;
  pcl16::registration::TransformationEstimationSVD<pcl16::PointXYZ, pcl16::PointXYZ> feature_point_transform_est_;
  double ratio_test_thresh_;
  double segment_search_radius_;
  double feature_point_max_ransac_iters_;
  double feature_point_inlier_squared_dist_thresh_;
  double feature_point_ransac_inlier_percent_thresh_;
};
};
#endif // object_tracker_25d_h_DEFINED
