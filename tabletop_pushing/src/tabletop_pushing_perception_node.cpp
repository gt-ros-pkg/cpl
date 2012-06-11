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
#include <cv_bridge/CvBridge.h>
#include <actionlib/server/simple_action_server.h>

// TF
#include <tf/transform_listener.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/io/io.h>
#include <pcl_ros/transforms.h>
#include <pcl/ros/conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

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
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>

// STL
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <iostream>
#include <utility>
#include <float.h>
#include <math.h>
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND
#include <cmath>

// Debugging IFDEFS
#define DISPLAY_INPUT_COLOR 1
// #define DISPLAY_INPUT_DEPTH 1
// #define DISPLAY_WORKSPACE_MASK 1
#define DISPLAY_PROJECTED_OBJECTS 1
#define DISPLAY_CHOSEN_BOUNDARY 1
#define DISPLAY_3D_BOUNDARIES 1
#define DISPLAY_PUSH_VECTOR 1
#define DISPLAY_WAIT 1
#define DEBUG_PUSH_HISTORY 1
#define randf() static_cast<float>(rand())/RAND_MAX

using boost::shared_ptr;
using tabletop_pushing::LearnPush;
using tabletop_pushing::LocateTable;
using tabletop_pushing::PushVector;
using geometry_msgs::PoseStamped;
using geometry_msgs::PointStamped;
using geometry_msgs::Pose2D;
typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                        sensor_msgs::Image,
                                                        sensor_msgs::PointCloud2> MySyncPolicy;
typedef pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>
TransformEstimator;
using tabletop_pushing::PointCloudSegmentation;
using tabletop_pushing::ProtoObject;
using tabletop_pushing::ProtoObjects;
using cpl_visual_features::upSample;
using cpl_visual_features::downSample;
typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;
typedef tabletop_pushing::VisFeedbackPushTrackingGoal PushTrackerGoal;
typedef tabletop_pushing::VisFeedbackPushTrackingResult PushTrackerResult;
typedef tabletop_pushing::VisFeedbackPushTrackingAction PushTrackerAction;

struct Tracker25DKeyPoint
{
  typedef std::vector<float> FeatureVector;
  Tracker25DKeyPoint() : point2D_(NULL_X, NULL_Y), point3D_(0.0,0.0,0.0),
                         delta2D_(0,0), delta3D_(0.0,0.0,0.0)
  {
  }

  Tracker25DKeyPoint(cv::Point point2D, pcl::PointXYZ point3D,
                     FeatureVector descriptor) :
      point2D_(point2D), point3D_(point3D), descriptor_(descriptor),
      delta2D_(0,0), delta3D_(0,0,0)
  {
  }

  void updateVelocities(Tracker25DKeyPoint prev)
  {
    if (prev.point2D_.x == NULL_X && prev.point2D_.x == NULL_Y)
    {
      delta2D_.x = 0;
      delta2D_.y = 0;
      delta3D_.x = 0.0;
      delta3D_.y = 0.0;
      delta3D_.z = 0.0;
      return;
    }
    delta2D_ = point2D_ - prev.point2D_;
    delta3D_.x = point3D_.x - prev.point3D_.x;
    delta3D_.y = point3D_.y - prev.point3D_.y;
    delta3D_.z = point3D_.z - prev.point3D_.z;
  }

  pcl::PointXYZ getPrevious3DPoint()
  {
    pcl::PointXYZ prev;
    prev.x = point3D_.x - delta3D_.x;
    prev.y = point3D_.y - delta3D_.y;
    prev.z = point3D_.z - delta3D_.z;
    return prev;
  }
  static const int NULL_X = -1;
  static const int NULL_Y = -1;
  cv::Point point2D_;
  pcl::PointXYZ point3D_;
  FeatureVector descriptor_;
  cv::Point delta2D_;
  pcl::PointXYZ delta3D_;
};

class ObjectTracker25D
{
 protected:
  typedef std::vector<Tracker25DKeyPoint> KeyPoints;
  typedef Tracker25DKeyPoint KeyPoint;
  typedef Tracker25DKeyPoint::FeatureVector FeatureVector;
  typedef std::vector<FeatureVector> FeatureVectors;

 public:
  ObjectTracker25D(int num_downsamples = 0,
                   double ratio_thresh=0.5, double match_thresh=128,
                   int fast_thresh=9, bool extended_feature=true,
                   int max_ransac_iter=100,
                   double sufficient_support_percent=0.7,
                   double support_dist_thresh=0.03) :
      num_downsamples_(num_downsamples), initialized_(false),
      fast_thresh_(fast_thresh), surf_(0.05, 4, 2, extended_feature),
      ratio_threshold_(ratio_thresh), match_score_threshold_(match_thresh),
      frame_count_(0), max_ransac_iter_(max_ransac_iter),
      sufficient_support_percent_(sufficient_support_percent),
      support_dist_thresh_(support_dist_thresh)
  {
    upscale_ = std::pow(2,num_downsamples_);
  }

  PushTrackerState initTracks(cv::Mat& in_frame, cv::Mat& obj_mask,
                              XYZPointCloud& cloud)
  {
    initialized_ = false;
    // Update current points and descriptors
    // cv::Mat fake_mask(obj_mask.size(), CV_8UC1, cv::Scalar(255));
    // cur_all_keys_ = extractFeatures(in_frame, fake_mask, cloud);
    // prev_all_keys_ = cur_all_keys_;

    init_obj_keys_ = extractFeatures(in_frame, obj_mask, cloud);
    cur_obj_keys_ = init_obj_keys_;
    prev_obj_keys_ = init_obj_keys_;

    // TODO: Create bounding box for volume estimation in tracking
    initialized_ = true;
    frame_count_ = 0;
    PushTrackerState state;
    state.header.seq = 0;
    state.header.stamp = ros::Time::now();
    state.header.frame_id = cloud.header.frame_id;
    state.x = estimateCentroid(init_obj_keys_);
    state.x_dot.x = 0.0;
    state.x_dot.y = 0.0;
    state.x_dot.theta = 0.0;
    previous_time_ = state.header.stamp.toSec();
    return state;
  }

  PushTrackerState updateTracks(cv::Mat& in_frame, cv::Mat& obj_mask,
                                XYZPointCloud& cloud)
  {
    if (!initialized_)
    {
      return initTracks(in_frame, obj_mask, cloud);
    }
    prev_obj_keys_ = cur_obj_keys_;
    // Get current features
    KeyPoints extracted_points = extractFeatures(in_frame, obj_mask, cloud);
    // Match to current object set
    KeyPoints obj_matches = matchFeatures(extracted_points, prev_obj_keys_);

    // Display matches and individual tracks
    drawMatches(in_frame, obj_matches, "-obj");

    // cur_all_keys_ = extracted_points;
    // KeyPoints all_matches = matchFeatures(extracted_points, prev_all_keys_);
    // prev_all_keys_ = cur_all_keys_;
    // drawMatches(in_frame, all_matches, "-all");

    Eigen::Matrix4f transform = estimateMotionVector(obj_matches, in_frame);

    float delta_x = transform(0,3);
    float delta_y = transform(1,3);
    float tr_a = (transform(0,0)+transform(1,1)+transform(2,2));
    float delta_theta = std::acos((tr_a - 1.0)/2.0);

    // Update model

    PushTrackerState state;
    state.header.seq = frame_count_;
    state.header.stamp = ros::Time::now();
    state.header.frame_id = cloud.header.frame_id;

    // TODO: Set object centroid
    state.x = estimateCentroid(obj_matches);

    // Convert delta_x to x_dot
    double delta_t = state.header.stamp.toSec() - previous_time_;
    state.x_dot.x = delta_x/delta_t;
    state.x_dot.y = delta_y/delta_t;
    state.x_dot.theta = delta_theta/delta_t;
    previous_time_ = state.header.stamp.toSec();
    cur_obj_keys_ = obj_matches;
    frame_count_++;

    // ROS_INFO_STREAM("x: (" << state.x << ")");
    // ROS_INFO_STREAM("x_dot: (" << state.x_dot << ")");

    return state;
  }

  Pose2D estimateCentroid(KeyPoints& points)
  {
    Pose2D centroid;
    centroid.x = 0.0;
    centroid.y = 0.0;
    centroid.theta = 0.0;
    if (points.size() < 1)
    {
      return centroid;
    }
    for (unsigned int i; i < points.size(); ++i)
    {
      if (isnan(points[i].point3D_.x) || isnan(points[i].point3D_.x))
      {
        continue;
      }
      centroid.x += points[i].point3D_.x;
      centroid.y += points[i].point3D_.x;
    }

    centroid.x /= points.size();
    centroid.y /= points.size();
    // TODO: Estimate theta somehow...
    return centroid;
  }

  KeyPoints extractFeatures(cv::Mat& in_frame, cv::Mat& obj_mask,
                            XYZPointCloud& cloud)
  {
    cv::Mat bw_frame(in_frame.size(), CV_8UC1);
    if (in_frame.channels() == 3)
    {
      cv::cvtColor(in_frame, bw_frame, CV_BGR2GRAY);
    }
    else
    {
      bw_frame = in_frame;
    }
    // TODO: Compare points returned to extracting features from pre-masked
    // image
    std::vector<cv::KeyPoint> key_points;
    cv::FAST(bw_frame, key_points, fast_thresh_);

    // Remove keypoints not on the object mask
    for (unsigned int i = 0; i < key_points.size();)
    {
      if (obj_mask.at<uchar>(key_points[i].pt.y,
                             key_points[i].pt.x) == 0)
      {
        key_points.erase(key_points.begin() + i);
      }
      else
      {
        i++;
      }
    }
    // Extract descriptors for those points on the object
    FeatureVector raw_descriptors;
    surf_(bw_frame, obj_mask, key_points, raw_descriptors, true);
    // Populate feature vectors and key point locations
    const int descriptor_length = float(raw_descriptors.size()) /
        key_points.size();
    KeyPoints obj_key_points;
    FeatureVectors feats;
    for (int i = 0; i < key_points.size(); ++i)
    {
      FeatureVector f;
      for (int j = 0; j < descriptor_length; ++j)
      {
        f.push_back(raw_descriptors[i*descriptor_length + j]);
      }
      // Extract 3D locations of points
      KeyPoint k(key_points[i].pt, getPoint3D(key_points[i].pt, cloud), f);
      obj_key_points.push_back(k);
      feats.push_back(f);
    }

    cv::Mat disp_img;
    cv::drawKeypoints(in_frame, key_points, disp_img, cv::Scalar(0,255,0));
    if (!initialized_)
    {
      cv::imshow("Initial Features", disp_img);
    }
    else
    {
      cv::imshow("Extracted Features", disp_img);
    }
    return obj_key_points;
  }

  KeyPoints matchFeatures(KeyPoints& extracted, KeyPoints& previous)
  {
    KeyPoints matched;
    int null_count = 0;
    // Match extracted against previous;
    for (unsigned int i = 0; i < previous.size(); ++i)
    {
      int match_idx = ratioTest(previous[i], extracted, ratio_threshold_,
                                match_score_threshold_);
      // NOTE: Ignore bad matches
      if (match_idx < 0)
      {
        KeyPoint null;
        matched.push_back(null);
        matched[i].descriptor_ = previous[i].descriptor_;
        null_count++;
      }
      else
      {
        KeyPoint match = extracted[match_idx];
        match.updateVelocities(previous[i]);
        matched.push_back(match);
      }
    }
    return matched;
  }

  Eigen::Matrix4f estimateMotionVector(KeyPoints& tracks, cv::Mat& frame)
  {
    XYZPointCloud current_pts;
    XYZPointCloud previous_pts;
    current_pts.width = tracks.size();
    current_pts.height = 1;
    current_pts.is_dense = false;
    current_pts.resize(current_pts.width*current_pts.height);
    previous_pts.width = tracks.size();
    previous_pts.height = 1;
    previous_pts.is_dense = false;
    previous_pts.resize(previous_pts.width*previous_pts.height);
    std::vector<int> current_idx;
    std::vector<int> previous_idx;
    for(unsigned int i=0; i < tracks.size(); ++i)
    {
      current_pts.points[i] = tracks[i].point3D_;
      previous_pts.points[i] = tracks[i].getPrevious3DPoint();
      if (tracks[i].point2D_.x == Tracker25DKeyPoint::NULL_X ||
          tracks[i].point2D_.y == Tracker25DKeyPoint::NULL_Y ||
          tracks[i].point2D_.x < 0 || tracks[i].point2D_.y < 0)
      {
      }
      else if (isnan(tracks[i].point3D_.x) || isnan(tracks[i].point3D_.y) || isnan(tracks[i].point3D_.z) )
      {
        // // TODO: Look if any points in the downsampled neighborhood have values
        // ROS_ERROR_STREAM("Nan in point: " << tracks[i].point3D_);
        // ROS_ERROR_STREAM("2D point: " << tracks[i].point2D_);
        // ROS_ERROR_STREAM("2D delta: " << tracks[i].delta2D_);
        // ROS_ERROR_STREAM("3D delta: " << tracks[i].delta3D_);
      }
      else if (isnan(previous_pts.points[i].x) ||
               isnan(previous_pts.points[i].y) ||
               isnan(previous_pts.points[i].z))
      {
        // // TODO: Look if any points in the downsampled neighborhood have values
        // ROS_ERROR_STREAM("Nan in prev point: " << previous_pts.points[i]);
        // ROS_ERROR_STREAM("3D point: " << tracks[i].point3D_);
        // ROS_ERROR_STREAM("2D point: " << tracks[i].point2D_);
        // ROS_ERROR_STREAM("2D delta: " << tracks[i].delta2D_);
        // ROS_ERROR_STREAM("3D delta: " << tracks[i].delta3D_);
      }
      else
      {
        current_idx.push_back(i);
        previous_idx.push_back(i);
        // ROS_INFO_STREAM("Current pt3D: " << current_pts.points[i]);
        // ROS_INFO_STREAM("Previous pt3D: " << previous_pts.points[i]);
        // TODO: If negative x, then print all info
        if (tracks[i].point3D_.x < 0)
        {
          ROS_WARN_STREAM("Current point has negative 3D x: " <<
                          tracks[i].point3D_);
        }
        if (previous_pts.points[i].x < 0)
        {
          ROS_WARN_STREAM("Previous point has negative 3D x.");
          ROS_WARN_STREAM("Tracked point is: " << tracks[i].point3D_);
          ROS_WARN_STREAM("Tracked 2D point is: " << tracks[i].point2D_);
          ROS_WARN_STREAM("Tracked 2D vector is: " << tracks[i].delta2D_);
          ROS_WARN_STREAM("Tracked 3D vector is: " << tracks[i].delta3D_);
          ROS_WARN_STREAM("Current point is: " << current_pts.points[i]);
          ROS_WARN_STREAM("Previous point is: " << previous_pts.points[i]
                          << "\n");
        }
      }
    }

    int max_support = 0;
    std::vector<int> best_support;
    Eigen::Matrix4f best_transform;
    if (current_idx.size() < 2)
    {
      ROS_ERROR_STREAM("Too few matches to estimate transform");
      return Eigen::Matrix4f::Identity();
    }
    for (int i = 0; i < max_ransac_iter_; ++i)
    {
      // Choose 2 random, unique indices
      int idx0 = current_idx[rand() % current_idx.size()];
      int idx1 = -1;
      do
      {
        idx1 = current_idx[rand() % current_idx.size()];
      } while (idx1 == idx0);
      std::vector<int> rand_idx;
      rand_idx.push_back(idx0);
      rand_idx.push_back(idx1);
      // Compute transform from random pts
      Eigen::Matrix4f transform = estimateTransform(previous_pts, current_pts,
                                                    rand_idx, rand_idx);
      std::vector<int> support = determineSupport(previous_pts, current_pts,
                                                  previous_idx, current_idx,
                                                  transform);
      if (support.size() > max_support)
      {
        max_support = support.size();
        best_transform = transform;
        best_support = support;
      }
      // Exit if support percentage is above a certain threhsold
      if (best_support.size() >
          sufficient_support_percent_*current_pts.size())
      {
        break;
      }
    }

    // Estimate final transform with least squares (SVD)
    // ROS_INFO_STREAM("Number of support pts is: " << best_support.size()
    //                 << " / " << tracks.size());

    Eigen::Matrix4f final_transform = estimateTransform(previous_pts,
                                                        current_pts,
                                                        best_support,
                                                        best_support);
    // ROS_INFO_STREAM("Best guess transform is: \n" << final_transform << "\n");

    // TODO: Display transform (applied to centroid of points?)
    // TODO: Need to project 3D into the image...
    cv::Mat disp_img;
    frame.copyTo(disp_img);
    for (unsigned int i = 0; i < best_support.size(); ++i)
    {
      int idx = best_support[i];
      cv::circle(disp_img, tracks[idx].point2D_, 4, cv::Scalar(0,0,255));
    }
    cv::imshow("Support Points", disp_img);
    return final_transform;
  }

  Eigen::Matrix4f estimateTransform(XYZPointCloud& previous_pts,
                                    XYZPointCloud& current_pts,
                                    std::vector<int>& previous_indices,
                                    std::vector<int>& current_indices)
  {
    TransformEstimator estimator;
    Eigen::Matrix4f transform;
    estimator.estimateRigidTransformation(previous_pts, previous_indices,
                                          current_pts, current_indices,
                                          transform);
    return transform;
  }

  std::vector<int> determineSupport(XYZPointCloud& previous_pts,
                                    XYZPointCloud& current_pts,
                                    std::vector<int>& previous_indices,
                                    std::vector<int>& current_indices,
                                    Eigen::Matrix4f& transform)
  {
    std::vector<int> support_idx;
    for (unsigned int i = 0; i < previous_indices.size(); ++i)
    {
      int idx = previous_indices[i];
      double fit_error = computeReprojectionError(previous_pts.points[idx],
                                                  current_pts.points[idx],
                                                  transform);
      if (fit_error < support_dist_thresh_)
      {
        support_idx.push_back(idx);
      }
    }
    return support_idx;
  }

  //
  // Helper functions
  //
  double computeReprojectionError(pcl::PointXYZ& prev_pt, pcl::PointXYZ& cur_pt,
                                  Eigen::Matrix4f& transform)
  {
    Eigen::Vector4f x_t0(prev_pt.x, prev_pt.y, prev_pt.z, 1.0);
    Eigen::Vector4f x_t1(cur_pt.x, cur_pt.y, cur_pt.z, 1.0);
    Eigen::Vector4f x_t1_hat = transform*x_t0;
    Eigen::Vector4f error_vec = x_t1_hat - x_t1;
    return error_vec.norm();
  }


  int ratioTest(KeyPoint& a, KeyPoints& bList, double ratio_threshold = 0.5,
                double match_threshold=1.0)
  {
    double best_score = 1000000;
    double second_best = 1000000;
    int best_index = -1;

    for (unsigned int b = 0; b < bList.size(); ++b) {
      double score = 0;
      score = SSD(a.descriptor_, bList[b].descriptor_);

      if (score < best_score) {
        second_best = best_score;
        best_score = score;
        best_index = b;
      } else if (score < second_best) {
        second_best = score;
      }
    }
    if ( second_best == 0 ||
         best_score / second_best > ratio_threshold) {
      best_index = -1;
    }
    if ( best_score > match_threshold) {
      best_index = -1;
    }
    return best_index;
  }

  double SSD(FeatureVector& a, FeatureVector& b)
  {
    double diff = 0;

    for (unsigned int i = 0; i < a.size(); ++i) {
      float delta = a[i] - b[i];
      diff += delta*delta;
    }

    return diff;
  }

  pcl::PointXYZ getPoint3D(cv::Point pt, XYZPointCloud& cloud) const
  {
    // Compensate for downsampling
    return cloud.at(pt.x*upscale_, pt.y*upscale_);
  }

  //
  // I/O Functions
  //
  void drawMatches(cv::Mat& in_frame, KeyPoints& matches, std::string append="")
  {
    cv::Mat disp_img;
    in_frame.copyTo(disp_img);
    for (unsigned int i = 0; i < matches.size(); ++i)
    {
      if (matches[i].point2D_.x < 0 || matches[i].point2D_.y < 0)
      {
        continue;
      }
      cv::line(disp_img, matches[i].point2D_,
               matches[i].point2D_ + matches[i].delta2D_, cv::Scalar(0,255,0));
      cv::circle(disp_img, matches[i].point2D_, 4, cv::Scalar(0,255,0));
    }
    std::stringstream title;
    title << "Tracker 2.5D Matches" << append;
    cv::imshow(title.str(), disp_img);
  }

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
  }

  void setNumDownsamples(int num_downsamples)
  {
    num_downsamples_ = num_downsamples;
    upscale_ = std::pow(2,num_downsamples_);
  }

 protected:
  int num_downsamples_;
  bool initialized_;
  int fast_thresh_;
  KeyPoints init_obj_keys_;
  KeyPoints prev_obj_keys_;
  KeyPoints cur_obj_keys_;
  KeyPoints prev_all_keys_;
  KeyPoints cur_all_keys_;
  cv::SURF surf_;
  double ratio_threshold_;
  double match_score_threshold_;
  int frame_count_;
  int upscale_;
  int max_ransac_iter_;
  double sufficient_support_percent_;
  double support_dist_thresh_;
  double previous_time_;
};

class TabletopPushingPerceptionNode
{
 public:
  TabletopPushingPerceptionNode(ros::NodeHandle &n) :
      n_(n), n_private_("~"),
      image_sub_(n, "color_image_topic", 1),
      depth_sub_(n, "depth_image_topic", 1),
      cloud_sub_(n, "point_cloud_topic", 1),
      sync_(MySyncPolicy(15), image_sub_, depth_sub_, cloud_sub_),
      as_(n, "push_tracker", false),
      have_depth_data_(false),
      camera_initialized_(false), recording_input_(false), record_count_(0),
      callback_count_(0)
  {
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());
    pcl_segmenter_ = shared_ptr<PointCloudSegmentation>(
        new PointCloudSegmentation(tf_));
    // Get parameters from the server
    n_private_.param("crop_min_x", crop_min_x_, 0);
    n_private_.param("crop_max_x", crop_max_x_, 640);
    n_private_.param("crop_min_y", crop_min_y_, 0);
    n_private_.param("crop_max_y", crop_max_y_, 480);
    n_private_.param("display_wait_ms", display_wait_ms_, 3);
    n_private_.param("use_displays", use_displays_, false);
    n_private_.param("write_input_to_disk", write_input_to_disk_, false);
    n_private_.param("write_to_disk", write_to_disk_, false);
    n_private_.param("min_workspace_x", min_workspace_x_, 0.0);
    n_private_.param("min_workspace_y", min_workspace_y_, 0.0);
    n_private_.param("min_workspace_z", min_workspace_z_, 0.0);
    n_private_.param("max_workspace_x", max_workspace_x_, 0.0);
    n_private_.param("max_workspace_y", max_workspace_y_, 0.0);
    n_private_.param("max_workspace_z", max_workspace_z_, 0.0);
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_,
                     default_workspace_frame);

    std::string output_path_def = "~";
    n_private_.param("img_output_path", base_output_path_, output_path_def);

    n_private_.param("min_table_z", pcl_segmenter_->min_table_z_, -0.5);
    n_private_.param("max_table_z", pcl_segmenter_->max_table_z_, 1.5);
    pcl_segmenter_->min_workspace_x_ = min_workspace_x_;
    pcl_segmenter_->max_workspace_x_ = max_workspace_x_;
    pcl_segmenter_->min_workspace_z_ = min_workspace_z_;
    pcl_segmenter_->max_workspace_z_ = max_workspace_z_;
    n_private_.param("moved_count_thresh", pcl_segmenter_->moved_count_thresh_,
                     1);

    n_private_.param("autostart_pcl_segmentation", autorun_pcl_segmentation_,
                     false);

    n_private_.param("num_downsamples", num_downsamples_, 2);
    pcl_segmenter_->num_downsamples_ = num_downsamples_;

    std::string cam_info_topic_def = "/kinect_head/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_,
                     cam_info_topic_def);
    n_private_.param("table_ransac_thresh", pcl_segmenter_->table_ransac_thresh_,
                     0.01);
    n_private_.param("table_ransac_angle_thresh",
                     pcl_segmenter_->table_ransac_angle_thresh_, 30.0);
    n_private_.param("pcl_cluster_tolerance", pcl_segmenter_->cluster_tolerance_,
                     0.25);
    n_private_.param("pcl_difference_thresh", pcl_segmenter_->cloud_diff_thresh_,
                     0.01);
    n_private_.param("pcl_min_cluster_size", pcl_segmenter_->min_cluster_size_,
                     100);
    n_private_.param("pcl_max_cluster_size", pcl_segmenter_->max_cluster_size_,
                     2500);
    n_private_.param("pcl_voxel_downsample_res", pcl_segmenter_->voxel_down_res_,
                     0.005);
    n_private_.param("pcl_cloud_intersect_thresh",
                     pcl_segmenter_->cloud_intersect_thresh_, 0.005);
    n_private_.param("pcl_concave_hull_alpha", pcl_segmenter_->hull_alpha_,
                     0.1);
    n_private_.param("use_pcl_voxel_downsample",
                     pcl_segmenter_->use_voxel_down_, true);
    n_private_.param("icp_max_iters", pcl_segmenter_->icp_max_iters_, 100);
    n_private_.param("icp_transform_eps", pcl_segmenter_->icp_transform_eps_,
                     0.0);
    n_private_.param("icp_max_cor_dist",
                     pcl_segmenter_->icp_max_cor_dist_, 1.0);
    n_private_.param("icp_ransac_thresh",
                     pcl_segmenter_->icp_ransac_thresh_, 0.015);

    double ratio_thresh;
    double match_score_thresh;
    int fast_thresh;
    bool extended_feats;
    int max_ransac_iter;
    double support_percent;
    double support_dist;
    n_private_.param("obj_tracker_ratio_threshold", ratio_thresh, 0.5);
    n_private_.param("obj_tracker_score_threshold", match_score_thresh, 128.0);
    n_private_.param("obj_tracker_fast_threshold", fast_thresh, 9);
    n_private_.param("obj_tracker_extended_feats", extended_feats, true);
    n_private_.param("obj_tracker_extended_feats", extended_feats, true);
    n_private_.param("obj_tracker_max_ransac_iter", max_ransac_iter, 100);
    n_private_.param("obj_tracker_ransac_support_percent", support_percent, 0.7);
    n_private_.param("obj_tracker_ransac_dist_thresh", support_dist, 0.03);
    n_private_.param("push_tracker_dist_thresh", tracker_dist_thresh_, 0.01);
    n_private_.param("push_tracker_angle_thresh", tracker_angle_thresh_, 0.01);

    // Initialize classes requiring parameters
    obj_tracker_ = shared_ptr<ObjectTracker25D>(
        new ObjectTracker25D(num_downsamples_, ratio_thresh, match_score_thresh,
                             fast_thresh, extended_feats, max_ransac_iter,
                             support_percent, support_dist));

    // Setup ros node connections
    sync_.registerCallback(&TabletopPushingPerceptionNode::sensorCallback,
                           this);
    push_pose_server_ = n_.advertiseService(
        "get_learning_push_vector",
        &TabletopPushingPerceptionNode::learnPushCallback, this);
    table_location_server_ = n_.advertiseService(
        "get_table_location", &TabletopPushingPerceptionNode::getTableLocation,
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
    cv::Mat color_frame(bridge_.imgMsgToCv(img_msg));
    cv::Mat depth_frame(bridge_.imgMsgToCv(depth_msg));

    // Swap kinect color channel order
    cv::cvtColor(color_frame, color_frame, CV_RGB2BGR);

    // Transform point cloud into the correct frame and convert to PCL struct
    XYZPointCloud cloud;
    pcl::fromROSMsg(*cloud_msg, cloud);
    tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
                          cloud.header.stamp, ros::Duration(0.5));
    pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);
    // ROS_INFO_STREAM("Transformed point cloud");
    // tf::StampedTransform transform;
    // tf_->lookupTransform(cloud.header.frame_id, workspace_frame_, ros::Time(0),
    //                      transform);
    // tf::Vector3 trans = transform.getOrigin();
    // tf::Quaternion rot = transform.getRotation();
    // ROS_INFO_STREAM("Transform trans: (" << trans.x() << ", " << trans.y() <<
    //                 ", " << trans.z() << ")");
    // ROS_INFO_STREAM("Transform rot: (" << rot.x() << ", " << rot.y() <<
    //                 ", " << rot.z() << ", " << rot.w() << ")");

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

    // Downsample everything first
    cv::Mat color_frame_down = downSample(color_frame, num_downsamples_);
    cv::Mat depth_frame_down = downSample(depth_frame, num_downsamples_);
    cv::Mat workspace_mask_down = downSample(workspace_mask, num_downsamples_);

    // Save internally for use in the service callback
    prev_color_frame_ = cur_color_frame_.clone();
    prev_depth_frame_ = cur_depth_frame_.clone();
    prev_workspace_mask_ = cur_workspace_mask_.clone();
    prev_camera_header_ = cur_camera_header_;

    // Update the current versions
    cur_color_frame_ = color_frame_down.clone();
    cur_depth_frame_ = depth_frame_down.clone();
    cur_workspace_mask_ = workspace_mask_down.clone();
    cur_point_cloud_ = cloud;
    have_depth_data_ = true;
    cur_camera_header_ = img_msg->header;
    pcl_segmenter_->cur_camera_header_ = cur_camera_header_;

    if (obj_tracker_->isInitialized())
    {
      PushTrackerState tracker_state = obj_tracker_->updateTracks(
          cur_color_frame_, cur_workspace_mask_, cur_point_cloud_);

      // make sure that the action hasn't been canceled
      if (as_.isActive())
      {
        as_.publishFeedback(tracker_state);
        // Check for goal conditions
        float x_dist = fabs(tracker_goal_pose_.x - tracker_state.x.x);
        float y_dist = fabs(tracker_goal_pose_.y - tracker_state.x.y);
        // TODO: Make sub1/2pi diff
        float theta_dist = fabs(tracker_goal_pose_.theta - tracker_state.x.theta);
        if (x_dist < tracker_dist_thresh_ && y_dist < tracker_dist_thresh_ &&
            theta_dist < tracker_angle_thresh_)
        {
          ROS_INFO_STREAM("Cur state: (" << tracker_state.x.x << ", " <<
                          tracker_state.x.y << ", " << tracker_state.x.theta << ")");
          ROS_INFO_STREAM("Desired goal: (" << tracker_goal_pose_.x << ", " <<
                          tracker_goal_pose_.y << ", " << tracker_goal_pose_.theta << ")");
          ROS_INFO_STREAM("Goal error: (" << x_dist << ", " << y_dist << ", "
                          << theta_dist << ")");
          PushTrackerResult res;
          as_.setSucceeded(res);
          stopTracking();
        }
      }
    }

    // Debug stuff
    if (autorun_pcl_segmentation_)
    {
      LearnPush::Request req;
      req.push_angle = randf()*2.0*M_PI-M_PI;
      req.use_goal_pose = false;
      req.rand_angle = false;
      getPushStartPose(req);
    }

    // Display junk
#ifdef DISPLAY_INPUT_COLOR
    if (use_displays_)
    {
      cv::imshow("color", cur_color_frame_);
    }
    // Way too much disk writing!
    if (write_input_to_disk_ && recording_input_)
    {
      std::stringstream out_name;
      out_name << base_output_path_ << "input" << record_count_ << ".png";
      record_count_++;
      cv::imwrite(out_name.str(), cur_color_frame_);
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
#ifdef DISPLAY_WORKSPACE_MASK
    if (use_displays_)
    {
      cv::imshow("workspace_mask", cur_workspace_mask_);
    }
#endif // DISPLAY_WORKSPACE_MASK
#ifdef DISPLAY_WAIT
    if (use_displays_)
    {
      cv::waitKey(display_wait_ms_);
    }
#endif // DISPLAY_WAIT
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
      if (req.initialize)
      {
        record_count_ = 0;
        callback_count_ = 0;
        // Initialize stuff if necessary (i.e. angle to push from)
        res.no_push = true;
        recording_input_ = false;
      }
      else if (req.analyze_previous)
      {
        res = getAnalysisVector(req.push_angle);
        res.no_push = true;
      }
      else
      {
        res = getPushStartPose(req);
        res.no_push = false;
      }
    }
    else
    {
      ROS_ERROR_STREAM("Calling getPushStartPose prior to receiving sensor data.");
      recording_input_ = false;
      res.no_push = true;
      return false;
    }
    return true;
  }

  LearnPush::Response getPushStartPose(LearnPush::Request& req)
  {
    bool rand_angle = req.rand_angle;
    double desired_push_angle = req.push_angle;

    // Segment objects
    ProtoObjects objs = pcl_segmenter_->findTabletopObjects(cur_point_cloud_);
    cv::Mat disp_img = pcl_segmenter_->projectProtoObjectsIntoImage(
        objs, cur_color_frame_.size(), workspace_frame_);
    pcl_segmenter_->displayObjectImage(disp_img, "Objects", true);

    // Assume we care about the biggest currently
    int chosen_idx = 0;
    unsigned int max_size = 0;
    for (unsigned int i = 0; i < objs.size(); ++i)
    {
      if (objs[i].cloud.size() > max_size)
      {
        max_size = objs[i].cloud.size();
        chosen_idx = i;
      }
    }

    LearnPush::Response res;
    if (objs.size() == 0)
    {
      ROS_WARN_STREAM("No objects found");
      res.centroid.x = 0.0;
      res.centroid.y = 0.0;
      res.centroid.z = 0.0;
      return res;
    }
    res.centroid.x = objs[chosen_idx].centroid[0];
    res.centroid.y = objs[chosen_idx].centroid[1];
    res.centroid.z = objs[chosen_idx].centroid[2];

    ROS_INFO_STREAM("Found " << objs.size() << " objects.");
    ROS_INFO_STREAM("Chosen object idx is " << chosen_idx << " with " <<
                    objs[chosen_idx].cloud.size() << " points");

    if (rand_angle)
    {
      // desired_push_angle = randf()*2.0*M_PI-M_PI;
      desired_push_angle = randf()*M_PI-0.5*M_PI;
    }

    // Get straight line from current location to goal pose as start
    if (req.use_goal_pose)
    {
      desired_push_angle = atan2(req.goal_pose.y - res.centroid.y,
                                 req.goal_pose.x - res.centroid.x);
    }

    // Set basic push information
    PushVector p;
    p.header.frame_id = workspace_frame_;
    p.push_angle = desired_push_angle;

    // Get vector through centroid and determine start point and distance
    Eigen::Vector3f push_unit_vec(std::cos(desired_push_angle),
                                  std::sin(desired_push_angle), 0.0f);
    XYZPointCloud intersection = pcl_segmenter_->lineCloudIntersection(
        objs[chosen_idx].cloud, push_unit_vec, objs[chosen_idx].centroid);

    unsigned int min_y_idx = intersection.size();
    unsigned int max_y_idx = intersection.size();
    unsigned int min_x_idx = intersection.size();
    unsigned int max_x_idx = intersection.size();
    float min_y = FLT_MAX;
    float max_y = -FLT_MAX;
    float min_x = FLT_MAX;
    float max_x = -FLT_MAX;
    for (unsigned int i = 0; i < intersection.size(); ++i)
    {
      if (intersection.at(i).y < min_y)
      {
        min_y = intersection.at(i).y;
        min_y_idx = i;
      }
      if (intersection.at(i).y > max_y)
      {
        max_y = intersection.at(i).y;
        max_y_idx = i;
      }
      if (intersection.at(i).x < min_x)
      {
        min_x = intersection.at(i).x;
        min_x_idx = i;
      }
      if (intersection.at(i).x > max_x)
      {
        max_x = intersection.at(i).x;
        max_x_idx = i;
      }
    }
    const double y_dist_obs = max_y - min_y;
    const double x_dist_obs = max_x - min_x;
    int start_idx = min_x_idx;
    int end_idx = max_x_idx;

    if (x_dist_obs > y_dist_obs)
    {
      // Use X index
      if (push_unit_vec[0] > 0)
      {
        // Use min
        start_idx = min_x_idx;
        end_idx = max_x_idx;
      }
      else
      {
        // use max
        start_idx = max_x_idx;
        end_idx = min_x_idx;
      }
    }
    else
    {
      // Use Y index
      if (push_unit_vec[1] > 0)
      {
        // Use min
        start_idx = min_y_idx;
        end_idx = max_y_idx;
      }
      else
      {
        // use max
        start_idx = max_y_idx;
        end_idx = min_y_idx;
      }

    }
    p.start_point.x = intersection.at(start_idx).x;
    p.start_point.y = intersection.at(start_idx).y;
    p.start_point.z = intersection.at(start_idx).z;

    // Get push distance
    if (req.use_goal_pose)
    {
      p.push_dist = hypot(res.centroid.x - req.goal_pose.x,
                          res.centroid.y - req.goal_pose.y);
    }
    else
    {
      p.push_dist = std::sqrt(pcl_segmenter_->sqrDistXY(
          intersection.at(start_idx), intersection.at(end_idx)));
    }
    // Visualize push vector
    displayPushVector(cur_color_frame_, p);
    callback_count_++;
    ROS_INFO_STREAM("Chosen push start point: (" << p.start_point.x << ", "
                    << p.start_point.y << ", " << p.start_point.z << ")");
    ROS_INFO_STREAM("Push dist: " << p.push_dist);
    ROS_INFO_STREAM("Push angle: " << p.push_angle);
    prev_centroid_ = objs[chosen_idx].centroid;
    res.push = p;
    return res;
  }

  LearnPush::Response getAnalysisVector(double desired_push_angle)
  {
    // Segment objects
    ProtoObjects objs = pcl_segmenter_->findTabletopObjects(cur_point_cloud_);
    cv::Mat disp_img = pcl_segmenter_->projectProtoObjectsIntoImage(
        objs, cur_color_frame_.size(), workspace_frame_);
    pcl_segmenter_->displayObjectImage(disp_img, "Objects", true);

    // Assume we care about the biggest currently
    int chosen_idx = 0;
    unsigned int max_size = 0;
    for (unsigned int i = 0; i < objs.size(); ++i)
    {
      if (objs[i].cloud.size() > max_size)
      {
        max_size = objs[i].cloud.size();
        chosen_idx = i;
      }
    }
    LearnPush::Response res;
    if (objs.size() == 0)
    {
      ROS_WARN_STREAM("No objects found");
      res.moved.x = 0.0;
      res.moved.y = 0.0;
      res.moved.z = 0.0;
      res.centroid.x = 0.0;
      res.centroid.y = 0.0;
      res.centroid.z = 0.0;
      return res;
    }

    Eigen::Vector4f move_vec = objs[chosen_idx].centroid - prev_centroid_;
    res.moved.x = move_vec[0];
    res.moved.y = move_vec[1];
    res.moved.z = move_vec[2];
    res.centroid.x = objs[chosen_idx].centroid[0];
    res.centroid.y = objs[chosen_idx].centroid[1];
    res.centroid.z = objs[chosen_idx].centroid[2];

    return res;
  }

  bool startTracking()
  {
    // Segment objects
    ProtoObjects objs = pcl_segmenter_->findTabletopObjects(cur_point_cloud_);
    cv::Mat disp_img = pcl_segmenter_->projectProtoObjectsIntoImage(
        objs, cur_color_frame_.size(), workspace_frame_);
    pcl_segmenter_->displayObjectImage(disp_img, "Objects", true);

    // Assume we care about the biggest currently
    int chosen_idx = 0;
    unsigned int max_size = 0;
    for (unsigned int i = 0; i < objs.size(); ++i)
    {
      if (objs[i].cloud.size() > max_size)
      {
        max_size = objs[i].cloud.size();
        chosen_idx = i;
      }
    }

    // NOTE: disp_image has i+1 for object ids
    cv::Mat obj_mask_raw = (disp_img == (chosen_idx+1));
    cv::Mat obj_mask;
    cv::Mat element(3,3, CV_8UC1, cv::Scalar(255));
    cv::dilate(obj_mask_raw, obj_mask, element);
    cv::erode(obj_mask, obj_mask, element);
    // cv::imshow("obj_mask: raw", obj_mask_raw);
    // cv::imshow("obj_mask: closed", obj_mask);
    PushTrackerState tracker_state = obj_tracker_->initTracks(
        cur_color_frame_, obj_mask, cur_point_cloud_);
    Pose2D obj_pose;
    if (objs.size() == 0)
    {
      ROS_WARN_STREAM("No objects found");
      obj_pose.x = 0.0;
      obj_pose.y = 0.0;
      return false;
    }
    obj_pose.x = objs[chosen_idx].centroid[0];
    obj_pose.y = objs[chosen_idx].centroid[1];

    ROS_INFO_STREAM("Found " << objs.size() << " objects.");
    ROS_INFO_STREAM("Chosen object idx is " << chosen_idx << " with " <<
                    objs[chosen_idx].cloud.size() << " points");
    return true;
  }

  bool stopTracking()
  {
    obj_tracker_->stopTracking();
    bool obj_tracking = false;
    return obj_tracking;
  }

  void pushTrackerGoalCB()
  {
    ROS_INFO_STREAM("pushTrackerGoalCB(): starting tracking");
    bool obj_tracking = startTracking();
    if (!obj_tracking)
    {
      ROS_WARN_STREAM("Nothing to track. Push tracking aborted.");
      as_.setAborted();
      return;
    }
    ROS_INFO_STREAM("Accepting goal");
    boost::shared_ptr<const PushTrackerGoal> tracker_goal = as_.acceptNewGoal();
    // TODO: Transform into workspace frame...
    tracker_goal_pose_ = tracker_goal->desired_pose;
    pushing_arm_ = tracker_goal->which_arm;
    ROS_INFO_STREAM("Accepted goal of " << tracker_goal_pose_);
    ROS_INFO_STREAM("Push with arm " << pushing_arm_);
  }

  void pushTrackerPreemptCB()
  {
    bool obj_tracking = stopTracking();
    ROS_INFO_STREAM("Preempted push tracker");
    // set the action state to preempted
    as_.setPreempted();
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
    return p;
  }

  void displayPushVector(cv::Mat& img, PushVector& push)
  {
    cv::Mat disp_img;
    img.copyTo(disp_img);
    PointStamped start_point;
    start_point.point = push.start_point;
    start_point.header.frame_id = workspace_frame_;
    PointStamped end_point;
    end_point.point.x = start_point.point.x+std::cos(push.push_angle)*push.push_dist;
    end_point.point.y = start_point.point.y+std::sin(push.push_angle)*push.push_dist;
    end_point.point.z = start_point.point.z;
    end_point.header.frame_id = workspace_frame_;

    cv::Point img_start_point = pcl_segmenter_->projectPointIntoImage(
        start_point);
    cv::Point img_end_point = pcl_segmenter_->projectPointIntoImage(
        end_point);
    cv::line(disp_img, img_start_point, img_end_point, cv::Scalar(0,255,0));
    cv::circle(disp_img, img_end_point, 4, cv::Scalar(0,255,0));

    if (use_displays_)
    {
      cv::imshow("push_vector", disp_img);
    }
    if (write_to_disk_)
    {
      // Write to disk to create video output
      std::stringstream push_out_name;
      push_out_name << base_output_path_ << "push_vector" << callback_count_
                    << ".png";
      cv::Mat push_out_img(disp_img.size(), CV_8UC3);
      disp_img.convertTo(push_out_img, CV_8UC3, 255);
      cv::imwrite(push_out_name.str(), push_out_img);
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
  sensor_msgs::CameraInfo cam_info_;
  sensor_msgs::CvBridge bridge_;
  shared_ptr<tf::TransformListener> tf_;
  ros::ServiceServer push_pose_server_;
  ros::ServiceServer table_location_server_;
  actionlib::SimpleActionServer<PushTrackerAction> as_;
  cv::Mat cur_color_frame_;
  cv::Mat cur_depth_frame_;
  cv::Mat cur_workspace_mask_;
  cv::Mat prev_color_frame_;
  cv::Mat prev_depth_frame_;
  cv::Mat prev_workspace_mask_;
  std_msgs::Header cur_camera_header_;
  std_msgs::Header prev_camera_header_;
  XYZPointCloud cur_point_cloud_;
  shared_ptr<PointCloudSegmentation> pcl_segmenter_;
  bool have_depth_data_;
  int crop_min_x_;
  int crop_max_x_;
  int crop_min_y_;
  int crop_max_y_;
  int display_wait_ms_;
  bool use_displays_;
  bool write_input_to_disk_;
  bool write_to_disk_;
  std::string base_output_path_;
  double min_workspace_x_;
  double max_workspace_x_;
  double min_workspace_y_;
  double max_workspace_y_;
  double min_workspace_z_;
  double max_workspace_z_;
  int num_downsamples_;
  std::string workspace_frame_;
  PoseStamped table_centroid_;
  bool camera_initialized_;
  std::string cam_info_topic_;
  bool autorun_pcl_segmentation_;
  bool recording_input_;
  int record_count_;
  int callback_count_;
  Eigen::Vector4f prev_centroid_;
  shared_ptr<ObjectTracker25D> obj_tracker_;
  Pose2D tracker_goal_pose_;
  std::string pushing_arm_;
  double tracker_dist_thresh_;
  double tracker_angle_thresh_;
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

