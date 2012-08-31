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
#include <pcl/common/pca.h>

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
#include <cmath>

// Debugging IFDEFS
#define DISPLAY_INPUT_COLOR 1
// #define DISPLAY_INPUT_DEPTH 1
#define DISPLAY_PROJECTED_OBJECTS 1
#define DISPLAY_CHOSEN_BOUNDARY 1
#define DISPLAY_3D_BOUNDARIES 1
#define DISPLAY_PUSH_VECTOR 1
#define DISPLAY_WAIT 1
#define DEBUG_PUSH_HISTORY 1

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
                                                        sensor_msgs::Image,
                                                        sensor_msgs::PointCloud2> MySyncPolicy;
typedef pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>
TransformEstimator;
using tabletop_pushing::PointCloudSegmentation;
using tabletop_pushing::ProtoObject;
using tabletop_pushing::ProtoObjects;
using cpl_visual_features::upSample;
using cpl_visual_features::downSample;
using cpl_visual_features::subPIAngle;
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

static inline double sqrDistXY(Eigen::Vector4f a, Pose2D b)
{
  const double dx = a[0]-b.x;
  const double dy = a[1]-b.y;
  return dx*dx+dy*dy;
}

class ObjectTracker25D
{
 protected:
  typedef std::vector<Tracker25DKeyPoint> KeyPoints;
  typedef Tracker25DKeyPoint KeyPoint;
  typedef Tracker25DKeyPoint::FeatureVector FeatureVector;
  typedef std::vector<FeatureVector> FeatureVectors;

 public:
  ObjectTracker25D(shared_ptr<PointCloudSegmentation> segmenter, int num_downsamples = 0,
                   bool use_displays=false, bool write_to_disk=false,
                   std::string base_output_path="", std::string camera_frame="",
                   bool use_cv_ellipse = false) :
      pcl_segmenter_(segmenter), num_downsamples_(num_downsamples), initialized_(false),
      frame_count_(0), use_displays_(use_displays), write_to_disk_(write_to_disk),
      base_output_path_(base_output_path), record_count_(0), swap_orientation_(false),
      paused_(false), frame_set_count_(0), camera_frame_(camera_frame),
      use_cv_ellipse_fit_(use_cv_ellipse)
  {
    upscale_ = std::pow(2,num_downsamples_);
  }

  ProtoObject findTargetObject(cv::Mat& in_frame, XYZPointCloud& cloud,
                               bool& no_objects, bool init=false)
  {
    ProtoObjects objs = pcl_segmenter_->findTabletopObjects(cloud);
    if (objs.size() == 0)
    {
      ROS_WARN_STREAM("No objects found");
      ProtoObject empty;
      no_objects = true;
      return empty;
    }

    int chosen_idx = 0;
    if (objs.size() == 1)
    {
    }
    else if (init || frame_count_ == 0)
    {
      // Assume we care about the biggest currently
      unsigned int max_size = 0;
      for (unsigned int i = 0; i < objs.size(); ++i)
      {
        if (objs[i].cloud.size() > max_size)
        {
          max_size = objs[i].cloud.size();
          chosen_idx = i;
        }
      }

      // TODO: Extract color histogram
    }
    else // Find closest object to last time
    {
      double min_dist = 1000.0;
      for (unsigned int i = 0; i < objs.size(); ++i)
      {
        double centroid_dist = pcl_segmenter_->sqrDist(objs[i].centroid, previous_obj_.centroid);
        if (centroid_dist  < min_dist)
        {
          min_dist = centroid_dist;
          chosen_idx = i;
        }
        // TODO: Match color histogram
      }
    }

    if (use_displays_)
    {
      cv::Mat disp_img = pcl_segmenter_->projectProtoObjectsIntoImage(
          objs, in_frame.size(), cloud.header.frame_id);
      pcl_segmenter_->displayObjectImage(disp_img, "Objects", true);
    }

    no_objects = false;
    return objs[chosen_idx];
  }

  cv::RotatedRect findFootprintEllipse(ProtoObject& obj)
  {
    // Get 2D footprint of object and fit an ellipse to it
    std::vector<cv::Point2f> obj_pts;
    for (unsigned int i = 0; i < obj.cloud.size(); ++i)
    {
      obj_pts.push_back(cv::Point2f(obj.cloud[i].x, obj.cloud[i].y));
    }
    ROS_DEBUG_STREAM("Number of points is: " << obj_pts.size());
    cv::RotatedRect obj_ellipse = fitEllipse(obj_pts);
    // cv::RotatedRect obj_ellipse2 = fit2DMassEllipse(obj);
    // ROS_DEBUG_STREAM("obj_ellipse: (" << obj_ellipse.center.x << ", " <<
    //                  obj_ellipse.center.y << ", " <<
    //                  subPIAngle(DEG2RAD(obj_ellipse.angle)) << ")" << "\t(" <<
    //                  obj_ellipse.size.width << ", " << obj_ellipse.size.height
    //                  << ")");
    return obj_ellipse;
  }

  cv::RotatedRect fit2DMassEllipse(ProtoObject& obj)
  {
    pcl::PCA<pcl::PointXYZ> pca;
    XYZPointCloud cloud_no_z;
    ROS_INFO_STREAM("Copying point cloud");
    cloud_no_z.header = obj.cloud.header;
    cloud_no_z.width = obj.cloud.size();
    cloud_no_z.height = 1;
    cloud_no_z.resize(obj.cloud.size());
    for (int i = 0; i < obj.cloud.size(); ++i)
    {
      cloud_no_z[i] = obj.cloud[i];
      cloud_no_z[i].z = 0.0f;
    }

    pca.setInputCloud(cloud_no_z.makeShared());
    Eigen::Vector3f eigen_values = pca.getEigenValues();
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
    Eigen::Vector4f centroid = pca.getMean();
    cv::RotatedRect obj_ellipse;
    obj_ellipse.center.x = centroid[0];
    obj_ellipse.center.y = centroid[1];
    obj_ellipse.angle = RAD2DEG(atan2(eigen_vectors(1,0), eigen_vectors(0,0))-0.5*M_PI);
    // TODO: Set obj_ellipse.size
    // NOTE: major axis is defined by height
    obj_ellipse.size.height = eigen_values(0)*0.05;
    obj_ellipse.size.width = eigen_values(1)*0.05;
    // TODO: Get relative size of eigen1 to eigen2 and use that as ratio between two axes?
    // ROS_INFO_STREAM("Centroid from PCA is: " << centroid);
    // ROS_INFO_STREAM("Eigen vectors from PCA are: " << eigen_vectors);
    // ROS_INFO_STREAM("Eigen values from PCA are: " << eigen_values);
    return obj_ellipse;
  }

  PushTrackerState initTracks(cv::Mat& in_frame, cv::Mat& self_mask, XYZPointCloud& cloud)
  {
    paused_ = false;
    initialized_ = false;
    swap_orientation_ = false;
    bool no_objects = false;
    frame_count_ = 0;
    record_count_ = 0;
    frame_set_count_++;
    ProtoObject cur_obj = findTargetObject(in_frame, cloud,  no_objects, true);
    initialized_ = true;
    PushTrackerState state;
    state.header.seq = 0;
    state.header.stamp = cloud.header.stamp;
    state.header.frame_id = cloud.header.frame_id;
    if (no_objects)
    {
      state.no_detection = true;
    }
    ROS_INFO_STREAM("Finding ellipse");
    cv::RotatedRect obj_ellipse;
    if (use_cv_ellipse_fit_)
    {
      obj_ellipse = findFootprintEllipse(cur_obj);
    }
    else
    {
      obj_ellipse = fit2DMassEllipse(cur_obj);
    }
    state.x.theta = getThetaFromEllipse(obj_ellipse);
    state.x.x = cur_obj.centroid[0];
    state.x.y = cur_obj.centroid[1];
    state.z = cur_obj.centroid[2];
    state.x_dot.x = 0.0;
    state.x_dot.y = 0.0;
    state.x_dot.theta = 0.0;

    if (use_displays_ || write_to_disk_)
    {
      ROS_INFO_STREAM("TrackerIO()");
      trackerIO(in_frame, cur_obj, obj_ellipse);
      // cv::RotatedRect new_obj_ellipse = fit2DMassEllipse(cur_obj);
      // tempTrackerIO(in_frame, cur_obj, new_obj_ellipse);
    }

    ROS_DEBUG_STREAM("x: (" << state.x.x << ", " << state.x.y << ", " <<
                     state.x.theta << ")");
    ROS_DEBUG_STREAM("x_dot: (" << state.x_dot.x << ", " << state.x_dot.y
                     << ", " << state.x_dot.theta << ")\n");

    previous_time_ = state.header.stamp.toSec();
    previous_state_ = state;
    previous_obj_ = cur_obj;
    previous_obj_ellipse_ = obj_ellipse;
    return state;
  }

  double getThetaFromEllipse(cv::RotatedRect& obj_ellipse)
  {
    return subPIAngle(DEG2RAD(obj_ellipse.angle)+0.5*M_PI);
  }

  PushTrackerState updateTracks(cv::Mat& in_frame, cv::Mat& self_mask, XYZPointCloud& cloud)
  {
    if (!initialized_)
    {
      return initTracks(in_frame, self_mask, cloud);
    }
    bool no_objects = false;
    ProtoObject cur_obj = findTargetObject(in_frame, cloud, no_objects);

    // Update model
    PushTrackerState state;
    state.header.seq = frame_count_;
    state.header.stamp = cloud.header.stamp;
    state.header.frame_id = cloud.header.frame_id;

    cv::RotatedRect obj_ellipse;
    if (no_objects)
    {
      state.no_detection = true;
      state.x = previous_state_.x;
      state.x_dot = previous_state_.x_dot;
      state.z = previous_state_.z;
      ROS_WARN_STREAM("Using previous state, but updating time!");
      if (use_displays_ || write_to_disk_)
      {
        trackerIO(in_frame, previous_obj_, previous_obj_ellipse_);
      }
    }
    else
    {
      if (use_cv_ellipse_fit_)
      {
        obj_ellipse = findFootprintEllipse(cur_obj);
      }
      else
      {
        obj_ellipse = fit2DMassEllipse(cur_obj);
      }
      state.x.theta = getThetaFromEllipse(obj_ellipse);
      state.x.x = cur_obj.centroid[0];
      state.x.y = cur_obj.centroid[1];
      state.z = cur_obj.centroid[2];

      if(swap_orientation_)
      {
        if(state.x.theta > 0.0)
          state.x.theta += - M_PI;
        else
          state.x.theta += M_PI;
      }

      if ((state.x.theta > 0) != (previous_state_.x.theta > 0))
      {
        if ((fabs(state.x.theta) > M_PI*0.25 &&
             fabs(state.x.theta) < (M_PI*0.75 )) ||
            (fabs(previous_state_.x.theta) > 1.0 &&
             fabs(previous_state_.x.theta) < (M_PI - 0.5)))
        {
          swap_orientation_ = !swap_orientation_;
          // We either need to swap or need to undo the swap
          if(state.x.theta > 0.0)
            state.x.theta += -M_PI;
          else
            state.x.theta += M_PI;
        }
      }

      // Convert delta_x to x_dot
      double delta_x = state.x.x - previous_state_.x.x;
      double delta_y = state.x.y - previous_state_.x.y;
      double delta_theta = subPIAngle(state.x.theta - previous_state_.x.theta);
      double delta_t = state.header.stamp.toSec() - previous_time_;
      state.x_dot.x = delta_x/delta_t;
      state.x_dot.y = delta_y/delta_t;
      state.x_dot.theta = delta_theta/delta_t;

      ROS_DEBUG_STREAM("x: (" << state.x.x << ", " << state.x.y << ", " <<
                       state.x.theta << ")");
      ROS_DEBUG_STREAM("x_dot: (" << state.x_dot.x << ", " << state.x_dot.y
                       << ", " << state.x_dot.theta << ")");

      if (use_displays_ || write_to_disk_)
      {
        trackerIO(in_frame, cur_obj, obj_ellipse);
        // cv::RotatedRect new_obj_ellipse = fit2DMassEllipse(cur_obj);
        // tempTrackerIO(in_frame, cur_obj, new_obj_ellipse);
      }
      previous_obj_ = cur_obj;
      previous_obj_ellipse_ = obj_ellipse;
    }
    previous_time_ = state.header.stamp.toSec();
    previous_state_ = state;
    frame_count_++;
    record_count_++;
    return state;
  }

  void pausedUpdate(cv::Mat in_frame)
  {
    if (use_displays_ || write_to_disk_)
    {
      trackerIO(in_frame, previous_obj_, previous_obj_ellipse_);
    }
    record_count_++;
  }

  //
  // Helper functions
  //

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

  PushTrackerState getMostRecentState() const
  {
    return previous_state_;
  }

  ProtoObject getMostRecentObject() const
  {
    return previous_obj_;
  }

  cv::RotatedRect getMostRecentEllipse() const
  {
    return previous_obj_ellipse_;
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

 protected:
  //
  // I/O Functions
  //

  void trackerIO(cv::Mat& in_frame, ProtoObject& cur_obj, cv::RotatedRect& obj_ellipse)
  {
    cv::Mat centroid_frame;
    in_frame.copyTo(centroid_frame);
    pcl::PointXYZ centroid_point(cur_obj.centroid[0], cur_obj.centroid[1],
                                 cur_obj.centroid[2]);
    const cv::Point img_c_idx = pcl_segmenter_->projectPointIntoImage(
        centroid_point, cur_obj.cloud.header.frame_id, camera_frame_);
    // double ellipse_angle_rad = subPIAngle(DEG2RAD(obj_ellipse.angle));
    double theta = getThetaFromEllipse(obj_ellipse);
    if(swap_orientation_)
    {
      if(theta > 0.0)
        theta += - M_PI;
      else
        theta += M_PI;
    }
    const float x_min_rad = (std::cos(theta+0.5*M_PI)* obj_ellipse.size.width*0.5);
    const float y_min_rad = (std::sin(theta+0.5*M_PI)* obj_ellipse.size.width*0.5);
    pcl::PointXYZ table_min_point(centroid_point.x+x_min_rad, centroid_point.y+y_min_rad,
                                  centroid_point.z);
    const float x_maj_rad = (std::cos(theta)*obj_ellipse.size.height*0.5);
    const float y_maj_rad = (std::sin(theta)*obj_ellipse.size.height*0.5);
    pcl::PointXYZ table_maj_point(centroid_point.x+x_maj_rad, centroid_point.y+y_maj_rad,
                                  centroid_point.z);
    const cv::Point2f img_min_idx = pcl_segmenter_->projectPointIntoImage(
        table_min_point, cur_obj.cloud.header.frame_id, camera_frame_);
    const cv::Point2f img_maj_idx = pcl_segmenter_->projectPointIntoImage(
        table_maj_point, cur_obj.cloud.header.frame_id, camera_frame_);
    cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,255),2);
    cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,255,0),2);
    cv::Size img_size;
    img_size.width = std::sqrt(std::pow(img_maj_idx.x-img_c_idx.x,2) +
                               std::pow(img_maj_idx.y-img_c_idx.y,2))*2.0;
    img_size.height = std::sqrt(std::pow(img_min_idx.x-img_c_idx.x,2) +
                                std::pow(img_min_idx.y-img_c_idx.y,2))*2.0;
    float img_angle = RAD2DEG(std::atan2(img_maj_idx.y-img_c_idx.y,
                                         img_maj_idx.x-img_c_idx.x));
    cv::RotatedRect img_ellipse(img_c_idx, img_size, img_angle);
    cv::ellipse(centroid_frame, img_ellipse, cv::Scalar(255,0,255), 1);
    if (use_displays_)
    {
      cv::imshow("Ellipse Axes", centroid_frame);
    }
    if (write_to_disk_)
    {
      std::stringstream out_name;
      out_name << base_output_path_ << "ellipse_axes_" << frame_set_count_ << "_"
               << record_count_ << ".png";
      cv::imwrite(out_name.str(), centroid_frame);
    }

  }

  shared_ptr<PointCloudSegmentation> pcl_segmenter_;
  int num_downsamples_;
  bool initialized_;
  int frame_count_;
  int upscale_;
  double previous_time_;
  ProtoObject previous_obj_;
  PushTrackerState previous_state_;
  cv::RotatedRect previous_obj_ellipse_;
  bool use_displays_;
  bool write_to_disk_;
  std::string base_output_path_;
  int record_count_;
  bool swap_orientation_;
  bool paused_;
  int frame_set_count_;
  std::string camera_frame_;
  bool use_cv_ellipse_fit_;
};

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
      learn_callback_count_(0), goal_out_count_(0), goal_heading_count_(0), frame_callback_count_(0),
      just_spun_(false), major_axis_spin_pos_scale_(0.75), spin_to_heading_(false)
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

    n_private_.param("push_tracker_dist_thresh", tracker_dist_thresh_, 0.05);
    n_private_.param("push_tracker_angle_thresh", tracker_angle_thresh_, 0.01);
    n_private_.param("major_axis_spin_pos_scale", major_axis_spin_pos_scale_, 0.75);
    bool use_cv_ellipse;
    n_private_.param("use_cv_ellipse", use_cv_ellipse, false);
    // Initialize classes requiring parameters
    obj_tracker_ = shared_ptr<ObjectTracker25D>(
        new ObjectTracker25D(pcl_segmenter_, num_downsamples_, use_displays_, write_to_disk_,
                             base_output_path_, camera_frame_, use_cv_ellipse));

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

    // Swap kinect color channel order
    // cv::cvtColor(color_frame, color_frame, CV_RGB2BGR);

    // Transform point cloud into the correct frame and convert to PCL struct
    XYZPointCloud cloud;
    pcl::fromROSMsg(*cloud_msg, cloud);
    tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
                          cloud.header.stamp, ros::Duration(0.5));
    pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);

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

    XYZPointCloud cloud_self_filtered;
    cloud_self_filtered.header = cloud.header;
    cloud_self_filtered.width = cloud.size();
    cloud_self_filtered.height = 1;
    cloud_self_filtered.is_dense = false;
    cloud_self_filtered.resize(cloud_self_filtered.width);
    for (unsigned int x = 0, i = 0; x < cloud.width; ++x)
    {
      for (unsigned int y = 0; y < cloud.height; ++y, ++i)
      {
        if (self_mask.at<uchar>(y,x) != 0)
        {
          cloud_self_filtered.at(i) = cloud.at(x,y);
        }
      }
    }

    // Downsample everything first
    cv::Mat color_frame_down = downSample(color_frame, num_downsamples_);
    cv::Mat depth_frame_down = downSample(depth_frame, num_downsamples_);
    cv::Mat self_mask_down = downSample(self_mask, num_downsamples_);

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
      PushTrackerState tracker_state = obj_tracker_->updateTracks(
          cur_color_frame_, cur_self_mask_, cur_self_filtered_cloud_);
      tracker_state.proxy_name = proxy_name_;
      tracker_state.controller_name = controller_name_;
      tracker_state.action_primitive = action_primitive_;

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
      // displayGoalHeading(cur_color_frame_, start_point, tracker_state.x.theta,
      //                    tracker_goal_pose_.theta);

      // make sure that the action hasn't been canceled
      if (as_.isActive())
      {
        as_.publishFeedback(tracker_state);
        // Check for goal conditions
        float x_error = tracker_goal_pose_.x - tracker_state.x.x;
        float y_error = tracker_goal_pose_.y - tracker_state.x.y;
        float theta_error = subPIAngle(tracker_goal_pose_.theta - tracker_state.x.theta);

        float x_dist = fabs(x_error);
        float y_dist = fabs(y_error);
        float theta_dist = fabs(theta_error);

        // TODO: Better abstract the goal and validity (abort) checks here
        // TODO: Based on controller_name / proxy_name call different goal functions
        if (spin_to_heading_ && theta_dist < tracker_angle_thresh_)
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
        else if (!spin_to_heading_ &&
                 x_dist < tracker_dist_thresh_ && y_dist < tracker_dist_thresh_)
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
        // Check hand between object and goal
        else if (!spin_to_heading_ && objectNotBetweenGoalAndGripper(tracker_state.x))
        {
          ROS_WARN_STREAM("Object is not between gripper and goal. Aborting");
          PushTrackerResult res;
          res.aborted = true;
          as_.setAborted(res);
          obj_tracker_->pause();
        }
        else
        {
          if (frame_callback_count_ % 15)
          {
            ROS_DEBUG_STREAM("Error in (x,y, theta): (" << x_error << ", " << y_error <<  ", " <<
                             theta_error << ")\n");
          }
        }
      }
    }
    else if (obj_tracker_->isPaused())
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
      displayPushVector(cur_color_frame_, start_point, end_point);
      // displayGoalHeading(cur_color_frame_, start_point, tracker_state.x.theta,
      //                    tracker_goal_pose_.theta);
    }

    // Display junk
#ifdef DISPLAY_INPUT_COLOR
    if (use_displays_)
    {
      cv::imshow("color", cur_color_frame_);
      cv::imshow("self_mask", cur_self_mask_);
    }
    // Way too much disk writing!
    if (write_input_to_disk_ && recording_input_)
    {
      std::stringstream out_name;
      out_name << base_output_path_ << "input" << record_count_ << ".png";
      cv::imwrite(out_name.str(), cur_color_frame_);
      std::stringstream self_out_name;
      self_out_name << base_output_path_ << "self" << record_count_ << ".png";
      cv::imwrite(self_out_name.str(), cur_self_mask_);
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
        ROS_INFO_STREAM("Initializing");
        record_count_ = 0;
        learn_callback_count_ = 0;
        // Initialize stuff if necessary (i.e. angle to push from)
        res.no_push = true;
        recording_input_ = false;
      }
      else if (req.analyze_previous)
      {
        ROS_INFO_STREAM("Analyzing previous");
        res = getAnalysisVector(req.goal_pose);
        res.no_push = true;
        recording_input_ = false;
      }
      else if (req.controller_name == "spin_to_heading")
      {
        ROS_INFO_STREAM("Getting spin push start pose");
        res = getSpinPushStartPose(req);
        recording_input_ = !res.no_objects;
        res.no_push = false;
      }
      else
      {
        ROS_INFO_STREAM("Getting push start pose");
        res = getPushStartPose(req);
        recording_input_ = !res.no_objects;
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
    PushTrackerState cur_state;
    if (just_spun_)
    {
      just_spun_ = false;
      cur_state = obj_tracker_->getMostRecentState();
    }
    else
    {
      ROS_INFO_STREAM("Starting tracker");
      cur_state = startTracking();
    }
    ROS_INFO_STREAM("Cur state: (" << cur_state.x.x << ", " << cur_state.x.y << ", " <<
                    cur_state.x.theta << ")");

    ProtoObject cur_obj = obj_tracker_->getMostRecentObject();
    tracker_goal_pose_ = req.goal_pose;
    if (!start_tracking_on_push_call_)
    {
      obj_tracker_->pause();
    }

    LearnPush::Response res;
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
    // Get straight line from current location to goal pose as start
    p.push_angle = atan2(req.goal_pose.y - res.centroid.y, req.goal_pose.x - res.centroid.x);

    // Get vector through centroid and determine start point and distance
    Eigen::Vector3f push_unit_vec(std::cos(p.push_angle), std::sin(p.push_angle), 0.0f);
    std::vector<pcl::PointXYZ> end_points = pcl_segmenter_->lineCloudIntersectionEndPoints(
        cur_obj.cloud, push_unit_vec, cur_obj.centroid);
    p.start_point.x = end_points[0].x;
    p.start_point.y = end_points[0].y;
    p.start_point.z = end_points[0].z;

    // Get push distance
    p.push_dist = hypot(res.centroid.x - req.goal_pose.x, res.centroid.y - req.goal_pose.y);
    // Visualize push vector
    displayPushVector(cur_color_frame_, p);
    PointStamped centroid;
    centroid.header.frame_id = cur_obj.cloud.header.frame_id;
    centroid.point = res.centroid;
    // displayGoalHeading(cur_color_frame_, centroid, cur_state.x.theta, tracker_goal_pose_.theta);

    learn_callback_count_++;
    ROS_INFO_STREAM("Chosen push start point: (" << p.start_point.x << ", "
                    << p.start_point.y << ", " << p.start_point.z << ")");
    ROS_INFO_STREAM("Push dist: " << p.push_dist);
    ROS_INFO_STREAM("Push angle: " << p.push_angle);
    start_centroid_ = cur_obj.centroid;
    res.push = p;
    return res;
  }

  LearnPush::Response getSpinPushStartPose(LearnPush::Request& req)
  {
    PushTrackerState cur_state = startTracking();
    ProtoObject cur_obj = obj_tracker_->getMostRecentObject();
    ROS_INFO_STREAM("Cur state: (" << cur_state.x.x << ", " << cur_state.x.y << ", " <<
                    cur_state.x.theta << ")");

    cv::RotatedRect cur_ellipse = obj_tracker_->getMostRecentEllipse();
    tracker_goal_pose_ = req.goal_pose;
    if (!start_tracking_on_push_call_)
    {
      obj_tracker_->pause();
    }
    LearnPush::Response res;
    if (cur_state.no_detection)
    {
      ROS_WARN_STREAM("No objects found");
      res.centroid.x = 0.0;
      res.centroid.y = 0.0;
      res.centroid.z = 0.0;
      res.no_objects = true;
      return res;
    }
    res.no_objects = false;
    res.centroid.x = cur_obj.centroid[0];
    res.centroid.y = cur_obj.centroid[1];
    res.centroid.z = cur_obj.centroid[2];

    // Use estimated ellipse to determine object extent and pushing locations
    Eigen::Vector3f major_axis(std::cos(cur_state.x.theta),
                               std::sin(cur_state.x.theta), 0.0f);
    Eigen::Vector3f minor_axis(std::cos(cur_state.x.theta+0.5*M_PI),
                               std::sin(cur_state.x.theta+0.5*M_PI), 0.0f);
    std::vector<pcl::PointXYZ> major_pts;
    major_pts = pcl_segmenter_->lineCloudIntersectionEndPoints(cur_obj.cloud,
                                                               major_axis,
                                                               cur_obj.centroid);
    std::vector<pcl::PointXYZ> minor_pts;
    minor_pts = pcl_segmenter_->lineCloudIntersectionEndPoints(cur_obj.cloud,
                                                               minor_axis,
                                                               cur_obj.centroid);
    Eigen::Vector3f centroid(cur_obj.centroid[0], cur_obj.centroid[1], cur_obj.centroid[2]);
    Eigen::Vector3f major_pos((major_pts[0].x-centroid[0]), (major_pts[0].y-centroid[1]), 0.0);
    Eigen::Vector3f minor_pos((minor_pts[0].x-centroid[0]), (minor_pts[0].y-centroid[1]), 0.0);
    ROS_DEBUG_STREAM("major_pts: " << major_pts[0] << ", " << major_pts[1]);
    ROS_DEBUG_STREAM("minor_pts: " << minor_pts[0] << ", " << minor_pts[1]);
    Eigen::Vector3f major_neg = -major_pos;
    Eigen::Vector3f minor_neg = -minor_pos;
    Eigen::Vector3f push_pt0 = centroid + major_axis_spin_pos_scale_*major_pos + minor_pos;
    Eigen::Vector3f push_pt1 = centroid + major_axis_spin_pos_scale_*major_pos + minor_neg;
    Eigen::Vector3f push_pt2 = centroid + major_axis_spin_pos_scale_*major_neg + minor_neg;
    Eigen::Vector3f push_pt3 = centroid + major_axis_spin_pos_scale_*major_neg + minor_pos;

    std::vector<Eigen::Vector3f> push_pts;
    std::vector<float> sx;
    push_pts.push_back(push_pt0);
    sx.push_back(1.0);
    push_pts.push_back(push_pt1);
    sx.push_back(-1.0);
    push_pts.push_back(push_pt2);
    sx.push_back(-1.0);
    push_pts.push_back(push_pt3);
    sx.push_back(1.0);

    // TODO: Display the pushing point locations
    cv::Mat disp_img;
    cur_color_frame_.copyTo(disp_img);

    // Set basic push information
    PushVector p;
    p.header.frame_id = workspace_frame_;

    // Choose point and rotation direction
    unsigned int chosen_idx = 0;
    double theta_error = subPIAngle(req.goal_pose.theta - cur_state.x.theta);
    ROS_INFO_STREAM("Theta error is: " << theta_error);
    // TODO: Make this choice to find the point on the outside
    if (theta_error > 0.0)
    {
      // Positive push is corner 1 or 3
      if (push_pts[1][1] > push_pts[3][1])
      {
        if (centroid[1] > 0)
        {
          chosen_idx = 1;
        }
        else
        {
          chosen_idx = 3;
        }
      }
      else
      {
        if (centroid[1] < 0)
        {
          chosen_idx = 1;
        }
        else
        {
          chosen_idx = 3;
        }
      }
    }
    else
    {
      // Negative push is corner 0 or 2
      if (push_pts[0][1] > push_pts[2][1])
      {
        if (centroid[1] > 0)
        {
          chosen_idx = 0;
        }
        else
        {
          chosen_idx = 2;
        }
      }
      else
      {
        if (centroid[1] < 0)
        {
          chosen_idx = 0;
        }
        else
        {
          chosen_idx = 2;
        }
      }
    }
    ROS_DEBUG_STREAM("Chosen idx is : " << chosen_idx);
    p.start_point.x = push_pts[chosen_idx][0];
    p.start_point.y = push_pts[chosen_idx][1];
    p.start_point.z = centroid[2];
    p.push_angle = cur_state.x.theta+sx[chosen_idx]*0.5*M_PI;
    // NOTE: This is useless here, whatever
    p.push_dist = hypot(res.centroid.x - req.goal_pose.x, res.centroid.y - req.goal_pose.y);
    res.push = p;
    res.theta = cur_state.x.theta;
    just_spun_ = true;

    if (use_displays_|| write_to_disk_)
    {
      for (unsigned int i = 0; i < push_pts.size(); ++i)
      {
        ROS_DEBUG_STREAM("Point " << i << " is: " << push_pts[i]);
        const cv::Point2f img_idx = pcl_segmenter_->projectPointIntoImage(
            push_pts[i], cur_obj.cloud.header.frame_id, camera_frame_);
        cv::Scalar draw_color;
        if (i == chosen_idx)
        {
          draw_color = cv::Scalar(0,0,255);
        }
        else
        {
          draw_color = cv::Scalar(0,255,0);
        }
        cv::circle(disp_img, img_idx, 4, draw_color);
      }
      if (use_displays_)
      {
        cv::imshow("push points", disp_img);
      }
      if (write_to_disk_)
      {
        // Write to disk to create video output
        std::stringstream push_out_name;
        push_out_name << base_output_path_ << "push_points" << frame_set_count_ << ".png";
        cv::imwrite(push_out_name.str(), disp_img);
      }
    }
    PointStamped centroid_pt;
    centroid_pt.header.frame_id = cur_obj.cloud.header.frame_id;
    centroid_pt.point = res.centroid;
    displayGoalHeading(cur_color_frame_, centroid_pt, cur_state.x.theta, tracker_goal_pose_.theta);
    return res;
  }

  LearnPush::Response getAnalysisVector(Pose2D goal_pose)
  {
    PushTrackerState tracker_state = obj_tracker_->getMostRecentState();
    ProtoObject cur_obj = obj_tracker_->getMostRecentObject();

    LearnPush::Response res;
    if (tracker_state.no_detection)
    {
      ROS_WARN_STREAM("No objects found");
      res.centroid.x = 0.0;
      res.centroid.y = 0.0;
      res.centroid.z = 0.0;
      res.no_objects = true;
      return res;
    }
    res.no_objects = false;
    res.centroid.x = cur_obj.centroid[0];
    res.centroid.y = cur_obj.centroid[1];
    res.centroid.z = cur_obj.centroid[2];
    res.theta = tracker_state.x.theta;

    return res;
  }

  PushTrackerState startTracking()
  {
    frame_set_count_++;
    goal_out_count_ = 0;
    goal_heading_count_ = 0;
    frame_callback_count_ = 0;
    return obj_tracker_->initTracks(cur_color_frame_, cur_self_mask_, cur_self_filtered_cloud_);
  }

  void lArmStateCartCB(const pr2_manipulation_controllers::JTTaskControllerState l_arm_state)
  {
    l_arm_pose_ = l_arm_state.x;
  }
  void rArmStateCartCB(const pr2_manipulation_controllers::JTTaskControllerState r_arm_state)
  {
    r_arm_pose_ = r_arm_state.x;
  }
  void lArmStateVelCB(const pr2_manipulation_controllers::JinvTeleopControllerState l_arm_state)
  {
    l_arm_pose_ = l_arm_state.x;
  }
  void rArmStateVelCB(const pr2_manipulation_controllers::JinvTeleopControllerState r_arm_state)
  {
    r_arm_pose_ = r_arm_state.x;
  }

  void pushTrackerGoalCB()
  {
    ROS_DEBUG_STREAM("pushTrackerGoalCB(): starting tracking");
    if (obj_tracker_->isInitialized())
    {
      obj_tracker_->unpause();
    }
    else
    {
      startTracking();
    }
    ROS_INFO_STREAM("Accepting goal");
    shared_ptr<const PushTrackerGoal> tracker_goal = as_.acceptNewGoal();
    // TODO: Transform into workspace frame...
    tracker_goal_pose_ = tracker_goal->desired_pose;
    pushing_arm_ = tracker_goal->which_arm;
    controller_name_ = tracker_goal->controller_name;
    proxy_name_ = tracker_goal->proxy_name;
    action_primitive_ = tracker_goal->action_primitive;
    spin_to_heading_ = tracker_goal->controller_name == "spin_to_heading";
    ROS_INFO_STREAM("Accepted goal of " << tracker_goal_pose_);
  }

  void pushTrackerPreemptCB()
  {
    obj_tracker_->pause();
    ROS_INFO_STREAM("Preempted push tracker");
    // set the action state to preempted
    as_.setPreempted();
  }

  bool objectNotBetweenGoalAndGripper(Pose2D& obj_state)
  {
    if (pushing_arm_ == "l")
    {
      return pointIsBetweenOthers(l_arm_pose_.pose.position, obj_state, tracker_goal_pose_);
    }
    else
    {
      return pointIsBetweenOthers(r_arm_pose_.pose.position, obj_state, tracker_goal_pose_);
    }
  }

  bool pointIsBetweenOthers(geometry_msgs::Point pt, Pose2D& x1, Pose2D& x2)
  {
    // Project the vector pt->x2 onto the vector x1->x2
    const float a_x = x2.x - pt.x;
    const float a_y = x2.y - pt.y;
    const float b_x = x2.x - x1.x;
    const float b_y = x2.y - x1.y;
    const float a_dot_b = a_x*b_x + a_y*b_y;
    const float b_dot_b = b_x*b_x + b_y*b_y;

    // If the (squared) distance of the projection is less than the vector from x1->x2 then it is between them
    const float a_onto_b = a_dot_b/b_dot_b;
    const float d_1_x = a_onto_b*b_x;
    const float d_1_y = a_onto_b*b_y;
    const float d_1 = d_1_x*d_1_x + d_1_y*d_1_y;
    const float d_2 = b_x*b_x + b_y*b_y;

    // if (frame_callback_count_ % 15)
    // {
    //   ROS_INFO_STREAM("ee_pose (" << pt.x << ", " << pt.y << ")");
    //   ROS_INFO_STREAM("goal_pose (" << x2.x << ", " << x2.y << ")");
    //   ROS_INFO_STREAM("obj_pose (" << x1.x << ", " << x1.y << ")");
    //   ROS_INFO_STREAM("a: (" << a_x << ", " << a_y << ")");
    //   ROS_INFO_STREAM("b: (" << b_x << ", " << b_y << ")");
    //   ROS_INFO_STREAM("a_onto_b: (" << d_1_x << ", " << d_1_y << ")");
    //   ROS_INFO_STREAM("Distance d_1 is: " << d_1);
    //   ROS_INFO_STREAM("Distance d_2 is: " << d_2 << "\n");
    // }
    return d_1 < d_2;
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

  void displayPushVector(cv::Mat& img, PushVector& push)
  {
    PointStamped start_point;
    start_point.point = push.start_point;
    start_point.header.frame_id = workspace_frame_;
    PointStamped end_point;
    end_point.point.x = start_point.point.x+std::cos(push.push_angle)*push.push_dist;
    end_point.point.y = start_point.point.y+std::sin(push.push_angle)*push.push_dist;
    end_point.point.z = start_point.point.z;
    end_point.header.frame_id = workspace_frame_;
    displayPushVector(img, start_point, end_point);
  }

  void displayPushVector(cv::Mat& img, PointStamped& start_point, PointStamped& end_point)
  {
    cv::Mat disp_img;
    img.copyTo(disp_img);

    cv::Point img_start_point = pcl_segmenter_->projectPointIntoImage(
        start_point);
    cv::Point img_end_point = pcl_segmenter_->projectPointIntoImage(
        end_point);
    cv::line(disp_img, img_start_point, img_end_point, cv::Scalar(0,255,0));
    cv::circle(disp_img, img_end_point, 4, cv::Scalar(0,255,0));

    if (use_displays_)
    {
      cv::imshow("goal_vector", disp_img);
    }
    if (write_to_disk_)
    {
      // Write to disk to create video output
      std::stringstream push_out_name;
      push_out_name << base_output_path_ << "goal_vector_" << frame_set_count_ << "_"
                    << goal_out_count_++ << ".png";
      cv::imwrite(push_out_name.str(), disp_img);
    }
  }

  void displayGoalHeading(cv::Mat& img, PointStamped& centroid, double theta, double goal_theta)
  {
    cv::Mat disp_img;
    img.copyTo(disp_img);

    cv::Point img_c_idx = pcl_segmenter_->projectPointIntoImage(centroid);
    cv::RotatedRect obj_ellipse = obj_tracker_->getMostRecentEllipse();
    const float x_maj_rad = (std::cos(theta)*obj_ellipse.size.height*0.5);
    const float y_maj_rad = (std::sin(theta)*obj_ellipse.size.height*0.5);
    pcl::PointXYZ table_heading_point(centroid.point.x+x_maj_rad, centroid.point.y+y_maj_rad,
                                      centroid.point.z);
    const cv::Point2f img_maj_idx = pcl_segmenter_->projectPointIntoImage(
        table_heading_point, centroid.header.frame_id, camera_frame_);
    const float goal_x_rad = (std::cos(goal_theta)*obj_ellipse.size.height*0.5);
    const float goal_y_rad = (std::sin(goal_theta)*obj_ellipse.size.height*0.5);
    pcl::PointXYZ goal_heading_point(centroid.point.x+goal_x_rad, centroid.point.y+goal_y_rad,
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
  Eigen::Vector4f start_centroid_;
  shared_ptr<ObjectTracker25D> obj_tracker_;
  Pose2D tracker_goal_pose_;
  std::string pushing_arm_;
  std::string proxy_name_;
  std::string controller_name_;
  std::string action_primitive_;
  double tracker_dist_thresh_;
  double tracker_angle_thresh_;
  bool just_spun_;
  double major_axis_spin_pos_scale_;
  bool spin_to_heading_;
  int frame_set_count_;
  PoseStamped l_arm_pose_;
  PoseStamped r_arm_pose_;
};

int main(int argc, char ** argv)
{
  int seed = time(NULL);
  ros::init(argc, argv, "tabletop_pushing_perception_node");
  ros::NodeHandle n;
  TabletopPushingPerceptionNode perception_node(n);
  perception_node.spin();
  return 0;
}
