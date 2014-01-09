// TabletopPushing
#include <tabletop_pushing/object_tracker_25d.h>
#include <tabletop_pushing/push_primitives.h>
#include <tabletop_pushing/shape_features.h>
#include <tabletop_pushing/extern/Timer.hpp>

// OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL
#include <pcl16/common/pca.h>

// cpl_visual_features
#include <cpl_visual_features/helpers.h>

// Debugging IFDEFS
// #define PROFILE_TRACKING_TIME 1
// #define PROFILE_FIND_TARGET_TIME 1
// #define PROFILE_COMPUTE_STATE_TIME 1
// #define VISUALIZE_FEATURE_POINT_ICP_PROXY 1
// #define USE_DISPLAY 1

// Functional IFDEFS
// #define USE_BOUNDARY_TRANSFORM_GUESS 1
// #define USE_FRAME_TO_FRAME_MATCHING 1
// #define USE_RATIO_TEST 1

using namespace tabletop_pushing;
using geometry_msgs::PoseStamped;
using boost::shared_ptr;
using cpl_visual_features::subPIAngle;

typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;
typedef tabletop_pushing::VisFeedbackPushTrackingGoal PushTrackerGoal;
typedef tabletop_pushing::VisFeedbackPushTrackingResult PushTrackerResult;
typedef tabletop_pushing::VisFeedbackPushTrackingAction PushTrackerAction;

typedef pcl16::PointCloud<pcl16::PointXYZ> XYZPointCloud;

ObjectTracker25D::ObjectTracker25D(shared_ptr<PointCloudSegmentation> segmenter,
                                   shared_ptr<ArmObjSegmentation> arm_segmenter, int num_downsamples,
                                   bool use_displays, bool write_to_disk, std::string base_output_path,
                                   std::string camera_frame, bool use_cv_ellipse, bool use_mps_segmentation,
                                   bool use_graphcut_arm_seg, double hull_alpha,
                                   int feature_point_close_size, int icp_max_iters, float icp_transform_eps,
                                   float icp_ransac_thresh, int icp_max_ransac_iters,
                                   float icp_max_fitness_eps, int brief_descriptor_byte_size,
                                   float feature_point_ratio_test_thresh) :
    pcl_segmenter_(segmenter), arm_segmenter_(arm_segmenter),
    num_downsamples_(num_downsamples), initialized_(false),
    frame_count_(0), use_displays_(use_displays), write_to_disk_(write_to_disk),
    base_output_path_(base_output_path), record_count_(0), swap_orientation_(false),
    paused_(false), frame_set_count_(0), camera_frame_(camera_frame),
    use_cv_ellipse_fit_(use_cv_ellipse), use_mps_segmentation_(use_mps_segmentation),
    have_obj_color_model_(false), have_table_color_model_(false), use_graphcut_arm_seg_(use_graphcut_arm_seg),
    hull_alpha_(hull_alpha),
#ifdef USE_ORB
    // feature_extractor_(orb_params),
#else
    feature_extractor_(brief_descriptor_byte_size),
#endif
#ifdef USE_RATIO_TEST
    matcher_(cv::NORM_HAMMING, false),
#else
    matcher_(cv::NORM_HAMMING, true),
#endif
    ratio_test_thresh_(feature_point_ratio_test_thresh)
{
  upscale_ = std::pow(2,num_downsamples_);
  cv::Mat tmp_morph(feature_point_close_size, feature_point_close_size, CV_8UC1, cv::Scalar(255));
  tmp_morph.copyTo(feature_point_morph_element_);
  ROS_INFO_STREAM("Changing ICP iters from " << feature_point_icp_.getMaximumIterations() << " to " <<
                  icp_max_iters);
  feature_point_icp_.setMaximumIterations(icp_max_iters);

  ROS_INFO_STREAM("Changing Transfrom epsilon from " << feature_point_icp_.getTransformationEpsilon() << " to " <<
                  icp_transform_eps);
  feature_point_icp_.setTransformationEpsilon(icp_transform_eps);

  ROS_INFO_STREAM("Changin RANSAC Outlier Rejection from " << feature_point_icp_.getRANSACOutlierRejectionThreshold() <<
                  " to " << icp_ransac_thresh);
  feature_point_icp_.setRANSACOutlierRejectionThreshold(icp_ransac_thresh);

  ROS_INFO_STREAM("Changing RANSAC iters from " << feature_point_icp_.getRANSACIterations() << " to " <<
                  icp_max_ransac_iters);
  feature_point_icp_.setRANSACIterations(icp_max_ransac_iters);

  ROS_INFO_STREAM("Changing Euclidean fitness from " << feature_point_icp_.getEuclideanFitnessEpsilon() << " to " <<
                  icp_max_fitness_eps);
  feature_point_icp_.setEuclideanFitnessEpsilon(icp_max_fitness_eps);
}

ProtoObject ObjectTracker25D::findTargetObjectGC(cv::Mat& in_frame, XYZPointCloud& cloud, cv::Mat& depth_frame,
                                                 cv::Mat self_mask, bool& no_objects, bool init)
{
  // Segment arm from background using graphcut
  ROS_INFO_STREAM("Getting table cloud and mask.");
  XYZPointCloud table_cloud, non_table_cloud;
  cv::Mat table_mask = getTableMask(cloud, table_cloud, self_mask.size(), non_table_cloud);

  if (!have_table_color_model_)
  {
    ROS_INFO_STREAM("Building table color model.");
    table_color_model_ = buildColorModel(table_cloud, in_frame, 3);
    have_table_color_model_ = true;
    if (have_obj_color_model_)
    {
      ROS_INFO_STREAM("Combining bg color models");
      arm_segmenter_->buildBGColorModel(table_color_model_, obj_color_model_);
    }
  }
  ROS_INFO_STREAM("Segmenting arm.");
  cv::Mat segs = arm_segmenter_->segment(in_frame, depth_frame, self_mask, table_mask, init);
  pcl16::PointIndices obj_pts;
  ROS_INFO_STREAM("Removing table and arm points.");
  // Remove arm and table points from cloud
  for(int i = 0; i < non_table_cloud.size(); ++i)
  {
    if (isnan(non_table_cloud.at(i).x) || isnan(non_table_cloud.at(i).y) || isnan(non_table_cloud.at(i).z))
    {
      continue;
    }
    // ROS_INFO_STREAM("Projecting point.");
    cv::Point img_pt = pcl_segmenter_->projectPointIntoImage(non_table_cloud.at(i),
                                                             non_table_cloud.header.frame_id, camera_frame_);
    // ROS_INFO_STREAM("(" << cloud.at(i).x << ", " << cloud.at(i).y << ", " << cloud.at(i).z << ") -> (" <<
    //                 img_pt.x << ", " << img_pt.y << ")");
    const bool is_arm = segs.at<uchar>(img_pt.y, img_pt.x) != 0;
    // ROS_INFO_STREAM("is_arm " << is_arm);
    const bool is_table = table_mask.at<uchar>(img_pt.y, img_pt.x) != 0;
    // ROS_INFO_STREAM("is_table " << is_table);
    if ( !is_arm && !is_table)
    {
      obj_pts.indices.push_back(i);
    }
  }
  if (obj_pts.indices.size() < 1)
  {
    ROS_WARN_STREAM("No objects found in findTargetObjectGC");
    ProtoObject empty;
    no_objects = true;
    return empty;
  }
  ROS_INFO_STREAM("Copying object points");
  XYZPointCloud objs_cloud;
  pcl16::copyPointCloud(non_table_cloud, obj_pts, objs_cloud);

  // Cluster objects from remaining point cloud
  ProtoObjects objs;
  XYZPointCloud objects_cloud_down;
  // TODO: Add switch to choose between downsampling object cloud and not
  ROS_INFO_STREAM("Downsampling object points");
  pcl_segmenter_->downsampleCloud(objs_cloud, objects_cloud_down);
  // Find independent regions
  if (objects_cloud_down.size() > 0)
  {
    ROS_INFO_STREAM("Clustering object points");
    pcl_segmenter_->clusterProtoObjects(objects_cloud_down, objs);
  }
  else
  {
    ROS_WARN_STREAM("No objects found in findTargetObjectGC");
    ProtoObject empty;
    no_objects = true;
    return empty;
  }
  ROS_INFO_STREAM("Matching object");
  no_objects = false;
  return matchToTargetObject(objs, in_frame, init);
}

ProtoObject ObjectTracker25D::findTargetObject(cv::Mat& in_frame, XYZPointCloud& cloud,
                                               bool& no_objects, bool init)
{
#ifdef PROFILE_FIND_TARGET_TIME
  long long find_target_start_time = Timer::nanoTime();
#endif
  ProtoObjects objs;
  pcl_segmenter_->findTabletopObjects(cloud, objs, use_mps_segmentation_);
#ifdef PROFILE_FIND_TARGET_TIME
  double find_tabletop_objects_elapsed_time = (((double)(Timer::nanoTime() - find_target_start_time)) /
                                           Timer::NANOSECONDS_PER_SECOND);
  long long choose_object_start_time = Timer::nanoTime();
#endif
  if (objs.size() == 0)
  {
    ROS_WARN_STREAM("No objects found");
    ProtoObject empty;
    no_objects = true;
#ifdef PROFILE_FIND_TARGET_TIME
    double find_target_elapsed_time = (((double)(Timer::nanoTime() - find_target_start_time)) /
                                    Timer::NANOSECONDS_PER_SECOND);
    ROS_INFO_STREAM("\t find_target_elapsed_time " << find_target_elapsed_time);
    ROS_INFO_STREAM("\t\t find_tabletop_objects_elapsed_time " << find_tabletop_objects_elapsed_time);
#endif
    return empty;
  }
  no_objects = false;
#ifdef PROFILE_FIND_TARGET_TIME
  // Have to do an extra copy here...
  ProtoObject chosen = matchToTargetObject(objs, in_frame, init);

  double choose_object_elapsed_time = (((double)(Timer::nanoTime() - choose_object_start_time)) /
                                       Timer::NANOSECONDS_PER_SECOND);
  double find_target_elapsed_time = (((double)(Timer::nanoTime() - find_target_start_time)) /
                                     Timer::NANOSECONDS_PER_SECOND);
  ROS_INFO_STREAM("find_target_elapsed_time " << find_target_elapsed_time);
  ROS_INFO_STREAM("\t find tabletop_objects_elapsed_time " << find_tabletop_objects_elapsed_time <<
                  "\t " << (100.0*find_tabletop_objects_elapsed_time/find_target_elapsed_time) << "\%");
  ROS_INFO_STREAM("\t choose_object_elapsed_time " << choose_object_elapsed_time <<
                  "\t\t " << (100.0*choose_object_elapsed_time/find_target_elapsed_time) << "\%\n");
  return chosen;
#else // PROFILE_FIND_TARGET_TIME
  return matchToTargetObject(objs, in_frame, init);
#endif // PROFILE_FIND_TARGET_TIME
}

ProtoObject ObjectTracker25D::matchToTargetObject(ProtoObjects& objs, cv::Mat& in_frame, bool init)
{
  int chosen_idx = 0;
  if (objs.size() == 1)
  {
    // ROS_INFO_STREAM("Picking the only object");
  }
  else if (init || frame_count_ == 0)
  {
    // ROS_INFO_STREAM("Picking the biggest object at initialization");
    // NOTE: Assume we care about the biggest currently
    unsigned int max_size = 0;
    for (unsigned int i = 0; i < objs.size(); ++i)
    {
      if (objs[i].cloud.size() > max_size)
      {
        max_size = objs[i].cloud.size();
        chosen_idx = i;
      }
    }
  }
  else // Find closest object to last time
  {
    // ROS_INFO_STREAM("Finding the closest object to previous");
    double min_dist = 1000.0;
    for (unsigned int i = 0; i < objs.size(); ++i)
    {
      double centroid_dist = pcl_segmenter_->sqrDist(objs[i].centroid, previous_obj_.centroid);
      if (centroid_dist  < min_dist)
      {
        min_dist = centroid_dist;
        chosen_idx = i;
      }
      // TODO: Match color GMM model
    }
    // ROS_INFO_STREAM("Chose object " << chosen_idx << " at distance " << min_dist);
  }
  if (init && use_graphcut_arm_seg_)
  {
    // Extract color GMM model
    // ROS_INFO_STREAM("Building object color model.");
    obj_color_model_ = buildColorModel(objs[chosen_idx].cloud, in_frame, 5);
    have_obj_color_model_ = true;
    if (have_table_color_model_)
    {
      // ROS_INFO_STREAM("Combining bg color models");
      arm_segmenter_->buildBGColorModel(table_color_model_, obj_color_model_);
    }
  }

#ifdef USE_DISPLAY
  if (use_displays_)
  {
    cv::Mat disp_img = pcl_segmenter_->projectProtoObjectsIntoImage(
        objs, in_frame.size(), objs[0].cloud.header.frame_id);
    pcl_segmenter_->displayObjectImage(disp_img, "Objects", true);
    // ROS_INFO_STREAM("Updating display");
  }
#endif // USE_DISPLAY
  // ROS_INFO_STREAM("Returning matched object\n");

  return objs[chosen_idx];
}

void ObjectTracker25D::updateHeading(PushTrackerState& state, bool init_state)
{
  if(swap_orientation_)
  {
    state.x.theta += (state.x.theta > 0.0) ? - M_PI : M_PI;
  }
  // If not initializing, check if we need to swap our addition because of heading change
  if (!init_state && (state.x.theta > 0) != (previous_state_.x.theta > 0))
  {
    // Test if swapping makes a shorter distance than changing
    float augmented_theta = state.x.theta + (state.x.theta > 0.0) ? - M_PI : M_PI;
    float augmented_diff = fabs(subPIAngle(augmented_theta - previous_state_.x.theta));
    float current_diff = fabs(subPIAngle(state.x.theta - previous_state_.x.theta));
    if (augmented_diff < current_diff)
    {
      swap_orientation_ = !swap_orientation_;
      // We either need to swap or need to undo the swap
      state.x.theta = augmented_theta;
    }
  }
}

void ObjectTracker25D::computeState(ProtoObject& cur_obj, XYZPointCloud& cloud, std::string proxy_name,
                                    cv::Mat& in_frame, PushTrackerState& state, bool init_state)
{
#ifdef PROFILE_COMPUTE_STATE_TIME
  long long compute_state_start_time = Timer::nanoTime();
  double boundary_samples_elapsed_time = 0.0;
  double fit_ellipse_elapsed_time = 0.0;
  double copy_state_elapsed_time = 0.0;
  double icp_elapsed_time = 0.0;
  double update_stuff_elapsed_time = 0.0;
#endif // PROFILE_COMPUTE_STATE_TIME

  // TODO: Have each proxy create an image, and send that image to the trackerDisplay
  // function to deal with saving and display.
  cv::RotatedRect obj_ellipse;
  if (proxy_name == ELLIPSE_PROXY || proxy_name == CENTROID_PROXY ||
      proxy_name == SPHERE_PROXY || proxy_name == CYLINDER_PROXY)
  {
    updateStateEllipse(cur_obj, obj_ellipse, state, init_state);
  }
  else if (proxy_name == HULL_ELLIPSE_PROXY || proxy_name == HULL_ICP_PROXY ||
           proxy_name == HULL_SHAPE_CONTEXT_PROXY)
  {
#ifdef PROFILE_COMPUTE_STATE_TIME
      long long boundary_samples_start_time = Timer::nanoTime();
#endif // PROFILE_COMPUTE_STATE_TIME

    // Get 2D object boundary
    XYZPointCloud hull_cloud = tabletop_pushing::getObjectBoundarySamples(cur_obj, hull_alpha_);
#ifdef PROFILE_COMPUTE_STATE_TIME
    boundary_samples_elapsed_time = (((double)(Timer::nanoTime() - boundary_samples_start_time)) /
                                     Timer::NANOSECONDS_PER_SECOND);
#endif // PROFILE_COMPUTE_STATE_TIME

    // Get ellipse orientation from the 2D boundary
    if (frame_count_ < 1 || proxy_name == HULL_ELLIPSE_PROXY)
    {
#ifdef PROFILE_COMPUTE_STATE_TIME
      long long fit_ellipse_start_time = Timer::nanoTime();
#endif // PROFILE_COMPUTE_STATE_TIME

      fitHullEllipse(hull_cloud, obj_ellipse);
      state.x.theta = getThetaFromEllipse(obj_ellipse);
      // Get (x,y) centroid of boundary
      state.x.x = obj_ellipse.center.x;
      state.x.y = obj_ellipse.center.y;
      // Use vertical z centroid from object
      state.z = cur_obj.centroid[2];
      updateHeading(state, init_state);

#ifdef USE_BOUNDARY_TRANSFORM_GUESS
      previous_centroid_state_ = state;
#endif // USE_BOUNDARY_TRANSFORM_GUESS

#ifdef PROFILE_COMPUTE_STATE_TIME
      fit_ellipse_elapsed_time = (((double)(Timer::nanoTime() - fit_ellipse_start_time)) /
                                  Timer::NANOSECONDS_PER_SECOND);
#endif // PROFILE_COMPUTE_STATE_TIME

    }
    else
    {
#ifdef PROFILE_COMPUTE_STATE_TIME
      long long icp_start_time = Timer::nanoTime();
#endif // PROFILE_COMPUTE_STATE_TIME
      cpl_visual_features::Path matches;
      XYZPointCloud aligned;
      Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

      if (proxy_name == HULL_SHAPE_CONTEXT_PROXY)
      {
        double match_cost;
        matches = compareBoundaryShapes(previous_hull_cloud_, hull_cloud, match_cost);
        ROS_INFO_STREAM("Found minimum cost match of: " << match_cost);
        estimateTransformFromMatches(previous_hull_cloud_, hull_cloud, matches, transform);
      }
      else // (proxy_name == HULL_ICP_PROXY)
      {
        Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();

#ifdef USE_BOUNDARY_TRANSFORM_GUESS
        cv::RotatedRect centroid_obj_ellipse;
        tabletop_pushing::VisFeedbackPushTrackingFeedback centroid_state;
        fitHullEllipse(hull_cloud, centroid_obj_ellipse);
        centroid_state.x.theta = getThetaFromEllipse(centroid_obj_ellipse);
        // Get (x,y) centroid of boundary
        centroid_state.x.x = centroid_obj_ellipse.center.x;
        centroid_state.x.y = centroid_obj_ellipse.center.y;
        // Use vertical z centroid from object
        centroid_state.z = cur_obj.centroid[2];

        estimateTransformFromStateChange(centroid_state, previous_centroid_state_, guess);
        previous_centroid_state_ = centroid_state;
#endif // USE_BOUNDARY_TRANSFORM_GUESS

        double match_score = pcl_segmenter_->ICPBoundarySamples(previous_hull_cloud_, hull_cloud,
                                                                guess, transform, aligned);

        // ROS_INFO_STREAM("Found ICP match with score: " << match_score);

#ifdef PROFILE_COMPUTE_STATE_TIME
        icp_elapsed_time = (((double)(Timer::nanoTime() - icp_start_time)) / Timer::NANOSECONDS_PER_SECOND);
#endif // PROFILE_COMPUTE_STATE_TIME

      }
#ifdef PROFILE_COMPUTE_STATE_TIME
      long long copy_state_start_time = Timer::nanoTime();
#endif // PROFILE_COMPUTE_STATE_TIME


      // Transform previous state using the estimate and update current state
      // ROS_INFO_STREAM("Found transform of: \n" << transform);
      // // HACK: Remove z change components to keep in plane
      // // Remove any z rotation
      // transform(0,2) = 0.0;
      // transform(1,2) = 0.0;
      // transform(2,0) = 0.0;
      // transform(2,1) = 0.0;
      // // Remove any delta z
      // transform(2,3) = 0.0;
      // ROS_INFO_STREAM("Cleaned transform of: \n" << transform);
      Eigen::Vector4f x_t_0(previous_state_.x.x, previous_state_.x.y, previous_state_.z, 1.0);
      Eigen::Vector4f x_t_1 = transform*x_t_0;
      state.x.x = x_t_1(0);
      state.x.y = x_t_1(1);
      state.z = x_t_1(2);
      Eigen::Matrix3f rot = transform.block<3,3>(0,0);
      const Eigen::Vector3f x_axis(cos(previous_state_.x.theta), sin(previous_state_.x.theta), 0.0);
      const Eigen::Vector3f x_axis_t = rot*x_axis;
      state.x.theta = atan2(x_axis_t(1), x_axis_t(0));

#ifdef PROFILE_COMPUTE_STATE_TIME
      copy_state_elapsed_time = (((double)(Timer::nanoTime() - copy_state_start_time)) /
                                 Timer::NANOSECONDS_PER_SECOND);
#endif // PROFILE_COMPUTE_STATE_TIME

#ifdef USE_DISPLAY
      // Visualize the matches
      if (use_displays_ || write_to_disk_)
      {
        cv::Mat match_img;
        if (proxy_name == HULL_SHAPE_CONTEXT_PROXY)
        {
          match_img = visualizeObjectBoundaryMatches(previous_hull_cloud_, hull_cloud, state, matches);
        }
        else
        {
          for (int i = 0; i < previous_hull_cloud_.size(); ++i)
          {
            matches.push_back(i);
          }
          match_img = visualizeObjectBoundaryMatches(previous_hull_cloud_, aligned, state, matches);
        }
        if (use_displays_)
        {
          cv::imshow("Boundary matches", match_img);
        }
        if (write_to_disk_ && !isPaused())
        {
          std::stringstream out_name;
          out_name << base_output_path_ << "boundary_matches_" << frame_set_count_ << "_"
                   << record_count_ << ".png";
          cv::imwrite(out_name.str(), match_img);
        }
      }
#endif // USE_DISPLAY
    }

#ifdef PROFILE_COMPUTE_STATE_TIME
    long long update_stuff_start_time = Timer::nanoTime();
#endif // PROFILE_COMPUTE_STATE_TIME

    // Update stuff
    previous_hull_cloud_ = hull_cloud;

#ifdef PROFILE_COMPUTE_STATE_TIME
    update_stuff_elapsed_time = (((double)(Timer::nanoTime() - update_stuff_start_time)) /
                                 Timer::NANOSECONDS_PER_SECOND);
#endif // PROFILE_COMPUTE_STATE_TIME

  }
  else if (proxy_name == BOUNDING_BOX_XY_PROXY)
  {
    findFootprintBox(cur_obj, obj_ellipse);
    double min_z = 10000;
    double max_z = -10000;
    for (int i = 0; i < cur_obj.cloud.size(); ++i)
    {
      if (cur_obj.cloud.at(i).z < min_z)
      {
        min_z = cur_obj.cloud.at(i).z;
      }
      if (cur_obj.cloud.at(i).z > max_z)
      {
        max_z = cur_obj.cloud.at(i).z;
      }
    }
    state.x.x = obj_ellipse.center.x;
    state.x.y = obj_ellipse.center.y;
    state.z = (min_z+max_z)*0.5;
    state.x.theta = getThetaFromEllipse(obj_ellipse);
    updateHeading(state, init_state);
    // ROS_INFO_STREAM("box (x,y,z): " << state.x.x << ", " << state.x.y << ", " <<
    //                 state.z << ")");
    // ROS_INFO_STREAM("centroid (x,y,z): " << cur_obj.centroid[0] << ", " << cur_obj.centroid[1]
    //                 << ", " << cur_obj.centroid[2] << ")");
  }
  else if (proxy_name == FEATURE_POINT_ICP_PROXY)
  {
    if (init_state)
    {
      // Extract object model
      updateStateEllipse(cur_obj, obj_ellipse, state, init_state);
      extractFeaturePointModel(in_frame, cloud, cur_obj, obj_feature_point_model_);
    }
    else
    {
      // Get feature points for current model
      ObjectFeaturePointModel obj_model_detected;
      extractFeaturePointModel(in_frame, cloud, cur_obj, obj_model_detected);

      // Setup mask of bad locs to not match to them
      cv::Mat no_match_mask(obj_model_detected.keypoints.size(), obj_feature_point_model_.keypoints.size(),
                            CV_8UC1, cv::Scalar(1) );
      for (int i = 0; i < obj_feature_point_model_.bad_locs.size(); ++i)
      {
        int c = obj_feature_point_model_.bad_locs[i];
        for (int r = 0; r < no_match_mask.rows; ++r)
        {
          no_match_mask.at<uchar>(r,c) = 0;
        }
      }

      for (int i = 0; i < obj_model_detected.bad_locs.size(); ++i)
      {
        int r = obj_model_detected.bad_locs[i];
        for (int c = 0; c < no_match_mask.cols; ++c)
        {
          no_match_mask.at<uchar>(r,c) = 0;
        }
      }

      ROS_INFO_STREAM("object source model has " << obj_feature_point_model_.bad_locs.size() << " bad locs from " <<
                      obj_feature_point_model_.keypoints.size() << " total keypoints");
      ROS_INFO_STREAM("object target model has " << obj_model_detected.bad_locs.size() << " bad locs from " <<
                      obj_model_detected.keypoints.size() << " total keypoints");

      // Match feature points to model
      std::vector<int> source_indices;
      std::vector<int> target_indices;
#ifdef USE_RATIO_TEST
      std::vector<std::vector<cv::DMatch> > matches;
      matcher_.knnMatch(obj_model_detected.descriptors, obj_feature_point_model_.descriptors, matches, 2,
                        no_match_mask, true);
      for (int i = 0; i < matches.size(); ++i)
      {
        if (matches[i][0].distance < ratio_test_thresh_*matches[i][1].distance)
        {
          source_indices.push_back(matches[i][0].trainIdx);
          target_indices.push_back(matches[i][0].queryIdx);
        }
      }
      ROS_INFO_STREAM("Found " << source_indices.size() << " good matches with ratio test thresh of " <<
                      ratio_test_thresh_);
#else  // USE_RATIO_TEST
      std::vector<cv::DMatch> matches;
      matcher_.match(obj_model_detected.descriptors, obj_feature_point_model_.descriptors, matches);
      for (int i = 0; i < matches.size(); ++i)
      {
        source_indices.push_back(matches[i].trainIdx);
        target_indices.push_back(matches[i].queryIdx);
      }
      ROS_INFO_STREAM("Found " << source_indices.size() << " good matches");
#endif
#ifdef VISUALIZE_FEATURE_POINT_ICP_PROXY
      ROS_INFO_STREAM("Displaying stuff");
      cv::Mat match_img = in_frame.clone();
      for (int i = 0; i < matches.size(); ++i)
      {
#ifdef USE_RATIO_TEST
        // Must pass ratio test
        if (matches[i][0].distance < ratio_test_thresh_*matches[i][1].distance)
        {
          cv::circle(match_img, obj_feature_point_model_.keypoints[matches[i][0].trainIdx].pt, 3,
                     cv::Scalar(18, 18, 178));
          cv::circle(match_img, obj_model_detected.keypoints[matches[i][0].queryIdx].pt, 3,
                     cv::Scalar(51, 178, 0));
          cv::line(match_img, obj_feature_point_model_.keypoints[matches[i][0].trainIdx].pt,
                   obj_model_detected.keypoints[matches[i][0].queryIdx].pt, cv::Scalar(204, 133, 20), 1);
        }
#else
        cv::circle(match_img, obj_feature_point_model_.keypoints[matches[i].trainIdx].pt, 3,
                   cv::Scalar(18, 18, 178));
        cv::circle(match_img, obj_model_detected.keypoints[matches[i].queryIdx].pt, 3,
                   cv::Scalar(51, 178, 0));
        cv::line(match_img, obj_feature_point_model_.keypoints[matches[i].trainIdx].pt,
                 obj_model_detected.keypoints[matches[i].queryIdx].pt, cv::Scalar(204, 133, 20), 1);
#endif
      }
      cv::imshow("Matches", match_img);
#endif // VISUALIZE_FEATURE_POINT_ICP_PROXY

      // Estimate transform
      Eigen::Matrix4f transform;
      bool converged = estimateFeaturePointTransform(obj_feature_point_model_, source_indices, obj_model_detected,
                                                     target_indices, transform);
      if (!converged)
      {
        ROS_WARN_STREAM("ICP did not converge. Estimating state from ellipse");
        updateStateEllipse(cur_obj, obj_ellipse, state, init_state);

        ROS_INFO_STREAM("Copying transform to previous transform");
        Eigen::Matrix4f state_transform;
#ifdef USE_FRAME_TO_FRAME_MATCHING
        estimateTransformFromStateChange(state, previous_state_, state_transform);
#else
        estimateTransformFromStateChange(state, initial_state_, state_transform);
#endif
        previous_transform_ = state_transform;
      }
      else
      {
#ifdef USE_FRAME_TO_FRAME_MATCHING
        // Update state estimates
        // Transform initial state to current state using the estimate transform
        ROS_INFO_STREAM("Transforming previous state");
        Eigen::Vector4f x_t_0(previous_state_.x.x, previous_state_.x.y, previous_state_.z, 1.0);
        Eigen::Vector4f x_t_1 = transform*x_t_0;
        state.x.x = x_t_1(0);
        state.x.y = x_t_1(1);
        state.z = x_t_1(2);
        Eigen::Matrix3f rot = transform.block<3,3>(0,0);
        const Eigen::Vector3f x_axis(cos(previous_state_.x.theta), sin(previous_state_.x.theta), 0.0);
        const Eigen::Vector3f x_axis_t = rot*x_axis;
        state.x.theta = atan2(x_axis_t(1), x_axis_t(0));

#else // USE_FRAME_TO_FRAME_MATCHING
        // Update state estimates
        // Transform initial state to current state using the estimate transform
        ROS_INFO_STREAM("Transforming init state");
        Eigen::Vector4f x_t_0(initial_state_.x.x, initial_state_.x.y, initial_state_.z, 1.0);
        Eigen::Vector4f x_t_1 = transform*x_t_0;
        state.x.x = x_t_1(0);
        state.x.y = x_t_1(1);
        state.z = x_t_1(2);
        Eigen::Matrix3f rot = transform.block<3,3>(0,0);
        const Eigen::Vector3f x_axis(cos(initial_state_.x.theta), sin(initial_state_.x.theta), 0.0);
        const Eigen::Vector3f x_axis_t = rot*x_axis;
        state.x.theta = atan2(x_axis_t(1), x_axis_t(0));
#endif // USE_FRAME_TO_FRAME_MATCHING
        ROS_INFO_STREAM("Copying transform to previous transform");
        previous_transform_ = transform;
      }
#ifdef USE_FRAME_TO_FRAME_MATCHING
      // Update model to previous frame
      ROS_INFO_STREAM("Updating previous model");
      obj_model_detected.descriptors.copyTo(obj_feature_point_model_.descriptors);
      obj_feature_point_model_.locations.clear();
      obj_feature_point_model_.keypoints.clear();
      obj_feature_point_model_.bad_locs.clear();
      for (int i = 0; i < obj_model_detected.locations.size(); ++i)
      {
        obj_feature_point_model_.locations.push_back(obj_model_detected.locations[i]);
      }
      for (int i = 0; i < obj_model_detected.keypoints.size(); ++i)
      {
        obj_feature_point_model_.keypoints.push_back(obj_model_detected.keypoints[i]);
      }
      for (int i = 0; i < obj_model_detected.bad_locs.size(); ++i)
      {
        obj_feature_point_model_.bad_locs.push_back(obj_model_detected.bad_locs[i]);
      }
#endif// USE_FRAME_TO_FRAME_MATCHING
    }
  }
  else
  {
    ROS_WARN_STREAM("Unknown perceptual proxy: " << proxy_name << "!");
  }

  if (proxy_name == SPHERE_PROXY)
  {
    XYZPointCloud sphere_cloud;
    pcl16::ModelCoefficients sphere;
    pcl_segmenter_->fitSphereRANSAC(cur_obj,sphere_cloud, sphere);
    cv::Mat lbl_img(in_frame.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat disp_img(in_frame.size(), CV_8UC3, cv::Scalar(0,0,0));
    if (sphere_cloud.size() < 1)
    {
      ROS_INFO_STREAM("Sphere has 0 points");
    }
    else
    {
      pcl_segmenter_->projectPointCloudIntoImage(sphere_cloud, lbl_img);
      lbl_img*=255;
      pcl16::PointXYZ centroid_point(sphere.values[0], sphere.values[1], sphere.values[2]);
      cv::cvtColor(lbl_img, disp_img, CV_GRAY2BGR);
      const cv::Point img_c_idx = pcl_segmenter_->projectPointIntoImage(
          centroid_point, cur_obj.cloud.header.frame_id, camera_frame_);
      cv::circle(disp_img, img_c_idx, 4, cv::Scalar(0,255,0));
      cv::imshow("sphere",disp_img);
    }
    state.x.x = sphere.values[0];
    state.x.y = sphere.values[1];
    state.z = sphere.values[2];
    // state.x.theta = 0.0;
    // TODO: Draw ellipse of the projected circle parallel to the table
    // std::stringstream out_name;
    // out_name << base_output_path_ << "sphere_" << frame_set_count_ << "_"
    //          << record_count_ << ".png";
    // cv::imwrite(out_name.str(), disp_img);

    // ROS_INFO_STREAM("sphere (x,y,z,r): " << state.x.x << ", " << state.x.y << ", " << state.z
    //                 << ", " << sphere.values[3] << ")");
    // ROS_INFO_STREAM("centroid (x,y,z): " << cur_obj.centroid[0] << ", " << cur_obj.centroid[1]
    //                 << ", " << cur_obj.centroid[2] << ")");
  }
  if (proxy_name == CYLINDER_PROXY)
  {
    XYZPointCloud cylinder_cloud;
    pcl16::ModelCoefficients cylinder;
    pcl_segmenter_->fitCylinderRANSAC(cur_obj, cylinder_cloud, cylinder);
    cv::Mat lbl_img(in_frame.size(), CV_8UC1, cv::Scalar(0));
    pcl_segmenter_->projectPointCloudIntoImage(cylinder_cloud, lbl_img);
    lbl_img*=255;
    cv::imshow("cylinder",lbl_img);
    ROS_INFO_STREAM("cylinder: " << cylinder);
    // NOTE: Z may be bade, depending on how it is computed
    // TODO: Update this to the cylinder centroid
    state.x.x = cylinder.values[0];
    state.x.y = cylinder.values[1];
    state.z = cur_obj.centroid[2];//# cylinder.values[2];
    // state.x.theta = 0.0;
    // ROS_INFO_STREAM("cylinder (x,y,z): " << state.x.x << ", " << state.x.y << ", " <<
    //                 cylinder.values[2] << ")");
    // ROS_INFO_STREAM("centroid (x,y,z): " << cur_obj.centroid[0] << ", " << cur_obj.centroid[1]
    //                 << ", " << cur_obj.centroid[2] << ")");
  }

#ifdef USE_DISPLAY
  if (use_displays_ || write_to_disk_)
  {
    if (proxy_name == ELLIPSE_PROXY)
    {
      trackerDisplay(in_frame, cur_obj, obj_ellipse);
    }
    else if(proxy_name == BOUNDING_BOX_XY_PROXY)
    {
      trackerBoxDisplay(in_frame, cur_obj, obj_ellipse);
    }
    else
    {
      trackerDisplay(in_frame, state, cur_obj);
    }
  }
#endif // USE_DISPLAY

#ifdef PROFILE_COMPUTE_STATE_TIME
  double compute_state_elapsed_time = (((double)(Timer::nanoTime() - compute_state_start_time)) /
                                          Timer::NANOSECONDS_PER_SECOND);
  ROS_INFO_STREAM("compute_state_elapsed_time " << compute_state_elapsed_time);
  ROS_INFO_STREAM("\t fit_ellipse_elapsed_time " << fit_ellipse_elapsed_time << "\t\t\t" <<
                  (100.0*fit_ellipse_elapsed_time/compute_state_elapsed_time) << "\%");
  ROS_INFO_STREAM("\t boundary_samples_elapsed_time " << boundary_samples_elapsed_time << "\t" <<
                  (100.0*boundary_samples_elapsed_time/compute_state_elapsed_time) << "\%");
  ROS_INFO_STREAM("\t copy_state_elapsed_time " << copy_state_elapsed_time << "\t\t" <<
                  (100.0*copy_state_elapsed_time/compute_state_elapsed_time) << "\%");
  ROS_INFO_STREAM("\t icp_elapsed_time " << icp_elapsed_time << "\t\t\t" <<
                  (100.0*icp_elapsed_time/compute_state_elapsed_time) << "\%");
  ROS_INFO_STREAM("\t update_stuff_elapsed_time " << update_stuff_elapsed_time << "\t\t" <<
                  (100.0*update_stuff_elapsed_time/compute_state_elapsed_time) << "\%\n");
#endif // PROFILE_COMPUTE_STATE_TIME

}

void ObjectTracker25D::fitObjectEllipse(ProtoObject& obj, cv::RotatedRect& ellipse)
{
  if (use_cv_ellipse_fit_)
  {
    findFootprintEllipse(obj, ellipse);
  }
  else
  {
    fit2DMassEllipse(obj, ellipse);
  }
}

void ObjectTracker25D::fitHullEllipse(XYZPointCloud& hull_cloud, cv::RotatedRect& obj_ellipse)
{
  Eigen::Vector3f eigen_values;
  Eigen::Matrix3f eigen_vectors;
  Eigen::Vector4f centroid;

  // HACK: Copied/adapted from PCA in PCL because PCL was seg faulting after an update on the robot
  // Compute mean
  centroid = Eigen::Vector4f::Zero();
  // ROS_INFO_STREAM("Getting centroid");
  pcl16::compute3DCentroid(hull_cloud, centroid);
  // Compute demeanished cloud
  Eigen::MatrixXf cloud_demean;
  // ROS_INFO_STREAM("Demenaing point cloud");
  pcl16::demeanPointCloud(hull_cloud, centroid, cloud_demean);

  // Compute the product cloud_demean * cloud_demean^T
  // ROS_INFO_STREAM("Getting alpha");
  Eigen::Matrix3f alpha = static_cast<Eigen::Matrix3f> (cloud_demean.topRows<3> () * cloud_demean.topRows<3> ().transpose ());

  // Compute eigen vectors and values
  // ROS_INFO_STREAM("Getting eigenvectors");
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> evd (alpha);
  // Organize eigenvectors and eigenvalues in ascendent order
  for (int i = 0; i < 3; ++i)
  {
    eigen_values[i] = evd.eigenvalues()[2-i];
    eigen_vectors.col(i) = evd.eigenvectors().col(2-i);
  }

  // try{
  //   pca.setInputCloud(cloud_no_z.makeShared());
  //   ROS_INFO_STREAM("Getting mean");
  //   centroid = pca.getMean();
  //   ROS_INFO_STREAM("Getting eiven values");
  //   eigen_values = pca.getEigenValues();
  //   ROS_INFO_STREAM("Getting eiven vectors");
  //   eigen_vectors = pca.getEigenVectors();
  // } catch(pcl16::InitFailedException ife)
  // {
  //   ROS_WARN_STREAM("Failed to compute PCA");
  //   ROS_WARN_STREAM("ife: " << ife.what());
  // }

  obj_ellipse.center.x = centroid[0];
  obj_ellipse.center.y = centroid[1];
  obj_ellipse.angle = RAD2DEG(atan2(eigen_vectors(1,0), eigen_vectors(0,0))-0.5*M_PI);
  // NOTE: major axis is defined by height
  obj_ellipse.size.height = std::max(eigen_values(0)*0.1, 0.07);
  obj_ellipse.size.width = std::max(eigen_values(1)*0.1, 0.03);

}

void ObjectTracker25D::findFootprintEllipse(ProtoObject& obj, cv::RotatedRect& obj_ellipse)
{
  // Get 2D footprint of object and fit an ellipse to it
  std::vector<cv::Point2f> obj_pts;
  for (unsigned int i = 0; i < obj.cloud.size(); ++i)
  {
    obj_pts.push_back(cv::Point2f(obj.cloud[i].x, obj.cloud[i].y));
  }
  ROS_DEBUG_STREAM("Number of points is: " << obj_pts.size());
  obj_ellipse = cv::fitEllipse(obj_pts);
}

void ObjectTracker25D::updateStateEllipse(ProtoObject& cur_obj, cv::RotatedRect& obj_ellipse,
                                          PushTrackerState& state, bool init_state)
{
  fitObjectEllipse(cur_obj, obj_ellipse);
  state.x.theta = getThetaFromEllipse(obj_ellipse);
  state.x.x = cur_obj.centroid[0];
  state.x.y = cur_obj.centroid[1];
  state.z = cur_obj.centroid[2];
  updateHeading(state, init_state);
}


void ObjectTracker25D::findFootprintBox(ProtoObject& obj, cv::RotatedRect& box)
{
  // Get 2D footprint of object and fit an ellipse to it
  std::vector<cv::Point2f> obj_pts;
  for (unsigned int i = 0; i < obj.cloud.size(); ++i)
  {
    obj_pts.push_back(cv::Point2f(obj.cloud[i].x, obj.cloud[i].y));
  }
  ROS_DEBUG_STREAM("Number of points is: " << obj_pts.size());
  box = cv::minAreaRect(obj_pts);
}

void ObjectTracker25D::fit2DMassEllipse(ProtoObject& obj, cv::RotatedRect& obj_ellipse)
{
  // pcl16::PCA<pcl16::PointXYZ> pca(true);
  XYZPointCloud cloud_no_z;
  cloud_no_z.header = obj.cloud.header;
  cloud_no_z.width = obj.cloud.points.size();
  cloud_no_z.height = 1;
  cloud_no_z.is_dense = false;
  cloud_no_z.resize(cloud_no_z.width*cloud_no_z.height);
  if (obj.cloud.size() < 3)
  {
    ROS_WARN_STREAM("Too few points to find ellipse");
    obj_ellipse.center.x = 0.0;
    obj_ellipse.center.y = 0.0;
    obj_ellipse.angle = 0;
    obj_ellipse.size.width = 0;
    obj_ellipse.size.height = 0;
  }
  for (unsigned int i = 0; i < obj.cloud.size(); ++i)
  {
    cloud_no_z[i] = obj.cloud[i];
    cloud_no_z[i].z = 0.0f;
  }
  Eigen::Vector3f eigen_values;
  Eigen::Matrix3f eigen_vectors;
  Eigen::Vector4f centroid;

  // HACK: Copied/adapted from PCA in PCL because PCL was seg faulting after an update on the robot
  // Compute mean
  centroid = Eigen::Vector4f::Zero();
  // ROS_INFO_STREAM("Getting centroid");
  pcl16::compute3DCentroid(cloud_no_z, centroid);
  // Compute demeanished cloud
  Eigen::MatrixXf cloud_demean;
  // ROS_INFO_STREAM("Demenaing point cloud");
  pcl16::demeanPointCloud(cloud_no_z, centroid, cloud_demean);

  // Compute the product cloud_demean * cloud_demean^T
  // ROS_INFO_STREAM("Getting alpha");
  Eigen::Matrix3f alpha = static_cast<Eigen::Matrix3f> (cloud_demean.topRows<3> () * cloud_demean.topRows<3> ().transpose ());

  // Compute eigen vectors and values
  // ROS_INFO_STREAM("Getting eigenvectors");
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> evd (alpha);
  // Organize eigenvectors and eigenvalues in ascendent order
  for (int i = 0; i < 3; ++i)
  {
    eigen_values[i] = evd.eigenvalues()[2-i];
    eigen_vectors.col(i) = evd.eigenvectors().col(2-i);
  }

  // try{
  //   pca.setInputCloud(cloud_no_z.makeShared());
  //   ROS_INFO_STREAM("Getting mean");
  //   centroid = pca.getMean();
  //   ROS_INFO_STREAM("Getting eiven values");
  //   eigen_values = pca.getEigenValues();
  //   ROS_INFO_STREAM("Getting eiven vectors");
  //   eigen_vectors = pca.getEigenVectors();
  // } catch(pcl16::InitFailedException ife)
  // {
  //   ROS_WARN_STREAM("Failed to compute PCA");
  //   ROS_WARN_STREAM("ife: " << ife.what());
  // }

  obj_ellipse.center.x = centroid[0];
  obj_ellipse.center.y = centroid[1];
  obj_ellipse.angle = RAD2DEG(atan2(eigen_vectors(1,0), eigen_vectors(0,0))-0.5*M_PI);
  // NOTE: major axis is defined by height
  obj_ellipse.size.height = std::max(eigen_values(0)*0.1, 0.07);
  obj_ellipse.size.width = std::max(eigen_values(1)*0.1, 0.03);
}

void ObjectTracker25D::extractFeaturePointModel(cv::Mat& frame, XYZPointCloud& cloud, ProtoObject& obj,
                                                ObjectFeaturePointModel& model)
{
  // Get grayscale image for keypoint extraction
  cv::Mat frame_bw;
  cv::cvtColor(frame, frame_bw, CV_BGR2GRAY);

  // Get object mask in frame
  cv::Mat obj_mask(frame.size(), CV_8UC1, cv::Scalar(0));
  pcl_segmenter_->projectPointCloudIntoImage(obj.cloud, obj_mask);
  obj_mask*=255;
  cv::dilate(obj_mask, obj_mask, feature_point_morph_element_);
  cv::erode(obj_mask, obj_mask, feature_point_morph_element_);

  // Get keypoint descriptors from mask
  cv::Mat descriptors;
  feature_detector_.detect(frame_bw, model.keypoints, obj_mask);
#ifdef USE_ORB
  feature_extractor_(frame_bw, obj_mask, model.keypoints, descriptors, true);
#else
  feature_extractor_.compute(frame_bw, model.keypoints, descriptors);
#endif

  descriptors.copyTo(model.descriptors);
  model.locations.width = model.keypoints.size();
  model.locations.height = 1;
  model.locations.is_dense = false;
  model.locations.resize(model.locations.width);
  model.locations.header.frame_id = cloud.header.frame_id;
  model.locations.header.stamp = cloud.header.stamp;

  // Get associated 3D locations for the keypoints
  for (int i = 0; i < model.keypoints.size(); ++i)
  {
    model.locations.at(i) = cloud.at((int)(upscale_*model.keypoints[i].pt.x),
                                     (int)(upscale_*model.keypoints[i].pt.y));
    // Make sure this is a valid 3D point
    if (isnan(model.locations.at(i).x) || isnan(model.locations.at(i).y) || isnan(model.locations.at(i).z))
    {
      model.bad_locs.push_back(i);
    }
  }
  if (frame_count_ < 1)
  {
    ROS_INFO_STREAM("Found " << model.locations.size() - model.bad_locs.size() << " valid initial 3D keypoints.");
  }
  else
  {
    ROS_INFO_STREAM("Found " << model.locations.size() - model.bad_locs.size() << " valid 3D keypoints.");
  }

#ifdef VISUALIZE_FEATURE_POINT_ICP_PROXY
  cv::Mat key_disp_frame = frame.clone();
  cv::Mat model_points = frame.clone();
  cv::Scalar kuler_green(51, 178, 0);
  cv::Scalar kuler_red(18, 18, 178);
  for (int i = 0; i < model.locations.size(); ++i)
  {
    const cv::Point img_idx = pcl_segmenter_->projectPointIntoImage(model.locations.at(i),
                                                                    model.locations.header.frame_id,
                                                                    camera_frame_);
    if (frame_count_ < 1)
    {
      cv::circle(model_points, img_idx, 3, kuler_red);
    }
    else
    {
      cv::circle(model_points, img_idx, 3, kuler_green);
    }
  }

  // TODO: Show keypoints on 3D segmentaiton data...

  if (frame_count_ < 1)
  {
    // cv::imshow("init object mask", obj_mask);
    cv::imshow("init model points", model_points);
  }
  else
  {
    // cv::imshow("object mask", obj_mask);
    cv::imshow("model points", model_points);
  }
#endif // VISUALIZE_FEATURE_POINT_ICP_PROXY
}

bool ObjectTracker25D::estimateFeaturePointTransform(ObjectFeaturePointModel& source_model,
                                                     std::vector<int> source_indices,
                                                     ObjectFeaturePointModel& target_model,
                                                     std::vector<int> target_indices,
                                                     Eigen::Matrix4f& transform)
{
  boost::shared_ptr<XYZPointCloud> source_cloud(new XYZPointCloud);
  source_cloud->width = source_indices.size();
  source_cloud->height = 1;
  source_cloud->is_dense = false;
  source_cloud->resize(source_cloud->width);
  source_cloud->header = source_model.locations.header;

  boost::shared_ptr<XYZPointCloud> target_cloud(new XYZPointCloud);
  target_cloud->width = target_indices.size();
  target_cloud->height = 1;
  target_cloud->is_dense = false;
  target_cloud->resize(target_cloud->width);
  target_cloud->header = target_model.locations.header;

  for (int i = 0; i < source_indices.size(); ++i)
  {
    pcl16::PointXYZ source_pt = source_model.locations.at(source_indices[i]);
    pcl16::PointXYZ target_pt = target_model.locations.at(target_indices[i]);
    bool add_pts = true;
    if (isnan(source_pt.x) || isnan(source_pt.y) || isnan(source_pt.z))
    {
      ROS_WARN_STREAM("nan point in source_cloud at " << i << " -> " << source_pt);
      add_pts = false;
    }
    if (isnan(target_pt.x) || isnan(target_pt.y) || isnan(target_pt.z))
    {
      ROS_WARN_STREAM("nan point in target_cloud at " << i << " -> " << target_pt);
      add_pts = false;
    }
    if (add_pts)
    {
      target_cloud->at(i) = target_pt;
      source_cloud->at(i) = source_pt;
    }
  }

  feature_point_icp_.setInputCloud(source_cloud);
  feature_point_icp_.setInputTarget(target_cloud);

  ROS_INFO_STREAM("Aligning with initial guess: \n" << previous_transform_);
  XYZPointCloud aligned;
#ifdef USE_FRAME_TO_FRAME_MATCHING
  feature_point_icp_.align(aligned, Eigen::Matrix4f::Identity());
#else
  feature_point_icp_.align(aligned, previous_transform_);
#endif
  transform = feature_point_icp_.getFinalTransformation();
  ROS_INFO_STREAM("Found transform of: \n" << transform);
  double score = feature_point_icp_.getFitnessScore();
  ROS_INFO_STREAM("Fitness of: " << score << "\n");
  return feature_point_icp_.hasConverged();
}

void ObjectTracker25D::estimateTransformFromStateChange(PushTrackerState& state,
                                                        PushTrackerState& previous_state,
                                                        Eigen::Matrix4f& transform)
{
  double delta_x = state.x.x - previous_state_.x.x;
  double delta_y = state.x.y - previous_state_.x.y;

  // Make sure not a +/-pi swap of the heading between frames...
  if ( (state.x.theta > 0) != (previous_state_.x.theta > 0) )
  {
    // Test if swapping makes a shorter distance than changing
    float augmented_theta = state.x.theta + (state.x.theta > 0.0) ? - M_PI : M_PI;
    float augmented_diff = fabs(subPIAngle(augmented_theta - previous_state_.x.theta));
    float current_diff = fabs(subPIAngle(state.x.theta - previous_state_.x.theta));
    if (augmented_diff < current_diff)
    {
      state.x.theta = augmented_theta;
    }
  }

  double delta_theta = subPIAngle(state.x.theta - previous_state_.x.theta);

  transform(0,0) = cos(delta_theta);
  transform(0,1) = -sin(delta_theta);
  transform(1,0) = -transform(0,1);
  transform(1,1) = transform(0,0);
  transform(2,2) = 1.0;
  transform(3,3) = 1.0;
  transform(0,3) = delta_x;
  transform(1,3) = delta_y;
  ROS_INFO_STREAM("Delta X: " << delta_x);
  ROS_INFO_STREAM("Delta Y: " << delta_y);
  ROS_INFO_STREAM("Delta Theta: " << delta_theta);
  ROS_INFO_STREAM("Initial transform of: \n" << transform);
}



void ObjectTracker25D::initTracks(cv::Mat& in_frame, cv::Mat& self_mask, XYZPointCloud& cloud,
                                  std::string proxy_name,
                                  PushTrackerState& state, bool start_swap)
{
  paused_ = false;
  initialized_ = false;
  obj_saved_ = false;
  swap_orientation_ = start_swap;
  bool no_objects = false;
  frame_count_ = 0;
  record_count_ = 0;
  frame_set_count_++;
  previous_obj_ = findTargetObject(in_frame, cloud, no_objects, true);
  previous_transform_ = Eigen::Matrix4f::Identity();
  initialized_ = true;
  if (no_objects)
  {
    state.header.seq = 0;
    state.header.stamp = cloud.header.stamp;
    state.header.frame_id = cloud.header.frame_id;
    state.no_detection = true;
    return;
  }
  else
  {
    computeState(previous_obj_, cloud, proxy_name, in_frame, state, true);
    state.header.seq = 0;
    state.header.stamp = cloud.header.stamp;
    state.header.frame_id = cloud.header.frame_id;
    state.no_detection = false;
  }
  state.init_x.x = state.x.x;
  state.init_x.y = state.x.y;
  state.init_x.theta = state.x.theta;
  state.x_dot.x = 0.0;
  state.x_dot.y = 0.0;
  state.x_dot.theta = 0.0;

  ROS_DEBUG_STREAM("X: (" << state.x.x << ", " << state.x.y << ", " << state.z << ", " << state.x.theta << ")");
  ROS_DEBUG_STREAM("X_dot: (" << state.x_dot.x << ", " << state.x_dot.y << ", " << state.x_dot.theta << ")\n");

  previous_time_ = state.header.stamp.toSec();
  previous_state_ = state;
  initial_state_ = state;
  in_frame.copyTo(init_frame_);
  obj_saved_ = true;
  frame_count_ = 1;
  record_count_ = 1;
}

double ObjectTracker25D::getThetaFromEllipse(cv::RotatedRect& obj_ellipse)
{
  return subPIAngle(DEG2RAD(obj_ellipse.angle)+0.5*M_PI);
}

void ObjectTracker25D::updateTracks(cv::Mat& in_frame, cv::Mat& self_mask,
                                    XYZPointCloud& cloud, std::string proxy_name, PushTrackerState& state)
{
#ifdef PROFILE_TRACKING_TIME
  long long update_start_time = Timer::nanoTime();
#endif
  if (!initialized_)
  {
#ifdef PROFILE_TRACKING_TIME
    long long init_start_time = Timer::nanoTime();
#endif
    initTracks(in_frame, self_mask, cloud, proxy_name, state);
#ifdef PROFILE_TRACKING_TIME
    double init_elapsed_time = (((double)(Timer::nanoTime() - init_start_time)) / Timer::NANOSECONDS_PER_SECOND);
    ROS_INFO_STREAM("init_elapsed_time " << init_elapsed_time);
#endif
    return;
  }
  bool no_objects = false;
#ifdef PROFILE_TRACKING_TIME
  long long find_target_start_time = Timer::nanoTime();
#endif
  ProtoObject cur_obj;
  cur_obj = findTargetObject(in_frame, cloud, no_objects);
#ifdef PROFILE_TRACKING_TIME
  double find_target_elapsed_time = (((double)(Timer::nanoTime() - find_target_start_time)) / Timer::NANOSECONDS_PER_SECOND);
  long long update_model_start_time = Timer::nanoTime();
  double compute_state_elapsed_time = 0.0;
  long long copy_state_start_time = 0;
#endif

  // Update model
  if (no_objects)
  {
    state.header.seq = frame_count_;
    state.header.stamp = cloud.header.stamp;
    state.header.frame_id = cloud.header.frame_id;
    state.no_detection = true;
    state.x = previous_state_.x;
    state.x_dot = previous_state_.x_dot;
    state.z = previous_state_.z;
    ROS_WARN_STREAM("Using previous state, but updating time!");
    if (use_displays_ || write_to_disk_)
    {
      if (obj_saved_)
      {
        trackerDisplay(in_frame, previous_state_, previous_obj_);
      }
    }
  }
  else
  {
    obj_saved_ = true;
#ifdef PROFILE_TRACKING_TIME
    double compute_state_start_time = Timer::nanoTime();
#endif // PROFILE_TRACKING_TIME

    computeState(cur_obj, cloud, proxy_name, in_frame, state);

#ifdef PROFILE_TRACKING_TIME
    copy_state_start_time = Timer::nanoTime();
    compute_state_elapsed_time = (((double)(copy_state_start_time - compute_state_start_time)) / Timer::NANOSECONDS_PER_SECOND);
#endif // PROFILE_TRACKING_TIME

    state.header.seq = frame_count_;
    state.header.stamp = cloud.header.stamp;
    state.header.frame_id = cloud.header.frame_id;
    // Estimate dynamics and do some bookkeeping
    double delta_x = state.x.x - previous_state_.x.x;
    double delta_y = state.x.y - previous_state_.x.y;
    double delta_z = state.z - previous_state_.z;
    double delta_theta = subPIAngle(state.x.theta - previous_state_.x.theta);
    double delta_t = state.header.stamp.toSec() - previous_time_;
    state.x_dot.x = delta_x/delta_t;
    state.x_dot.y = delta_y/delta_t;
    state.x_dot.theta = delta_theta/delta_t;
    // ROS_INFO_STREAM("Delta X: (" << delta_x << ", " << delta_y << ", " << delta_z << ", " << delta_theta << ")");
    // ROS_INFO_STREAM("Delta t: " << delta_t);
    // ROS_INFO_STREAM("X: (" << state.x.x << ", " << state.x.y << ", " << state.z << ", " <<
    //                 state.x.theta << ")");
    // ROS_INFO_STREAM("X_dot: (" << state.x_dot.x << ", " << state.x_dot.y
    //                 << ", " << state.x_dot.theta << ")");
    previous_obj_ = cur_obj;
  }
  // We update the header and take care of other bookkeeping before returning
  state.init_x.x = initial_state_.x.x;
  state.init_x.y = initial_state_.x.y;
  state.init_x.theta = initial_state_.x.theta;

  previous_time_ = state.header.stamp.toSec();
  previous_state_ = state;
  frame_count_++;
  record_count_++;

#ifdef PROFILE_TRACKING_TIME
  long long update_model_end_time = Timer::nanoTime();
  double update_model_elapsed_time = (((double)(update_model_end_time - update_model_start_time)) /
                                   Timer::NANOSECONDS_PER_SECOND);
  double update_elapsed_time = (((double)(update_model_end_time - update_start_time)) /
                                Timer::NANOSECONDS_PER_SECOND);
  double copy_state_elapsed_time = (((double)(update_model_end_time - copy_state_start_time)) /
                                    Timer::NANOSECONDS_PER_SECOND);
  ROS_INFO_STREAM("update_tracks_elapsed_time " << update_elapsed_time);
  ROS_INFO_STREAM("\t find_target_elapsed_time " << find_target_elapsed_time <<
                  "\t\t " << (100.0*find_target_elapsed_time/update_elapsed_time) << "\%");
  ROS_INFO_STREAM("\t update_model_elapsed_time " << update_model_elapsed_time <<
                  "\t\t " << (100.0*update_model_elapsed_time/update_elapsed_time) << "\%");
  ROS_INFO_STREAM("\t\t compute_state_elapsed_time " << compute_state_elapsed_time <<
                  "\t " << (100.0*compute_state_elapsed_time/update_elapsed_time) << "\%");
  ROS_INFO_STREAM("\t\t copy_state_elapsed_time " << copy_state_elapsed_time <<
                  "\t " << (100.0*copy_state_elapsed_time/update_elapsed_time) << "\%\n");
#endif

}

void ObjectTracker25D::pausedUpdate(cv::Mat in_frame)
{
  if (use_displays_ || write_to_disk_)
  {
    trackerDisplay(in_frame, previous_state_, previous_obj_);
  }
  record_count_++;
}


//
// I/O Functions
//

void ObjectTracker25D::trackerDisplay(cv::Mat& in_frame, ProtoObject& cur_obj, cv::RotatedRect& obj_ellipse)
{
  cv::Mat centroid_frame;
  in_frame.copyTo(centroid_frame);
  pcl16::PointXYZ centroid_point(cur_obj.centroid[0], cur_obj.centroid[1],
                                 cur_obj.centroid[2]);
  const cv::Point img_c_idx = pcl_segmenter_->projectPointIntoImage(
      centroid_point, cur_obj.cloud.header.frame_id, camera_frame_);
  // double ellipse_angle_rad = subPIAngle(DEG2RAD(obj_ellipse.angle));
  double theta = getThetaFromEllipse(obj_ellipse);
  if(swap_orientation_)
  {
    theta += (theta > 0.0) ? - M_PI : M_PI;
  }
  const float x_min_rad = (std::cos(theta+0.5*M_PI)* obj_ellipse.size.width*0.5);
  const float y_min_rad = (std::sin(theta+0.5*M_PI)* obj_ellipse.size.width*0.5);
  pcl16::PointXYZ table_min_point(centroid_point.x+x_min_rad, centroid_point.y+y_min_rad,
                                  centroid_point.z);
  const float x_maj_rad = (std::cos(theta)*obj_ellipse.size.height*0.5);
  const float y_maj_rad = (std::sin(theta)*obj_ellipse.size.height*0.5);
  pcl16::PointXYZ table_maj_point(centroid_point.x+x_maj_rad, centroid_point.y+y_maj_rad,
                                  centroid_point.z);
  const cv::Point2f img_min_idx = pcl_segmenter_->projectPointIntoImage(
      table_min_point, cur_obj.cloud.header.frame_id, camera_frame_);
  const cv::Point2f img_maj_idx = pcl_segmenter_->projectPointIntoImage(
      table_maj_point, cur_obj.cloud.header.frame_id, camera_frame_);
  cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,0),3);
  cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,255),1);
  cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,0,0),3);
  cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,255,0),1);
  cv::Size img_size;
  img_size.width = std::sqrt(std::pow(img_maj_idx.x-img_c_idx.x,2) +
                             std::pow(img_maj_idx.y-img_c_idx.y,2))*2.0;
  img_size.height = std::sqrt(std::pow(img_min_idx.x-img_c_idx.x,2) +
                              std::pow(img_min_idx.y-img_c_idx.y,2))*2.0;
  float img_angle = RAD2DEG(std::atan2(img_maj_idx.y-img_c_idx.y,
                                       img_maj_idx.x-img_c_idx.x));
  cv::RotatedRect img_ellipse(img_c_idx, img_size, img_angle);
  cv::ellipse(centroid_frame, img_ellipse, cv::Scalar(0,0,0), 3);
  cv::ellipse(centroid_frame, img_ellipse, cv::Scalar(0,255,255), 1);
  if (use_displays_)
  {
    if (frame_count_ < 1)
    {
      cv::imshow("Init State", centroid_frame);
    }
    else
    {
      cv::imshow("Object State", centroid_frame);
    }
  }
  if (write_to_disk_ && !isPaused())
  {
    // ROS_INFO_STREAM("Writing ellipse to disk!");
    std::stringstream out_name;
    out_name << base_output_path_ << "obj_state_" << frame_set_count_ << "_"
             << record_count_ << ".png";
    cv::imwrite(out_name.str(), centroid_frame);
  }
}

void ObjectTracker25D::trackerBoxDisplay(cv::Mat& in_frame, ProtoObject& cur_obj, cv::RotatedRect& obj_ellipse)
{
  cv::Mat centroid_frame;
  in_frame.copyTo(centroid_frame);
  pcl16::PointXYZ centroid_point(cur_obj.centroid[0], cur_obj.centroid[1],
                                 cur_obj.centroid[2]);
  const cv::Point img_c_idx = pcl_segmenter_->projectPointIntoImage(
      centroid_point, cur_obj.cloud.header.frame_id, camera_frame_);
  // double ellipse_angle_rad = subPIAngle(DEG2RAD(obj_ellipse.angle));
  double theta = getThetaFromEllipse(obj_ellipse);
  if(swap_orientation_)
  {
    theta += (theta > 0.0) ? - M_PI : M_PI;
  }
  const float x_min_rad = (std::cos(theta+0.5*M_PI)* obj_ellipse.size.width*0.5);
  const float y_min_rad = (std::sin(theta+0.5*M_PI)* obj_ellipse.size.width*0.5);
  pcl16::PointXYZ table_min_point(centroid_point.x+x_min_rad, centroid_point.y+y_min_rad,
                                  centroid_point.z);
  const float x_maj_rad = (std::cos(theta)*obj_ellipse.size.height*0.5);
  const float y_maj_rad = (std::sin(theta)*obj_ellipse.size.height*0.5);
  pcl16::PointXYZ table_maj_point(centroid_point.x+x_maj_rad, centroid_point.y+y_maj_rad,
                                  centroid_point.z);
  const cv::Point2f img_min_idx = pcl_segmenter_->projectPointIntoImage(
      table_min_point, cur_obj.cloud.header.frame_id, camera_frame_);
  const cv::Point2f img_maj_idx = pcl_segmenter_->projectPointIntoImage(
      table_maj_point, cur_obj.cloud.header.frame_id, camera_frame_);
  cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,0),3);
  cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,255),1);
  cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,0,0),3);
  cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,255,0),1);
  cv::Size img_size;
  img_size.width = std::sqrt(std::pow(img_maj_idx.x-img_c_idx.x,2) +
                             std::pow(img_maj_idx.y-img_c_idx.y,2))*2.0;
  img_size.height = std::sqrt(std::pow(img_min_idx.x-img_c_idx.x,2) +
                              std::pow(img_min_idx.y-img_c_idx.y,2))*2.0;
  float img_angle = RAD2DEG(std::atan2(img_maj_idx.y-img_c_idx.y,
                                       img_maj_idx.x-img_c_idx.x));
  cv::RotatedRect img_ellipse(img_c_idx, img_size, img_angle);
  cv::Point2f vertices[4];
  img_ellipse.points(vertices);
  for (int i = 0; i < 4; i++)
  {
    cv::line(centroid_frame, vertices[i], vertices[(i+1)%4], cv::Scalar(0,0,0), 3);
  }
  for (int i = 0; i < 4; i++)
  {
    cv::line(centroid_frame, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,255), 1);
  }
  if (use_displays_)
  {
    if (frame_count_ < 1)
    {
      cv::imshow("Init State", centroid_frame);
    }
    else
    {
      cv::imshow("Object State", centroid_frame);
    }
  }
  if (write_to_disk_ && !isPaused())
  {
    std::stringstream out_name;
    out_name << base_output_path_ << "obj_state_" << frame_set_count_ << "_"
             << record_count_ << ".png";
    cv::imwrite(out_name.str(), centroid_frame);
  }
}

void ObjectTracker25D::trackerDisplay(cv::Mat& in_frame, PushTrackerState& state, ProtoObject& obj, bool other_color)
{
  cv::Mat centroid_frame;
  in_frame.copyTo(centroid_frame);
  pcl16::PointXYZ centroid_point(state.x.x, state.x.y, state.z);
  const cv::Point img_c_idx = pcl_segmenter_->projectPointIntoImage(
      centroid_point, obj.cloud.header.frame_id, camera_frame_);
  double theta = state.x.theta;

  // TODO: Change this based on proxy?
  const float x_min_rad = (std::cos(theta+0.5*M_PI)*0.05);
  const float y_min_rad = (std::sin(theta+0.5*M_PI)*0.05);
  pcl16::PointXYZ table_min_point(centroid_point.x+x_min_rad, centroid_point.y+y_min_rad,
                                  centroid_point.z);
  const float x_maj_rad = (std::cos(theta)*0.15);
  const float y_maj_rad = (std::sin(theta)*0.15);
  pcl16::PointXYZ table_maj_point(centroid_point.x+x_maj_rad, centroid_point.y+y_maj_rad,
                                  centroid_point.z);
  const cv::Point2f img_min_idx = pcl_segmenter_->projectPointIntoImage(
      table_min_point, obj.cloud.header.frame_id, camera_frame_);
  const cv::Point2f img_maj_idx = pcl_segmenter_->projectPointIntoImage(
      table_maj_point, obj.cloud.header.frame_id, camera_frame_);
  cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,0),3);
  cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,0,0),3);
  if( other_color)
  {
    cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(255, 255,0),1);
    cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0, 255, 255),1);
  }
  else
  {
    cv::line(centroid_frame, img_c_idx, img_maj_idx, cv::Scalar(0,0,255),1);
    cv::line(centroid_frame, img_c_idx, img_min_idx, cv::Scalar(0,255,0),1);
  }
  cv::Size img_size;
  img_size.width = std::sqrt(std::pow(img_maj_idx.x-img_c_idx.x,2) +
                             std::pow(img_maj_idx.y-img_c_idx.y,2))*2.0;
  img_size.height = std::sqrt(std::pow(img_min_idx.x-img_c_idx.x,2) +
                              std::pow(img_min_idx.y-img_c_idx.y,2))*2.0;
  // float img_angle = RAD2DEG(std::atan2(img_maj_idx.y-img_c_idx.y,
  //                                      img_maj_idx.x-img_c_idx.x));
  // cv::RotatedRect img_ellipse(img_c_idx, img_size, img_angle);
  // cv::ellipse(centroid_frame, img_ellipse, cv::Scalar(0,0,0), 3);
  // cv::ellipse(centroid_frame, img_ellipse, cv::Scalar(0,255,255), 1);

  if (use_displays_)
  {
    if (frame_count_ < 1)
    {
      cv::imshow("Init State", centroid_frame);
    }
    else
    {
      cv::imshow("Object State", centroid_frame);
    }
  }
  if (write_to_disk_ && !isPaused())
  {
    std::stringstream out_name;
    out_name << base_output_path_ << "obj_state_" << frame_set_count_ << "_"
             << record_count_ << ".png";
    cv::imwrite(out_name.str(), centroid_frame);
  }
}

cv::Mat ObjectTracker25D::getTableMask(XYZPointCloud& cloud, XYZPointCloud& table_cloud, cv::Size mask_size,
                                       XYZPointCloud& obj_cloud)
{
  cv::Mat table_mask(mask_size, CV_8UC1, cv::Scalar(0));
  Eigen::Vector4f table_centroid;
  pcl_segmenter_->getTablePlane(cloud, obj_cloud, table_cloud, table_centroid);
  pcl_segmenter_->projectPointCloudIntoImage(table_cloud, table_mask, camera_frame_, 255);
  return table_mask;
}


GMM ObjectTracker25D::buildColorModel(XYZPointCloud& cloud, cv::Mat& frame, int nc)
{
  cv::Mat frame_lab_uchar(frame.size(), frame.type());
  cv::Mat frame_lab(frame.size(), CV_32FC3);
  cv::cvtColor(frame, frame_lab_uchar, CV_BGR2HSV);
  frame_lab_uchar.convertTo(frame_lab, CV_32FC3, 1.0/255);

  std::vector<cv::Vec3f> pnts;
  for (int i = 0; i < cloud.size(); ++i)
  {
    cv::Point img_pt = pcl_segmenter_->projectPointIntoImage(cloud.at(i), cloud.header.frame_id, camera_frame_);
    cv::Vec3f img_val = frame_lab.at<cv::Vec3f>(img_pt.y, img_pt.x);
    pnts.push_back(img_val);
  }
  ROS_INFO_STREAM("Have " << pnts.size() << " points for the model");
  GMM color_model(0.0001);
  color_model.alloc(nc);
  if (pnts.size() > 1)
  {
    color_model.kmeansInit(pnts, 0.05);
    color_model.GmmEm(pnts);
    color_model.dispparams();
  }
  return color_model;
}
