#include <sstream>
#include <iostream>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <pcl16_ros/transforms.h>
#include <pcl16/ros/conversions.h>
#include <pcl16/surface/concave_hull.h>
#include <pcl16/common/pca.h>
// #include <pcl16/registration/transformation_estimation_svd.h>
#include <pcl16/registration/transformation_estimation_lm.h>
#include <cpl_visual_features/comp_geometry.h>
#include <cpl_visual_features/helpers.h>
#include <iostream>

#include <Eigen/Eigenvalues>

#define XY_RES 0.00075
#define DRAW_LR_LIMITS 1
// #define USE_RANGE_AND_VAR_FEATS 1

using namespace cpl_visual_features;
using tabletop_pushing::ProtoObject;
typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;

namespace tabletop_pushing
{

inline int getHistBinIdx(int x_idx, int y_idx, int n_x_bins, int n_y_bins)
{
  return x_idx*n_y_bins+y_idx;
}

inline int worldLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}

inline int objLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}

cv::Point worldPtToImgPt(pcl16::PointXYZ world_pt, double min_x, double max_x,
                         double min_y, double max_y)
{
  cv::Point img_pt(worldLocToIdx(world_pt.x, min_x, max_x),
                   worldLocToIdx(world_pt.y, min_y, max_y));
  return img_pt;
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

std::vector<int> getJumpIndices(XYZPointCloud& concave_hull, double alpha)
{
  std::vector<int> jump_indices;
  for (int i = 0; i < concave_hull.size(); i++)
  {
    if (dist(concave_hull[i], concave_hull[(i+1)%concave_hull.size()]) > 2.0*alpha)
    {
      jump_indices.push_back(i);
    }
  }
  return jump_indices;
}

XYZPointCloud getObjectBoundarySamples(ProtoObject& cur_obj, double hull_alpha)
{
  // Get 2D projection of object
  // TODO: Remove the z, then add it back after finding the hull... how do we do this?
  XYZPointCloud footprint_cloud(cur_obj.cloud);
  for (int i = 0; i < footprint_cloud.size(); ++i)
  {
    // HACK: This is a complete hack, based on the current table height used in pushing.
    footprint_cloud.at(i).z = -0.3;
  }

  // TODO: Examine sensitivity of hull_alpha...
  XYZPointCloud hull_cloud;
  pcl16::ConcaveHull<pcl16::PointXYZ> hull;
  hull.setDimension(2);  // NOTE: Get 2D projection of object
  hull.setInputCloud(footprint_cloud.makeShared());
  hull.setAlpha(hull_alpha);
  hull.reconstruct(hull_cloud);
  return hull_cloud;
}

cpl_visual_features::Path compareBoundaryShapes(XYZPointCloud& hull_a, XYZPointCloud& hull_b,
                                                double& min_cost, double epsilon_cost)
{
  Samples2f samples_a, samples_b;
  for (unsigned int i = 0; i < hull_a.size(); ++i)
  {
    cv::Point2f pt(hull_a[i].x, hull_a[i].y);
    samples_a.push_back(pt);
  }
  for (unsigned int i = 0; i < hull_b.size(); ++i)
  {
    cv::Point2f pt(hull_b[i].x, hull_b[i].y);
    samples_b.push_back(pt);
  }
  // TODO: Expose these parameters
  int radius_bins = 5;
  int theta_bins = 12;
  ShapeDescriptors descriptors_a = constructDescriptors<Samples2f>(samples_a, radius_bins, theta_bins);
  ShapeDescriptors descriptors_b = constructDescriptors<Samples2f>(samples_b, radius_bins, theta_bins);
  // ROS_INFO_STREAM("Computing cost matrix");
  cv::Mat cost_mat = computeCostMatrix(descriptors_a, descriptors_b, epsilon_cost);
  cv::imshow("cost matrix", cost_mat);
  cpl_visual_features::Path path;
  // ROS_INFO_STREAM("Getting min cost path");
  min_cost = getMinimumCostPath(cost_mat, path);

  return path;
}

void estimateTransformFromMatches(XYZPointCloud& cloud_t_0, XYZPointCloud& cloud_t_1,
                                  cpl_visual_features::Path p, Eigen::Matrix4f& transform, double max_dist)
{
  // Do least squares estimate of the change
  std::vector<int> source_indices;
  std::vector<int> target_indices;
  // Get inliers
  // TODO: Make sure have at least 1
  // TODO: Make max dist the average distance of matches? (median+epsilon?)
  // ROS_INFO_STREAM("Path has: " << p.size() << " elements");
  // ROS_INFO_STREAM("cloud_t_0 has: " << cloud_t_0.size() << " elements");
  // ROS_INFO_STREAM("cloud_t_1 has: " << cloud_t_1.size() << " elements");
  for (int i = 0; i < p.size(); ++i)
  {
    // NOTE: Filter out matches with distance greater than max_dist
    if(i < cloud_t_0.size() && p[i] < cloud_t_1.size() &&
       PointCloudSegmentation::sqrDist(cloud_t_0.at(i), cloud_t_1.at(p[i])) < max_dist)
    {
      source_indices.push_back(i);
      target_indices.push_back(p[i]);
    }
  }
  ROS_INFO_STREAM("Path had " << p.size() << " matches.\tNow have " << source_indices.size() << " matches.");
  pcl16::registration::TransformationEstimationLM<pcl16::PointXYZ, pcl16::PointXYZ> tlm;
  tlm.estimateRigidTransformation(cloud_t_0, source_indices, cloud_t_1, target_indices, transform);
}

cv::Mat visualizeObjectBoundarySamples(XYZPointCloud& hull_cloud, PushTrackerState& cur_state)
{
  double max_y = 0.2;
  double min_y = -0.2;
  double max_x = 0.2;
  double min_x = -0.2;
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC3, cv::Scalar(255,255,255));

  cv::Point prev_img_pt, init_img_pt;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    pcl16::PointXYZ obj_pt =  worldPointInObjectFrame(hull_cloud[i], cur_state);
    cv::Point img_pt(cols - objLocToIdx(obj_pt.x, min_x, max_x), objLocToIdx(obj_pt.y, min_y, max_y));
    cv::Scalar color(128, 0, 0);
    cv::circle(footprint, img_pt, 1, color, 3);
    if (i > 0)
    {
      cv::line(footprint, img_pt, prev_img_pt, color, 1);
    }
    else
    {
      init_img_pt = img_pt;
    }
    prev_img_pt = img_pt;
  }
  cv::line(footprint, prev_img_pt, init_img_pt, cv::Scalar(0,128,0),1);
  return footprint;
}

cv::Mat visualizeObjectBoundaryMatches(XYZPointCloud& hull_a, XYZPointCloud& hull_b,
                                       PushTrackerState& cur_state, cpl_visual_features::Path& path)
{
  double max_y = 0.2;
  double min_y = -0.2;
  double max_x = 0.2;
  double min_x = -0.2;
  double max_displacement = 100;
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat match_img(rows, cols, CV_8UC3, cv::Scalar(255,255,255));

  cv::Scalar blue(128, 0, 0);
  cv::Scalar green(0, 128, 0);
  cv::Scalar red(0, 0, 128);

  for (int i = 0; i < hull_a.size(); ++i)
  {
    pcl16::PointXYZ start_point_world = hull_a[i];
    pcl16::PointXYZ end_point_world = hull_b[path[i]];
    // Transform to display frame
    pcl16::PointXYZ start_point_obj = worldPointInObjectFrame(start_point_world, cur_state);
    pcl16::PointXYZ end_point_obj = worldPointInObjectFrame(end_point_world, cur_state);
    cv::Point start_point(cols - objLocToIdx(start_point_obj.x, min_x, max_x),
                          objLocToIdx(start_point_obj.y, min_y, max_y));
    cv::Point end_point(cols - objLocToIdx(end_point_obj.x, min_x, max_x),
                        objLocToIdx(end_point_obj.y, min_y, max_y));

    // Draw green point for previous, blue point for current
    cv::circle(match_img, start_point, 1, green, 3);
    cv::circle(match_img, end_point, 1, blue, 3);
    // Draw red line between matches
    if (end_point.x > 0 && end_point.x < match_img.rows &&
        end_point.y > 0 && end_point.x < match_img.cols &&
        std::abs(start_point.x - end_point.x) +
        std::abs(start_point.y - end_point.y) < max_displacement)
    {
      cv::line(match_img, start_point, end_point, red, 3);
    }

  }
  return match_img;
}

cv::Mat visualizeObjectContactLocation(XYZPointCloud& hull_cloud, PushTrackerState& cur_state,
                                       pcl16::PointXYZ& tool_pt0, pcl16::PointXYZ& tool_pt1)
{
  double max_y = 0.2;
  double min_y = -0.2;
  double max_x = 0.2;
  double min_x = -0.2;
  cv::Mat footprint = visualizeObjectBoundarySamples(hull_cloud, cur_state);

  pcl16::PointXYZ contact_pt_world = estimateObjectContactLocation(hull_cloud, cur_state,
                                                                   tool_pt0, tool_pt1);
  pcl16::PointXYZ contact_pt_obj =  worldPointInObjectFrame(contact_pt_world, cur_state);
  int img_contact_pt_x = footprint.cols - objLocToIdx(contact_pt_obj.x, min_x, max_x);
  int img_contact_pt_y = objLocToIdx(contact_pt_obj.y, min_y, max_y);

  pcl16::PointXYZ tool_pt_obj0 =  worldPointInObjectFrame(tool_pt0, cur_state);
  int img_x0 = footprint.cols - objLocToIdx(tool_pt_obj0.x, min_x, max_x);
  int img_y0 = objLocToIdx(tool_pt_obj0.y, min_y, max_y);
  pcl16::PointXYZ tool_pt_obj1 =  worldPointInObjectFrame(tool_pt1, cur_state);
  int img_x1 = footprint.cols - objLocToIdx(tool_pt_obj1.x, min_x, max_x);
  int img_y1 = objLocToIdx(tool_pt_obj1.y, min_y, max_y);
  cv::Scalar color(0, 0, 128);
  cv::line(footprint, cv::Point(img_x0, img_y0), cv::Point(img_x1, img_y1), color, 1);
  cv::circle(footprint, cv::Point(img_x0, img_y0), 3, color, 3);
  cv::circle(footprint, cv::Point(img_x1, img_y1), 3, color, 3);
  cv::circle(footprint, cv::Point(img_contact_pt_x, img_contact_pt_y), 3, cv::Scalar(0, 128, 128), 3);
  return footprint;
}

cv::Mat getObjectFootprint(cv::Mat obj_mask, pcl16::PointCloud<pcl16::PointXYZ>& cloud)
{
  cv::Mat kernel(5,5,CV_8UC1, 255);
  cv::Mat obj_mask_target;
  obj_mask.copyTo(obj_mask_target);
  // Perform close
  cv::dilate(obj_mask, obj_mask_target, kernel);
  cv::erode(obj_mask_target, obj_mask, kernel);
  // cv::erode(obj_mask, obj_mask, kernel);
  double min_x = 300., max_x = -300.;
  double min_y = 300., max_y = -300.;

  for (int r = 0; r < obj_mask.rows; ++r)
  {
    for (int c = 0; c < obj_mask.cols; ++c)
    {
      if (obj_mask.at<uchar>(r,c) > 0)
      {
        double x = cloud.at(c,r).x;
        double y = cloud.at(c,r).y;
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
      }
    }
  }
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC1, cv::Scalar(0));
  for (int r = 0; r < obj_mask.rows; ++r)
  {
    for (int c = 0; c < obj_mask.cols; ++c)
    {
      if (obj_mask.at<uchar>(r,c) > 0)
      {
        // TODO: Allow different max z in image.
        double x = cloud.at(c,r).x;
        double y = cloud.at(c,r).y;
        if (isnan(x) || isnan(y))
          continue;
        int img_x = worldLocToIdx(x, min_x, max_x);
        int img_y = worldLocToIdx(y, min_y, max_y);
        if (img_y > rows ||img_x > cols || img_y < 0 || img_x < 0)
        {
          ROS_WARN_STREAM("Image index out of bounds! at (" << img_y << ", " << img_x <<
                          ", " << y << ", " << x << ")");
          ROS_WARN_STREAM("X extremes: (" << min_x << ", " << max_x << ")");
          ROS_WARN_STREAM("Y extremes: (" << min_y << ", " << max_y << ")");
          ROS_WARN_STREAM("image size: (" << rows << ", " << cols << ")");
        }
        footprint.at<uchar>(img_y, img_x) = 255;
      }
    }
  }
  // Perform close
  cv::dilate(footprint, footprint, kernel);
  cv::erode(footprint, footprint, kernel);
  // Perform open
  cv::erode(footprint, footprint, kernel);
  cv::dilate(footprint, footprint, kernel);
  return footprint;
}

ShapeLocations extractObjectShapeContext(ProtoObject& cur_obj, bool use_center)
{
  XYZPointCloud samples_pcl = getObjectBoundarySamples(cur_obj);
  return extractShapeContextFromSamples(samples_pcl, cur_obj, use_center);
}

ShapeLocations extractShapeContextFromSamples(XYZPointCloud& samples_pcl, ProtoObject& cur_obj, bool use_center)
{
  Samples2f samples;
  for (unsigned int i = 0; i < samples_pcl.size(); ++i)
  {
    cv::Point2f pt(samples_pcl[i].x, samples_pcl[i].y);
    samples.push_back(pt);
  }
  int radius_bins = 5;
  int theta_bins = 12;
  double max_radius = 0.5;
  cv::Point2f center(cur_obj.centroid[0], cur_obj.centroid[1]);
  ShapeDescriptors descriptors = constructDescriptors(samples, center, use_center, radius_bins, theta_bins, max_radius);
  ShapeLocations locs;
  for (unsigned int i = 0; i < descriptors.size(); ++i)
  {
    ShapeLocation loc(samples_pcl[i], descriptors[i]);
    locs.push_back(loc);
  }
  // computeShapeFeatureAffinityMatrix(locs, use_center);
  return locs;
}

cv::Mat makeHistogramImage(ShapeDescriptor histogram, int n_x_bins, int n_y_bins, int bin_width_pixels)
{
  cv::Mat hist_img(n_x_bins*bin_width_pixels+1, n_y_bins*bin_width_pixels+1, CV_8UC1, cv::Scalar(255));
  int max_count = 10;
  for (int i = 0; i < histogram.size(); ++i)
  {
    if (histogram[i] == 0) continue;
    int pix_val = (1 - histogram[i]/max_count)*255;
    int start_x = (i%n_x_bins)*bin_width_pixels;
    int start_y = (i/n_x_bins)*bin_width_pixels;;
    // ROS_INFO_STREAM("(start_x, start_y): (" << start_x << ", " << start_y << ")");
    for (int x = start_x; x < start_x+bin_width_pixels; ++x)
    {
      for (int y = start_y; y < start_y+bin_width_pixels; ++y)
      {
        hist_img.at<uchar>(y, x) = pix_val;
      }
    }
  }
  // Draw hist bin columns
  for (int i = 0; i <= n_x_bins; ++i)
  {
    int x = i*bin_width_pixels;
    for (int y = 0; y < hist_img.rows; ++y)
    {
      hist_img.at<uchar>(y,x) = 0;
    }
  }
  // Draw hist bin rows
  for (int i = 0; i <= n_y_bins; ++i)
  {
    int y = i*bin_width_pixels;
    for (int x = 0; x < hist_img.cols; ++x)
    {
      hist_img.at<uchar>(y,x) = 0;
    }
  }
  // cv::imshow("local_hist_img", hist_img);
  // cv::waitKey();
  return hist_img;
}
void drawSamplePoints(XYZPointCloud& hull, XYZPointCloud& samples, double alpha, pcl16::PointXYZ& center_pt,
                      pcl16::PointXYZ& sample_pt, pcl16::PointXYZ& approach_pt,
                      pcl16::PointXYZ e_left, pcl16::PointXYZ e_right,
                      pcl16::PointXYZ c_left, pcl16::PointXYZ c_right,
                      pcl16::PointXYZ i_left, pcl16::PointXYZ i_right,
                      double x_res, double y_res, double x_range, double y_range, ProtoObject& cur_obj)
{
  double max_y = 0.5;
  double min_y = -0.2;
  double max_x = 1.0;
  double min_x = 0.2;
  // TODO: Make function to get cv::Size from (max_x, min_x, max_y, min_y, XY_RES)
  // TODO: Make sure everything is getting drawn
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC3, cv::Scalar(255,255,255));
  std::vector<int> jump_indices = getJumpIndices(hull, alpha);
  for (int i = 0; i < hull.size(); ++i)
  {
    int j = (i+1) % hull.size();
    bool is_jump = false;
    for (int k = 0; k < jump_indices.size(); ++k)
    {
      if (i == jump_indices[k])
      {
        is_jump = true;
      }
    }
    if (is_jump)
    {
      continue;
    }
    pcl16::PointXYZ obj_pt0 = hull[i];
    pcl16::PointXYZ obj_pt1 = hull[j];
    cv::Point img_pt0 = worldPtToImgPt(obj_pt0, min_x, max_x, min_y, max_y);
    cv::Point img_pt1 = worldPtToImgPt(obj_pt1, min_x, max_x, min_y, max_y);
    cv::Scalar color(0, 0, 128);
    cv::circle(footprint, img_pt0, 1, color, 3);
    cv::line(footprint, img_pt0, img_pt1, color, 1);
  }
  for (int i = 0; i < samples.size(); ++i)
  {
    cv::Point img_pt = worldPtToImgPt(samples[i], min_x, max_x, min_y, max_y);
    cv::Scalar color(0, 255, 0);
    cv::circle(footprint, img_pt, 1, color,3);
    cv::circle(footprint, img_pt, 3, color,3);
  }
  cv::Point img_center = worldPtToImgPt(center_pt, min_x, max_x, min_y, max_y);
  cv::Point img_approach_pt = worldPtToImgPt(approach_pt, min_x, max_x, min_y, max_y);
  cv::Point img_sample_pt = worldPtToImgPt(sample_pt, min_x, max_x, min_y, max_y);
  cv::Point e_left_img = worldPtToImgPt(e_left, min_x, max_x, min_y, max_y);
  cv::Point e_right_img = worldPtToImgPt(e_right, min_x, max_x, min_y, max_y);
  cv::Point c_left_img = worldPtToImgPt(c_left, min_x, max_x, min_y, max_y);
  cv::Point c_right_img = worldPtToImgPt(c_right, min_x, max_x, min_y, max_y);
  cv::Point i_left_img = worldPtToImgPt(i_left, min_x, max_x, min_y, max_y);
  cv::Point i_right_img = worldPtToImgPt(i_right, min_x, max_x, min_y, max_y);

  cv::Mat footprint_hist;
  footprint.copyTo(footprint_hist);

  cv::line(footprint, img_sample_pt, img_center, cv::Scalar(255,0,0), 2);
  cv::line(footprint, e_left_img, e_right_img, cv::Scalar(255,0,0), 2);
  // cv::line(footprint, e_left_img, c_left_img, cv::Scalar(255,0,255));
  // cv::line(footprint, e_right_img, c_right_img, cv::Scalar(0,255,255));
  cv::circle(footprint, img_center, 3, cv::Scalar(255,0,0), 2);
  // cv::circle(footprint, img_approach_pt, 3, cv::Scalar(255,0,0),2);
  // cv::circle(footprint, e_left_img, 3, cv::Scalar(255,0,255));
  // cv::circle(footprint, e_right_img, 3, cv::Scalar(0,255,255));


  // Draw sample point last
  // cv::line(footprint, img_sample_pt, img_center, cv::Scalar(255,255,255));
  cv::circle(footprint, img_sample_pt, 1, cv::Scalar(255,0,255),3);
  cv::circle(footprint, img_sample_pt, 3, cv::Scalar(255,0,255),3);
  // cv::circle(footprint, i_left_img, 3, cv::Scalar(255,0,255));
  // cv::circle(footprint, i_right_img, 3, cv::Scalar(0,255,255));

  int n_x_bins = ceil(x_range/x_res);
  int n_y_bins = ceil(y_range/y_res);
  double min_hist_x = -x_range*0.5;
  double min_hist_y = -y_range*0.5;
  std::vector<std::vector<cv::Point> > hist_corners;
  double center_angle = std::atan2(center_pt.y - sample_pt.y, center_pt.x - sample_pt.x);
  double ct = std::cos(center_angle);
  double st = std::sin(center_angle);
  for (int i = 0; i <= n_x_bins; ++i)
  {
    double local_x = min_hist_x+x_res*i;
    std::vector<cv::Point> hist_row;
    hist_corners.push_back(hist_row);
    for (int j = 0; j <= n_y_bins; ++j)
    {
      double local_y = min_hist_y+y_res*j;

      // Transform into world frame
      pcl16::PointXYZ corner_in_world;
      corner_in_world.x = (local_x*ct - local_y*st ) + sample_pt.x;
      corner_in_world.y = (local_x*st + local_y*ct ) + sample_pt.y;
      // Transform into image frame
      cv::Point corner_in_img = worldPtToImgPt(corner_in_world, min_x, max_x, min_y, max_y);
      // draw point
      cv::Scalar color(0,0,0);
      cv::circle(footprint_hist, corner_in_img, 1, color);
      // Draw grid lines
      if (i > 0)
      {
        cv::line(footprint_hist, corner_in_img, hist_corners[i-1][j], color);
      }
      if (j > 0)
      {
        cv::line(footprint_hist, corner_in_img, hist_corners[i][j-1], color);
      }
      hist_corners[i].push_back(corner_in_img);
    }
  }

  // cv::imshow("local samples", footprint);
  // cv::imshow("local hist", footprint_hist);
  // char c = cv::waitKey();
  // if (c == 'w')
  // {
  //   std::stringstream boundary_img_write_path;
  //   std::stringstream hist_img_write_path;
  //   int rand_int = rand();
  //   boundary_img_write_path << "/home/thermans/Desktop/" << rand_int << "_boundary_img_.png";
  //   hist_img_write_path << "/home/thermans/Desktop/" << rand_int << "_boundary_hist_img.png";
  //   std::cout << "Writing " << boundary_img_write_path.str() << " to disk";
  //   cv::imwrite(boundary_img_write_path.str(), footprint);
  //   std::cout << "Writing " << hist_img_write_path.str() << " to disk";
  //   cv::imwrite(hist_img_write_path.str(), footprint_hist);

  //   // HACK: Huge hack just recompute descriptor to write to disk like others
  //   XYZPointCloud transformed_pts = transformSamplesIntoSampleLocFrame(samples, cur_obj, sample_pt);
  //   ShapeDescriptor histogram = extractPointHistogramXY(transformed_pts, x_res, y_res, x_range, y_range);
  //   int bin_width_pixels = 10;
  //   cv::Mat hist_img = makeHistogramImage(histogram, n_x_bins, n_y_bins, bin_width_pixels);
  //   std::stringstream out_file_name;
  //   out_file_name << "/home/thermans/Desktop/" << rand_int << "_local_hist_img.png";
  //   cv::imwrite(out_file_name.str(), hist_img);
  // }

}

XYZPointCloud getLocalSamplesNew(XYZPointCloud& hull, ProtoObject& cur_obj, pcl16::PointXYZ sample_pt,
                              double sample_spread, double alpha)
{
  // TODO: This is going to get all points in front of the gripper, not only those on the close boundary
  double radius = sample_spread / 2.0;
  pcl16::PointXYZ center_pt(cur_obj.centroid[0], cur_obj.centroid[1], cur_obj.centroid[2]);
  double center_angle = std::atan2(center_pt.y - sample_pt.y, center_pt.x - sample_pt.x);
  double approach_dist = 0.05;
  pcl16::PointXYZ approach_pt(sample_pt.x - std::cos(center_angle)*approach_dist,
                              sample_pt.y - std::sin(center_angle)*approach_dist, 0.0);
  pcl16::PointXYZ e_vect(std::cos(center_angle+M_PI/2.0)*radius,
                         std::sin(center_angle+M_PI/2.0)*radius, 0.0);
  pcl16::PointXYZ e_left(approach_pt.x + e_vect.x, approach_pt.y + e_vect.y, 0.0);
  pcl16::PointXYZ e_right(approach_pt.x - e_vect.x, approach_pt.y - e_vect.y, 0.0);
  pcl16::PointXYZ c_left(center_pt.x + std::cos(center_angle)*approach_dist + e_vect.x,
                         center_pt.y + std::sin(center_angle)*approach_dist + e_vect.y, 0.0);
  pcl16::PointXYZ c_right(center_pt.x + std::cos(center_angle)*approach_dist - e_vect.x,
                          center_pt.y + std::sin(center_angle)*approach_dist - e_vect.y, 0.0);
  std::vector<int> jump_indices = getJumpIndices(hull, alpha);

  std::vector<int> inside_indices;
  for (int i = 0; i < cur_obj.cloud.size(); i++)
  {
    double dist_r = pointLineDistance2D(cur_obj.cloud[i], e_right, c_right);
    double dist_l = pointLineDistance2D(cur_obj.cloud[i], e_left, c_left);
    if (dist_r > sample_spread  || dist_l > sample_spread)
    {
      // pcl16::PointXYZ intersection;
      // bool intersects_l = lineSegmentIntersection2D(hull[i], hull[(i+1)%hull.size()], e_left, c_left,
      //                                               intersection);
      // bool intersects_r = lineSegmentIntersection2D(hull[i], hull[(i+1)%hull.size()], e_right, c_right,
      //                                               intersection);
      // if (intersects_l || intersects_r)
      // {
      //   inside_indices.push_back(i);
      // }
    }
    else
    {
      inside_indices.push_back(i);
    }
  }

  std::vector<int> local_indices;
  for (int i = 0; i < inside_indices.size(); ++i)
  {
    const int idx = inside_indices[i];
    // Get projection of this point in gripper y
    double dist_l = pointLineDistance2D(cur_obj.cloud[idx], e_left, c_left);
    pcl16::PointXYZ e_vect_scaled(std::cos(center_angle+M_PI/2.0)*radius-dist_l,
                                  std::sin(center_angle+M_PI/2.0)*radius-dist_l, 0.0);
    pcl16::PointXYZ e_pt(approach_pt.x + e_vect.x, approach_pt.y + e_vect.y, 0.0);

    // Check if the line segment between the gripper and this point intersects any other line segment
    bool intersects = false;
    for (int j = 0; j < hull.size(); ++j)
    {
      bool j_is_jump = false;
      for (int k = 0; k < jump_indices.size(); ++k)
      {
        if (jump_indices[k] == j)
        {
          j_is_jump = true;
        }
      }
      // Don't test jump indices for blocking, or ourself
      // if (j_is_jump || j == idx || j == idx-1 || j == (idx+1)%hull.size())
      // {
      //   continue;
      // }

      // Do the line test
      pcl16::PointXYZ intersection;
      if (lineSegmentIntersection2D(e_pt, cur_obj.cloud[idx], hull[j], hull[(j+1)%hull.size()], intersection))
      {
        intersects = true;
      }
    }
    if (!intersects)
    {
      local_indices.push_back(idx);
      // local_indices.push_back((idx+1)%hull.size());
    }
  }

  // Copy to new cloud and return
  XYZPointCloud local_samples;
  pcl16::copyPointCloud(cur_obj.cloud, local_indices, local_samples);
  int min_l_idx = 0;
  int min_r_idx = 0;
  // drawSamplePoints(hull, local_samples, center_pt, sample_pt, approach_pt, e_left, e_right,
  //                  c_left, c_right);

  // TODO: Transform samples into sample_loc frame
  return local_samples;
}

XYZPointCloud getLocalSamples(XYZPointCloud& hull, ProtoObject& cur_obj, pcl16::PointXYZ sample_pt,
                                 double sample_spread, double alpha)
{
  // TODO: This is going to get all points in front of the gripper, not only those on the close boundary
  double radius = sample_spread / 2.0;
  pcl16::PointXYZ center_pt(cur_obj.centroid[0], cur_obj.centroid[1], cur_obj.centroid[2]);
  double center_angle = std::atan2(center_pt.y - sample_pt.y, center_pt.x - sample_pt.x);
  double approach_dist = 0.05;
  pcl16::PointXYZ approach_pt(sample_pt.x - std::cos(center_angle)*approach_dist,
                              sample_pt.y - std::sin(center_angle)*approach_dist, 0.0);
  pcl16::PointXYZ e_vect(std::cos(center_angle+M_PI/2.0)*radius,
                         std::sin(center_angle+M_PI/2.0)*radius, 0.0);
  pcl16::PointXYZ e_left(sample_pt.x + e_vect.x, sample_pt.y + e_vect.y, 0.0);
  pcl16::PointXYZ e_right(sample_pt.x - e_vect.x, sample_pt.y - e_vect.y, 0.0);
  pcl16::PointXYZ c_left(center_pt.x + std::cos(center_angle)*approach_dist + e_vect.x,
                         center_pt.y + std::sin(center_angle)*approach_dist + e_vect.y, 0.0);
  pcl16::PointXYZ c_right(center_pt.x + std::cos(center_angle)*approach_dist - e_vect.x,
                          center_pt.y + std::sin(center_angle)*approach_dist - e_vect.y, 0.0);
  // ROS_INFO_STREAM("center_pt: " << center_pt);
  // ROS_INFO_STREAM("sample_pt: " << sample_pt);
  // ROS_INFO_STREAM("approach_pt: " << approach_pt);
  // ROS_INFO_STREAM("e_vect: " << e_vect);
  // ROS_INFO_STREAM("e_left: " << e_left);
  // ROS_INFO_STREAM("e_right: " << e_right);

  double min_sample_pt_dist = FLT_MAX;
  int sample_pt_idx = -1;
  std::vector<int> jump_indices;
  for (int i = 0; i < hull.size(); i++)
  {
    // SAMPLE PT
    double sample_pt_dist = dist(sample_pt, hull[i]);
    if (sample_pt_dist < min_sample_pt_dist)
    {
      min_sample_pt_dist = sample_pt_dist;
      sample_pt_idx = i;
    }
    if (dist(hull[i], hull[(i+1)%hull.size()]) > 2.0*alpha)
    {
      jump_indices.push_back(i);
      // ROS_INFO_STREAM("Jump from " << i << " to " << (i+1)%hull.size());
    }
  }

  // Test intersection of gripper end point rays and all line segments on the object boundary
  pcl16::PointXYZ l_intersection;
  pcl16::PointXYZ r_intersection;
  pcl16::PointXYZ c_intersection;
  double min_l_dist = FLT_MAX;
  double min_r_dist = FLT_MAX;
  double min_c_dist = FLT_MAX;
  int min_l_idx = -1;
  int min_r_idx = -1;
  int min_c_idx = -1;

  double min_far_l_dist = FLT_MAX;
  double min_far_r_dist = FLT_MAX;
  int far_l_idx = -1;
  int far_r_idx = -1;
  // TODO: Break this up into a couple of functions
  for (int i = 0; i < hull.size(); i++)
  {
    int idx0 = i;
    int idx1 = (i+1) % hull.size();

    // FAR LEFT PT
    double far_l_dist = pointLineDistance2D(hull[idx0], e_left, c_left);
    if (far_l_dist < min_far_l_dist)
    {
      far_l_idx = idx0;
      min_far_l_dist = far_l_dist;
    }

    // FAR RIGHT PT
    double far_r_dist = pointLineDistance2D(hull[idx0], e_right, c_right);
    if (far_r_dist < min_far_r_dist)
    {
      far_r_idx = idx0;
      min_far_r_dist = far_r_dist;
    }

    bool is_jump = false;
    for (int j = 0; j < jump_indices.size(); ++j)
    {
      if (jump_indices[j] == idx0)
      {
        is_jump = true;
      }
    }
    if (is_jump)
    {
      continue;
    }

    pcl16::PointXYZ intersection;
    // LEFT INTERSECTION
    if (lineSegmentIntersection2D(hull[idx0], hull[idx1], e_left, c_left, intersection))
    {
      double pt_dist = dist(intersection, e_left);
      if (pt_dist < min_l_dist)
      {
        min_l_dist = pt_dist;
        min_l_idx = i;
        l_intersection = intersection;
      }
    }
    // RIGHT INTERSECTION
    if (lineSegmentIntersection2D(hull[idx0], hull[idx1], e_right, c_right, intersection))
    {
      double pt_dist = dist(intersection, e_right);
      if (pt_dist < min_r_dist)
      {
        min_r_dist = pt_dist;
        min_r_idx = i;
        r_intersection = intersection;
      }
    }
    // CENTER INTERSECTION
    if (lineSegmentIntersection2D(hull[idx0], hull[idx1], approach_pt, center_pt, intersection))
    {
      double pt_dist = dist(intersection, approach_pt);
      if (pt_dist < min_c_dist)
      {
        min_c_dist = pt_dist;
        min_c_idx = i;
        c_intersection = intersection;
      }
    }
  }

  double sample_pt_dist = dist(approach_pt, sample_pt);
  // Default to smaple_pt if no intersection also
  if (true || min_c_idx == -1 || sample_pt_dist <= min_c_dist)
  {
    min_c_idx = sample_pt_idx;
    min_c_dist = sample_pt_dist;
  }
  // Default far left if no left intersection
  if (min_l_idx == -1)
  {
    min_l_idx = far_l_idx;
    min_l_dist = min_far_l_dist;
  }
  // Default far right if no right intersection
  if (min_r_idx == -1)
  {
    min_r_idx = far_r_idx;
    min_r_dist = min_far_r_dist;
  }

  std::vector<int> indices;
  // indices.push_back(min_l_idx);
  // indices.push_back((min_l_idx+1) % hull.size());
  // indices.push_back(min_r_idx);
  // indices.push_back((min_r_idx+1) % hull.size());

  int start_idx = min_l_idx < min_r_idx ? min_l_idx : min_r_idx;
  int end_idx = min_l_idx < min_r_idx ? min_r_idx : min_l_idx;
  if (min_c_idx >= start_idx && min_c_idx <= end_idx)
  {
    // Good:
    // ROS_INFO_STREAM("Walking inside");
  }
  else
  {
    // ROS_INFO_STREAM("Walking outside");
    int tmp_idx = start_idx;
    start_idx = end_idx;
    end_idx = tmp_idx;
  }
  int start_chunk = jump_indices.size();
  int center_chunk = jump_indices.size();
  int end_chunk = jump_indices.size();
  for (int i = jump_indices.size()-1; i >= 0; --i)
  {
    if (start_idx <= jump_indices[i])
    {
      start_chunk = i;
    }
    if (min_c_idx <= jump_indices[i])
    {
      center_chunk = i;
    }
    if (end_idx <= jump_indices[i])
    {
      end_chunk = i;
    }
  }
  // TODO: Dont walk through unimportant chunks
  // ROS_INFO_STREAM("Start chunk is: " << start_chunk);
  // ROS_INFO_STREAM("Center chunk is: " << center_chunk);
  // ROS_INFO_STREAM("End chunk is: " << end_chunk);

  // ROS_INFO_STREAM("far_l_dist is: " << min_far_l_dist << " at " << far_l_idx);
  // ROS_INFO_STREAM("far_r_dist is: " << min_far_r_dist << " at " << far_r_idx);
  // ROS_INFO_STREAM("min_l_dist is: " << min_l_dist << " at " << min_l_idx);
  // ROS_INFO_STREAM("min_r_dist is: " << min_r_dist << " at " << min_r_idx);
  // ROS_INFO_STREAM("min_c_dist is: " << min_c_dist << " at " << min_c_idx);
  // ROS_INFO_STREAM("sample_pt_dist is : " << sample_pt_dist << " at " << sample_pt_idx);
  // ROS_INFO_STREAM("start idx: " << start_idx);
  // ROS_INFO_STREAM("center idx: " << min_c_idx);
  // ROS_INFO_STREAM("end idx: " << end_idx);

  int cur_chunk = start_chunk;
  // Walk from one intersection to the other through the centroid
  for (int i = start_idx; i != end_idx; i = (i+1) % hull.size())
  {
    double dist_r = pointLineDistance2D(hull[i], e_right, c_right);
    double dist_l = pointLineDistance2D(hull[i], e_left, c_left);
    // Thorw out points outside the gripper channel
    if (dist_r > sample_spread  || dist_l > sample_spread)
    {
    }
    else if (cur_chunk == start_chunk || cur_chunk == end_chunk || cur_chunk == center_chunk)
    {
      indices.push_back(i);
    }
    for (int j = 0; j < jump_indices.size(); ++j)
    {
      if(jump_indices[j] == i)
      {
        cur_chunk += 1;
        cur_chunk = cur_chunk % jump_indices.size();
      }
    }
  }

  // Copy to new cloud and return
  XYZPointCloud local_samples;
  pcl16::copyPointCloud(hull, indices, local_samples);
  double x_res = 0.01, y_res=0.01, x_range=2.0*sample_spread, y_range=2.0*sample_spread;
  drawSamplePoints(hull, local_samples, alpha, center_pt, sample_pt, approach_pt, e_left, e_right,
                   c_left, c_right, hull[min_l_idx], hull[min_r_idx], x_res, y_res, x_range, y_range, cur_obj);
  return local_samples;
}

XYZPointCloud transformSamplesIntoSampleLocFrame(XYZPointCloud& samples, ProtoObject& cur_obj,
                                                 pcl16::PointXYZ sample_pt)
{
  XYZPointCloud samples_transformed(samples);
  pcl16::PointXYZ center_pt(cur_obj.centroid[0], cur_obj.centroid[1], cur_obj.centroid[2]);
  double center_angle = std::atan2(center_pt.y - sample_pt.y, center_pt.x - sample_pt.x);
  for (int i = 0; i < samples.size(); ++i)
  {
    // Remove mean and rotate based on pushing_direction
    pcl16::PointXYZ demeaned(samples[i].x - sample_pt.x, samples[i].y - sample_pt.y, samples[i].z);
    double ct = std::cos(center_angle);
    double st = std::sin(center_angle);
    samples_transformed[i].x =  ct*demeaned.x + st*demeaned.y;
    samples_transformed[i].y = -st*demeaned.x + ct*demeaned.y;
    samples_transformed[i].z = demeaned.z;
  }
  return samples_transformed;
}

ShapeDescriptor extractPointHistogramXY(XYZPointCloud& samples, double x_res, double y_res, double x_range,
                                        double y_range)
{
  int n_x_bins = ceil(x_range/x_res);
  int n_y_bins = ceil(y_range/y_res);
  ShapeDescriptor hist(n_y_bins*n_x_bins, 0);
  // Assume demeaned
  for (int i = 0; i < samples.size(); ++i)
  {
    double x_norm = (samples[i].x+x_range*0.5)/ x_range;
    double y_norm = (samples[i].y+y_range*0.5)/ y_range;
    if (x_norm  > 1.0 || x_norm < 0 || y_norm > 1.0 || y_norm <0)
    {
      continue;
    }
    int x_idx = (int)floor(x_norm*n_x_bins);
    int y_idx = (int)floor(y_norm*n_y_bins);
    int idx =  getHistBinIdx(x_idx, y_idx, n_x_bins, n_y_bins);
    hist[idx] += 1;
  }
  std::stringstream descriptor;
  int feat_sum = 0;
  for (int i = 0; i < n_x_bins; ++i)
  {
    for (int j = 0; j < n_y_bins; ++j)
    {
      int idx = getHistBinIdx(i, j, n_x_bins, n_y_bins);
      descriptor << hist[idx] << " ";
      feat_sum += hist[idx];
    }
    descriptor << "\n";
  }
  // ROS_INFO_STREAM("Descriptor: \n" << descriptor.str());
  // ROS_INFO_STREAM("Descriptor size: " << feat_sum << "\tsample size: " << samples.size());
  return hist;
}

void getPointRangesXY(XYZPointCloud& samples, ShapeDescriptor& sd)
{
  double x_min = FLT_MAX;
  double x_max = FLT_MIN;
  double y_min = FLT_MAX;
  double y_max = FLT_MIN;
  for (int i = 0; i < samples.size(); ++i)
  {
    if (samples[i].x > x_max)
    {
      x_max = samples[i].x;
    }
    if (samples[i].x < x_min)
    {
      x_min = samples[i].x;
    }
    if (samples[i].y > y_max)
    {
      y_max = samples[i].y;
    }
    if (samples[i].y < y_min)
    {
      y_min = samples[i].y;
    }
  }
  double x_range = x_max - x_min;
  double y_range = y_max - y_min;
  // ROS_INFO_STREAM("x_range: " << x_range << " : (" << x_min << ", " << x_max << ")");
  // ROS_INFO_STREAM("y_range: " << y_range << " : (" << y_min << ", " << y_max << ")");
  sd.push_back(x_range);
  sd.push_back(y_range);
}

void getCovarianceXYFromPoints(XYZPointCloud& pts, ShapeDescriptor& sd)
{
  Eigen::Matrix<float, 4, 1> centroid;
  if(pcl16::compute3DCentroid(pts, centroid) != 0)
  {
    Eigen::Matrix3f covariance;
    if(pcl16::computeCovarianceMatrix(pts, centroid, covariance) != 0)
    {
      std::stringstream disp_stream;
      for (int i = 0; i < 2; ++i)
      {
        for (int j = i; j < 2; ++j)
        {
          sd.push_back(covariance(i,j));
          disp_stream << covariance(i,j) << " ";
        }
      }
      // ROS_INFO_STREAM("Covariance: " << disp_stream.str());
    }
    else
    {
      ROS_WARN_STREAM("Failed to get covariance matrix");
      for (int i = 0; i < 3; ++i)
      {
        sd.push_back(0);
      }
    }
  }
  else
  {
    ROS_WARN_STREAM("Failed to get centroid");
    for (int i = 0; i < 3; ++i)
    {
      sd.push_back(0);
    }
  }
}

void extractPCAFeaturesXY(XYZPointCloud& samples, ShapeDescriptor& sd)
{
  pcl16::PCA<pcl16::PointXYZ> pca;
  pca.setInputCloud(samples.makeShared());
  Eigen::Vector3f eigen_values = pca.getEigenValues();
  Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
  Eigen::Vector4f centroid = pca.getMean();
  double lambda0 = eigen_values(0);
  double lambda1 = eigen_values(1);
  // ROS_INFO_STREAM("Lambda: " << lambda0 << ", " << lambda1 << ", " << eigen_values(2));
  // ROS_INFO_STREAM("lambda0/lambda1: " << lambda0/lambda1);
  // ROS_INFO_STREAM("Eigen vectors: \n" << eigen_vectors);
  // Get inertia of points
  sd.push_back(lambda0);
  sd.push_back(lambda1);
  sd.push_back(lambda0/lambda1);
  // Get angls of eigen vectors
  double theta0 = atan2(eigen_vectors(1,0), eigen_vectors(0,0));
  double theta1 = atan2(eigen_vectors(1,1), eigen_vectors(0,1));
  // ROS_INFO_STREAM("theta: " << theta0 << ", " << theta1);
  sd.push_back(theta0);
  sd.push_back(theta1);
}

void extractBoundingBoxFeatures(XYZPointCloud& samples, ShapeDescriptor& sd)
{
  std::vector<cv::Point2f> obj_pts;
  for (unsigned int i = 0; i < samples.size(); ++i)
  {
    obj_pts.push_back(cv::Point2f(samples[i].x, samples[i].y));
  }
  cv::RotatedRect box = cv::minAreaRect(obj_pts);
  double l = std::max(box.size.width, box.size.height);
  double w = std::min(box.size.width, box.size.height);
  // ROS_INFO_STREAM("(l,w): " << l << ", " << w << ") l/w: " << l/w);
  sd.push_back(l);
  sd.push_back(w);
  sd.push_back(l/w);
}

ShapeDescriptor extractLocalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj,
                                          pcl16::PointXYZ sample_pt, double sample_spread, double hull_alpha,
                                          double hist_res)
{
  XYZPointCloud local_samples = getLocalSamples(hull, cur_obj, sample_pt, sample_spread, hull_alpha);
  // Transform points into sample_pt frame
  XYZPointCloud transformed_pts = transformSamplesIntoSampleLocFrame(local_samples, cur_obj, sample_pt);

  // Compute features and populate the descriptor
  ShapeDescriptor sd;
#ifdef USE_RANGE_AND_VAR_FEATS
  getPointRangesXY(transformed_pts, sd);
  getCovarianceXYFromPoints(transformed_pts, sd);
  extractPCAFeaturesXY(transformed_pts, sd);
  extractBoundingBoxFeatures(transformed_pts, sd);
#endif // USE_RANGE_AND_VAR_FEATS

  // Get histogram to describe local distribution
  ShapeDescriptor histogram = extractPointHistogramXY(transformed_pts, hist_res, hist_res,
                                                      sample_spread*2, sample_spread*2);
  sd.insert(sd.end(), histogram.begin(), histogram.end());
  return sd;
}



ShapeDescriptor extractGlobalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj, pcl16::PointXYZ sample_pt,
                                           int sample_pt_idx, double sample_spread)
{
  XYZPointCloud transformed_pts = transformSamplesIntoSampleLocFrame(cur_obj.cloud, cur_obj, sample_pt);
  ShapeDescriptor sd;
#ifdef USE_RANGE_AND_VAR_FEATS
  // Get the general features describing the point cloud in the local object frame
  getPointRangesXY(transformed_pts, sd);
  getCovarianceXYFromPoints(transformed_pts, sd);
  extractPCAFeaturesXY(transformed_pts, sd);
  extractBoundingBoxFeatures(transformed_pts, sd);
#endif // USE_RANGE_AND_VAR_FEATS

  // Get shape context
  ShapeLocations sc = extractShapeContextFromSamples(hull, cur_obj, true);
  ShapeDescriptor histogram = sc[sample_pt_idx].descriptor_;
  sd.insert(sd.end(), histogram.begin(), histogram.end());
  std::stringstream descriptor;
  descriptor << "";
  for (unsigned int i = 0; i < histogram.size(); ++i)
  {
    if (i % 5 == 0)
    {
      descriptor << "\n";
    }
    descriptor << " " << histogram[i];
  }
  // ROS_INFO_STREAM("Shape context: " << descriptor.str());


  return sd;
}

ShapeDescriptors extractLocalAndGlobalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj,
                                                    double sample_spread, double hull_alpha,
                                                    double hist_res)
{
  ShapeDescriptors descs;
  for (unsigned int i = 0; i < hull.size(); ++i)
  {
    ShapeDescriptor d = extractLocalAndGlobalShapeFeatures(hull, cur_obj, hull[i], i, sample_spread, hull_alpha,
                                                           hist_res);
    descs.push_back(d);
  }
  return descs;
}

ShapeDescriptor extractLocalAndGlobalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj,
                                                   pcl16::PointXYZ sample_pt, int sample_pt_idx,
                                                   double sample_spread, double hull_alpha, double hist_res)
{
  // ROS_INFO_STREAM("Local");
  ShapeDescriptor local_raw = extractLocalShapeFeatures(hull, cur_obj, sample_pt, sample_spread, hull_alpha, hist_res);
  // Binarize local histogram
  for (unsigned int i = 0; i < local_raw.size(); ++i)
  {
    if (local_raw[i] > 0)
    {
      local_raw[i] = 1.0;
    }
    else
    {
      local_raw[i] = 0.0;
    }
  }
  // Convert back into image for resizing
  int hist_size = std::sqrt(local_raw.size());
  cv::Mat local_hist(cv::Size(hist_size, hist_size), CV_64FC1, cv::Scalar(0.0));
  std::stringstream raw_hist;
  for (int r = 0; r < hist_size; ++r)
  {
    for (int c = 0; c < hist_size; ++c)
    {
      local_hist.at<double>(r,c) = local_raw[r*hist_size+c];
      raw_hist << " " << local_hist.at<double>(r,c);
    }
    raw_hist << "\n";
  }
  // Resize to 6x6
  // TODO: Set 6 as a variable
  cv::Mat local_resize(cv::Size(6,6), CV_64FC1, cv::Scalar(0.0));
  cpl_visual_features::imResize(local_hist, 6./hist_size, local_resize);
  std::stringstream resized_hist;
  for (int r = 0; r < local_resize.rows; ++r)
  {
    for (int c = 0; c < local_resize.cols; ++c)
    {
      resized_hist << " " << local_resize.at<double>(r,c);
    }
    resized_hist << "\n";
  }
  // TODO: Compare the resized histogram to computing 6x6 directly
  // Filter with gaussian
  cv::Mat local_smooth(local_resize.size(), CV_64FC1, cv::Scalar(0.0));
  cv::Mat g_kernel = cv::getGaussianKernel(5, 0.2, CV_64F);
  cv::sepFilter2D(local_resize, local_smooth, CV_64F, g_kernel, g_kernel);
  // Threshold negatives then L1 normalize
  double local_sum = 0.0;
  std::stringstream smooth_hist;
  std::stringstream smooth_hist_clip;
  for (int r = 0; r < local_smooth.rows; ++r)
  {
    for (int c = 0; c < local_smooth.cols; ++c)
    {
      smooth_hist << " " << local_smooth.at<double>(r,c);
      if(local_smooth.at<double>(r,c) < 0)
      {
        local_smooth.at<double>(r,c) = 0.0;
      }
      else
      {
        local_sum += local_smooth.at<double>(r,c);
      }
      smooth_hist_clip << " " << local_smooth.at<double>(r,c);
    }
    smooth_hist << "\n";
    smooth_hist_clip << "\n";
  }
  // L1 normalize local histogram
  ShapeDescriptor local;
  std::stringstream l1_hist;
  for (int r = 0; r < local_smooth.rows; ++r)
  {
    for (int c = 0; c < local_smooth.cols; ++c)
    {
      local_smooth.at<double>(r,c) /= local_sum;
      local.push_back(local_smooth.at<double>(r,c));
      l1_hist << " " << local_smooth.at<double>(r,c);
    }
    l1_hist << "\n";
  }
  // ROS_INFO_STREAM("raw:\n" << raw_hist.str());
  // ROS_INFO_STREAM("resized:\n" << resized_hist.str());
  // ROS_INFO_STREAM("smooth:\n" << smooth_hist.str());
  // ROS_INFO_STREAM("smooth_clip:\n" << smooth_hist_clip.str());
  // ROS_INFO_STREAM("l1_normed:\n" << l1_hist.str());
  // ROS_INFO_STREAM("Local raw size: " << local_raw.size());

  // ROS_INFO_STREAM("Global");
  ShapeDescriptor global = extractGlobalShapeFeatures(hull, cur_obj, sample_pt, sample_pt_idx, sample_spread);
  // Binarize global histogram then L1 normalize
  double global_sum = 0.0;
  for (unsigned int i = 0; i < global.size(); ++i)
  {
    if (global[i] > 0)
    {
      global[i] = 1.0;
      global_sum += 1.0;
    }
    else
    {
      global[i] = 0.0;
    }
  }
  // L1 normalize global histogram
  for (unsigned int i = 0; i < global.size(); ++i)
  {
    global[i] /= global_sum;
  }

  // ROS_INFO_STREAM("local.size() << " << local.size());
  // ROS_INFO_STREAM("global.size() << " << global.size());
  local.insert(local.end(), global.begin(), global.end());
  // ROS_INFO_STREAM("local.size() << " << local.size());
  return local;
}

ShapeDescriptor extractHKSAndGlobalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj,
                                                 pcl16::PointXYZ sample_pt, int sample_pt_idx,
                                                 double sample_spread, double hull_alpha, double hist_res)
{
  ShapeDescriptor local = extractHKSDescriptor(hull, cur_obj, sample_pt, sample_pt_idx,
                                               sample_spread, hull_alpha, hist_res);

  // ROS_INFO_STREAM("Global");
  ShapeDescriptor global = extractGlobalShapeFeatures(hull, cur_obj, sample_pt, sample_pt_idx, sample_spread);
  // Binarize global histogram then L1 normalize
  double global_sum = 0.0;
  for (unsigned int i = 0; i < global.size(); ++i)
  {
    if (global[i] > 0)
    {
      global[i] = 1.0;
      global_sum += 1.0;
    }
    else
    {
      global[i] = 0.0;
    }
  }
  // L1 normalize global histogram
  for (unsigned int i = 0; i < global.size(); ++i)
  {
    global[i] /= global_sum;
  }
  // ROS_INFO_STREAM("local.size() << " << local.size());
  // ROS_INFO_STREAM("global.size() << " << global.size());
  local.insert(local.end(), global.begin(), global.end());
  // ROS_INFO_STREAM("combined.size() << " << local.size());
  return local;
}

/**
 * Create an affinity matrix for a set of ShapeLocations
 *
 * @param locs The vector of ShapeLocation descriptors to compare
 *
 * @return An upper-triangular matrix of all pairwise distances between descriptors
 */
cv::Mat computeShapeFeatureAffinityMatrix(ShapeLocations& locs, bool use_center)
{
  cv::Mat affinity(locs.size(), locs.size(), CV_64FC1, cv::Scalar(0.0));
  double max_affinity = 0.0;
  ShapeDescriptors normalized;
  for (int i = 0; i < locs.size(); ++i)
  {
    ShapeDescriptor desc(locs[i].descriptor_);
    double feature_sum = 0;
    for (int j = 0; j < desc.size(); ++j)
    {
      feature_sum += desc[j];
    }
    if (feature_sum == 0) continue;
    for (int j = 0; j < desc.size(); ++j)
    {
      desc[j] = sqrt(desc[j]/feature_sum);
    }
    normalized.push_back(desc);
  }

  for (int r = 0; r < affinity.rows; ++r)
  {
    for (int c = r; c < affinity.cols; ++c)
    {
      if (r == c)
      {
        affinity.at<double>(r,c) = 1.0;
        continue;
      }
      double sim_score = 1.0 - sqrt(shapeFeatureSquaredEuclideanDist(normalized[r],normalized[c]));
      affinity.at<double>(r,c) = sim_score;
      affinity.at<double>(c,r) = sim_score;
      if (affinity.at<double>(r,c) > max_affinity)
      {
        max_affinity = affinity.at<double>(r,c);
      }
    }
  }
  if (use_center)
  {
    cv::imshow("affinity-with-centers", affinity);
  }
  else
  {
    cv::imshow("affinity", affinity);
  }
  return affinity;
}

/**
 * Compute the (chi-squared) distance between two ShapeLocation descriptors
 *
 * @param a The first descriptor
 * @param b The second descriptor
 * @param gamma Optional scaling value for exponential chi-square kernel
 *
 * @return The distance between a and b
 */
double shapeFeatureChiSquareDist(ShapeDescriptor& a, ShapeDescriptor& b, double gamma)
{
  // compute dist between shape features a and b
  // using chi-squared test statistic
  double chi = 0;

  for (unsigned int k=0; k < a.size(); k++)
  {
    const double a_plus_b = a[k] + b[k];
    if (a_plus_b > 0)
    {
      chi += pow(a[k] - b[k], 2) / (a_plus_b);
    }
  }
  chi = chi;
  if (gamma != 0.0)
  {
    chi = exp(-gamma*chi);
  }
  return chi;
}

/**
 * Compute the squared euclidean distance between two ShapeLocation descriptors
 *
 * @param a The first descriptor
 * @param b The second descriptor
 *
 * @return The distance between a and b
 */
double shapeFeatureSquaredEuclideanDist(ShapeDescriptor& a, ShapeDescriptor& b)
{
  double dist = 0;
  for(int k = 0; k < a.size(); ++k)
  {
    double k_dist = a[k] - b[k];
    dist += k_dist*k_dist;
  }
  return dist;
}

void clusterShapeFeatures(ShapeLocations& locs, int num_clusters, std::vector<int>& cluster_ids, ShapeDescriptors& centers,
                          double min_err_change, int max_iter, int num_retries)
{
  cv::Mat samples(locs.size(), locs[0].descriptor_.size(), CV_64FC1);
  for (int r = 0; r < samples.rows; ++r)
  {
    // NOTE: Normalize features here
    double feature_sum = 0;
    for (int c = 0; c < samples.cols; ++c)
    {
      samples.at<double>(r,c) = locs[r].descriptor_[c];
      feature_sum += samples.at<double>(r,c);
    }
    if (feature_sum == 0)
    {
      continue;
    }
    for (int c = 0; c < samples.cols; ++c)
    {
      samples.at<double>(r,c) /= feature_sum;
      // NOTE: Use Hellinger distance for comparison
      samples.at<double>(r,c) = sqrt(samples.at<double>(r,c));
    }
  }
  cv::TermCriteria term_crit(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, max_iter, min_err_change);
  // cv::Mat labels;
  cv::Mat centers_cv;
  double slack = cv::kmeans(samples, num_clusters, cluster_ids, term_crit, num_retries, cv::KMEANS_PP_CENTERS,
                            centers_cv);
  for (int r = 0; r < centers_cv.rows; ++r)
  {
    ShapeDescriptor s(centers_cv.cols, 0);
    for (int c = 0; c < centers_cv.cols; ++c)
    {
      s[c] = centers_cv.at<double>(r,c);
    }
    centers.push_back(s);
  }
}

/**
 * Find the nearest cluster center to the given descriptor
 *
 * @param descriptor The query descriptor
 * @param centers The set of cluster centers
 *
 * @return The cluster id (index) of the nearest center
 */
int closestShapeFeatureCluster(ShapeDescriptor& descriptor, ShapeDescriptors& centers, double& min_dist)
{
  int min_idx = -1;
  min_dist = FLT_MAX;
  ShapeDescriptor normalized(descriptor);
  double feature_sum = 0;
  for (int i = 0; i < normalized.size(); ++i)
  {
    feature_sum += normalized[i];
  }
  for (int i = 0; i < normalized.size(); ++i)
  {
    normalized[i] = sqrt(normalized[i]/feature_sum);
  }
  for (int c = 0; c < centers.size(); ++c)
  {
    double c_dist = (shapeFeatureSquaredEuclideanDist(normalized, centers[c]));
    if (c_dist < min_dist)
    {
      min_dist = c_dist;
      min_idx = c;
    }
  }
  return min_idx;
}

ShapeDescriptors loadSVRTrainingFeatures(std::string feature_path, int feat_length)
{
  std::ifstream data_in(feature_path.c_str());
  ShapeDescriptors train_feats;
  while (data_in.good())
  {
    ShapeDescriptor feat(feat_length, 0.0);
    char c_line[4096];
    data_in.getline(c_line, 4096);
    std::stringstream line;
    line << c_line;
    int idx;
    double val;
    int num_feats = 0;
    // std::stringstream debug_display;
    while (line >> idx)
    {
      if (line.peek() == ':')
      {
        line.ignore();
        line >> val;
        feat[idx-1] = val;
        num_feats++;
        // debug_display << "[" << idx-1 << "] = " << val << " ";
      }
      if (line.peek() == ' ')
      {
        line.ignore();
      }
    }
    // ROS_INFO_STREAM(debug_display.str());
    if (num_feats > 0)
    {
      train_feats.push_back(feat);
    }
  }
  data_in.close();
  return train_feats;
}

cv::Mat computeChi2Kernel(ShapeDescriptors& sds, std::string feat_path, int local_length, int global_length,
                          double gamma_local, double gamma_global, double mixture_weight)
{
  ShapeDescriptors train_feats = loadSVRTrainingFeatures(feat_path, local_length + global_length);
  cv::Mat K_local(train_feats.size(), sds.size(), CV_64FC1, cv::Scalar(0.0));
  cv::Mat K_global(train_feats.size(), sds.size(), CV_64FC1, cv::Scalar(0.0));
  for (int i = 0; i < sds.size(); ++i)
  {
    ShapeDescriptor a_local(sds[i].begin(), sds[i].begin()+local_length);
    ShapeDescriptor a_global(sds[i].begin()+local_length, sds[i].end());
    for (int j = 0; j < train_feats.size(); ++j)
    {
      ShapeDescriptor b_local(train_feats[j].begin(), train_feats[j].begin()+local_length);
      ShapeDescriptor b_global(train_feats[j].begin()+local_length, train_feats[j].end());

      K_local.at<double>(j, i) = shapeFeatureChiSquareDist(a_local, b_local, gamma_local);
      K_global.at<double>(j, i) = shapeFeatureChiSquareDist(a_global, b_global, gamma_global);
    }
  }
  // Linear combination of local and global kernels
  cv::Mat K = mixture_weight * K_global + (1 - mixture_weight) * K_local;
  return K;
}

pcl16::PointXYZ estimateObjectContactLocation(XYZPointCloud& hull_cloud, PushTrackerState& cur_state,
                                              pcl16::PointXYZ& tool_pt0, pcl16::PointXYZ& tool_pt1)
{
  double min_dist = FLT_MAX;
  int min_idx = -1;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    double dist_i = dist(hull_cloud.at(i), tool_pt1);
    if (dist_i < min_dist)
    {
      min_dist = dist_i;
      min_idx = i;
    }
  }
  return hull_cloud.at(min_idx);
}


XYZPointCloud laplacianSmoothBoundary(XYZPointCloud& hull_cloud, int m)
{
  const int n = hull_cloud.size();

  // Construct the smoothing matrix
  Eigen::MatrixXd L(n,n);
  computeTutteLaplacian(hull_cloud, L);
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n,n) - 0.5*L;

  Eigen::MatrixXd X(n, 3);
  for (int i = 0; i < n; ++i)
  {
    X(i,0) = hull_cloud.at(i).x;
    X(i,1) = hull_cloud.at(i).y;
    X(i,2) = hull_cloud.at(i).z;
  }

  // Raise matrix to the m power
  Eigen::MatrixXd Sm(S);
  for (int i = 1; i < m; ++i)
  {
    Sm *= S;
  }
  Eigen::MatrixXd X_hat = Sm*X;

  XYZPointCloud smoothed_cloud;
  smoothed_cloud.header = hull_cloud.header;
  smoothed_cloud.width = hull_cloud.size();
  smoothed_cloud.height = 1;
  smoothed_cloud.is_dense = false;
  smoothed_cloud.resize(smoothed_cloud.width);
  for (int i = 0; i < n; ++i)
  {
    smoothed_cloud.at(i) = pcl16::PointXYZ(X_hat(i,0), X_hat(i,1), X_hat(i,2));
  }
  return smoothed_cloud;
}

XYZPointCloud laplacianBoundaryCompression(XYZPointCloud& hull_cloud, int k)
{
  const int n = hull_cloud.size();
  Eigen::MatrixXd X(n, 3);
  for (int i = 0; i < n; ++i)
  {
    X(i,0) = hull_cloud.at(i).x;
    X(i,1) = hull_cloud.at(i).y;
    X(i,2) = hull_cloud.at(i).z;
  }

  Eigen::MatrixXd L(n,n);
  computeInverseDistLaplacian(hull_cloud, L);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L);
  Eigen::VectorXd Lambda = es.eigenvalues();
  Eigen::MatrixXd E = es.eigenvectors();

  Eigen::MatrixXd X_tilde = E.transpose()*X;
  if (k > n) k = n;
  for (int i = n-1; i >= k; --i)
  {
    X_tilde(i,0) = 0.0;
    X_tilde(i,1) = 0.0;
    X_tilde(i,2) = 0.0;
  }

  Eigen::MatrixXd X_hat = E*X_tilde;

  XYZPointCloud compressed_cloud;
  compressed_cloud.header = hull_cloud.header;
  compressed_cloud.width = hull_cloud.size();
  compressed_cloud.height = 1;
  compressed_cloud.is_dense = false;
  compressed_cloud.resize(compressed_cloud.width);
  for (int i = 0; i < n; ++i)
  {
    compressed_cloud.at(i) = pcl16::PointXYZ(X_hat(i,0), X_hat(i,1), X_hat(i,2));
  }
  return compressed_cloud;
}

std::vector<XYZPointCloud> laplacianBoundaryCompressionAllKs(XYZPointCloud& hull_cloud)
{
  const int n = hull_cloud.size();
  Eigen::MatrixXd X(n, 3);
  for (int i = 0; i < n; ++i)
  {
    X(i,0) = hull_cloud.at(i).x;
    X(i,1) = hull_cloud.at(i).y;
    X(i,2) = hull_cloud.at(i).z;
  }

  Eigen::MatrixXd L(n,n);
  computeInverseDistLaplacian(hull_cloud, L);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L);
  Eigen::VectorXd Lambda = es.eigenvalues();
  Eigen::MatrixXd E = es.eigenvectors();

  Eigen::MatrixXd X_tilde = E.transpose()*X;
  std::vector<XYZPointCloud> clouds;
  for (int i = n-1; i > 0; --i)
  {
    X_tilde(i,0) = 0.0;
    X_tilde(i,1) = 0.0;
    X_tilde(i,2) = 0.0;

    Eigen::MatrixXd X_hat = E*X_tilde;

    XYZPointCloud compressed_cloud;
    compressed_cloud.header = hull_cloud.header;
    compressed_cloud.width = hull_cloud.size();
    compressed_cloud.height = 1;
    compressed_cloud.is_dense = false;
    compressed_cloud.resize(compressed_cloud.width);
    for (int j = 0; j < n; ++j)
    {
      compressed_cloud.at(j) = pcl16::PointXYZ(X_hat(j,0), X_hat(j,1), X_hat(j,2));
    }
    clouds.push_back(compressed_cloud);
  }
  return clouds;
}

ShapeDescriptor extractHKSDescriptor(XYZPointCloud& hull, ProtoObject& cur_obj,
                                     pcl16::PointXYZ sample_pt, int sample_pt_idx,
                                     double sample_spread, double hull_alpha, double hist_res)
{
  cv::Mat K_xx = extractHeatKernelSignatures(hull);
  ShapeDescriptor desc;
  for (int c = 0; c < K_xx.cols; ++c)
  {
    desc.push_back(K_xx.at<double>(sample_pt_idx,c));
  }
  return desc;
}

cv::Mat extractHeatKernelSignatures(XYZPointCloud& hull_cloud, int connectivity)
{
  const int n = hull_cloud.size();
  Eigen::MatrixXd L(n,n);
  // computeInverseDistLaplacian(hull_cloud, L);
  computeInverseKDistLaplacian(hull_cloud, L, connectivity);
  // Perform the generalized eigenvalue decomposition with matrix A
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ges(L);
  Eigen::VectorXd Lambda = ges.eigenvalues();
  Eigen::MatrixXd Phi = ges.eigenvectors();
  // Normalize the eigenvectors
  for (int i = 0; i < n; ++i)
  {
    if (Phi.col(i).sum() != 0.0)
    {
      Phi.col(i) = Phi.col(i)/Phi.col(i).sum();
    }
  }

  // ROS_INFO_STREAM("Eigenvalues: " << Lambda);
  // ROS_INFO_STREAM("Phi[0] = " << Phi.col(0));
  // ROS_INFO_STREAM("Phi[n-1] = " << Phi.col(n-1));
  const int num_ts = 100;
  const double min_t = abs(4.0*log(10.0) / Lambda(n-1));
  const double max_t = abs(4.0*log(10.0) / Lambda(1));
  const double t_step = (log(max_t)-log(min_t))/num_ts;
  // ROS_INFO_STREAM("Min_t: "  << min_t);
  // ROS_INFO_STREAM("Max_t: "  << max_t);

  Eigen::VectorXd T(num_ts);
  Eigen::MatrixXd hs(n, num_ts);
  // NOTE: Ignore eigenvector 0
  // const int start_eig = 0;
  const int start_eig = 1;
  for (int j = 0; j < num_ts; ++j)
  {
    T(j) = exp(t_step*(j+1.0));
    for (int i = start_eig; i < n; ++i)
    {
      const float h = std::exp(-Lambda(i)*T(j));
      // ROS_INFO_STREAM("h[" << i << ", " << j << "] = " << h);
      hs(i, j) = h;
    }
    // ROS_INFO_STREAM("Heat trace[" << j << "] = " << heat_trace);
  }

  // Compute the heat kernel signature at each point for the specified time scalse
  cv::Mat K_xx(n, num_ts, CV_64FC1, cv::Scalar(0.0));
  for (int x = 0; x < n; ++x)
  {
    for (int j = 0; j < T.size(); ++j)
    {
      for (int i = start_eig; i < n; ++i)
      {
        K_xx.at<double>(x,j) += hs(i,j)*Phi(x,i)*Phi(x,i);
      }
    }
  }
  return K_xx;
}

double compareHeatKernelSignatures(cv::Mat a, cv::Mat b)
{
  double sum = 0.0;
  for (int i = 0; i < a.cols; ++i)
  {
    double diff_i = a.at<double>(0,i) - b.at<double>(0,i);
    sum += diff_i*diff_i;
  }
  // ROS_INFO_STREAM("dist^2 = " << sum);
  return std::sqrt(sum);
}

double compareHeatKernelSignatures(cv::Mat& K_xx, int a, int b)
{
  double sum = 0.0;
  for (int i = 0; i < K_xx.cols; ++i)
  {
    double diff_i = K_xx.at<double>(a,i) - K_xx.at<double>(b,i);
    sum += diff_i*diff_i;
  }
  return std::sqrt(sum);
}

cv::Scalar getColorFromDist(const double dist, const double max_dist,const double min_dist)
{
  const double dist_norm = (dist-min_dist)/(max_dist-min_dist);
  cv::Scalar color(0, (1.0-dist_norm)*255, dist_norm*255);
  return color;
}

cv::Mat visualizeHKSDists(XYZPointCloud& hull_cloud, cv::Mat K_xx, PushTrackerState& cur_state, int target_idx)
{
  // ROS_INFO_STREAM("Comparing all points to point: " << target_idx);
  std::vector<double> K_dists;
  for (int x = 0; x < hull_cloud.size(); ++x)
  {
    K_dists.push_back(compareHeatKernelSignatures(K_xx.row(target_idx), K_xx.row(x)));
    // ROS_INFO_STREAM("dist = " << K_dists[x]);
  }

  double min_dist = 0;
  double max_dist = 0;
  cv::minMaxLoc(K_dists, &min_dist, &max_dist);
  // ROS_INFO_STREAM("Min dist: " << min_dist);
  // ROS_INFO_STREAM("Max dist: " << max_dist);

  // Project hull into image with colors projected based on distance
  double max_y = 0.2;
  double min_y = -0.2;
  double max_x = 0.2;
  double min_x = -0.2;
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC3, cv::Scalar(255,255,255));

  int prev_img_x;
  int prev_img_y;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    pcl16::PointXYZ obj_pt =  worldPointInObjectFrame(hull_cloud[i], cur_state);
    int img_x = cols - objLocToIdx(obj_pt.x, min_x, max_x);
    int img_y = objLocToIdx(obj_pt.y, min_y, max_y);
    if (i == target_idx) // Circle target index location
    {
      cv::circle(footprint, cv::Point(img_x, img_y), 3, cv::Scalar(0,0,0), 3);
      // ROS_INFO_STREAM("Dist from target to target is: " << K_dists[i]);
    }
    cv::Scalar color = getColorFromDist(K_dists[i], max_dist, min_dist);

    cv::circle(footprint, cv::Point(img_x, img_y), 1, color, 3);
    if (i > 0)
    {
      cv::line(footprint, cv::Point(img_x, img_y), cv::Point(prev_img_x, prev_img_y), color, 1);
    }
    prev_img_x = img_x;
    prev_img_y = img_y;
  }
  return footprint;
}

cv::Mat visualizeHKSDistMatrix(XYZPointCloud& hull_cloud, cv::Mat K_xx)
{
  cv::Mat dist_matrix(hull_cloud.size(), hull_cloud.size(), CV_64FC1);
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    for (int j = i; j < hull_cloud.size(); ++j)
    {
      dist_matrix.at<double>(i,j) = compareHeatKernelSignatures(K_xx.row(i), K_xx.row(j));
      dist_matrix.at<double>(j,i) = dist_matrix.at<double>(i,j);
    }
  }
  return dist_matrix;
}

void computeDistLaplacian(XYZPointCloud& hull_cloud, Eigen::MatrixXd& L)
{
  const int n = hull_cloud.size();
  for (int r = 0; r < n; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      L(r,c) = 0.0;
    }
  }
  for (int i = 0; i < n; ++i)
  {
    // Circular indexes
    const int i_minus_1 = (n+i-1)%n;
    const int i_plus_1 = (i+1)%n;
    const float dist_r = dist(hull_cloud.at(i), hull_cloud.at(i_minus_1));
    const float dist_f = dist(hull_cloud.at(i), hull_cloud.at(i_plus_1));
    const float dist_sum = dist_r + dist_f;
    L(i, i_minus_1) = -dist_r;
    L(i, i) = dist_sum;
    L(i, i_plus_1) = -dist_f;
  }
}

void computeNormalizedDistLaplacian(XYZPointCloud& hull_cloud, Eigen::MatrixXd& L)
{
  const int n = hull_cloud.size();
  for (int r = 0; r < n; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      L(r,c) = 0.0;
    }
  }
  for (int i = 0; i < n; ++i)
  {
    // Circular indexes
    const int i_minus_1 = (n+i-1)%n;
    const int i_plus_1 = (i+1)%n;
    const float dist_r = dist(hull_cloud.at(i), hull_cloud.at(i_minus_1));
    const float dist_f = dist(hull_cloud.at(i), hull_cloud.at(i_plus_1));
    const float dist_sum = dist_r + dist_f;
    L(i, i_minus_1) = -dist_r/dist_sum;
    L(i, i) = 1.0;
    L(i, i_plus_1) = -dist_f/dist_sum;
  }
}

void computeInverseDistLaplacian(XYZPointCloud& hull_cloud, Eigen::MatrixXd& L)
{
  const int n = hull_cloud.size();
  for (int r = 0; r < n; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      L(r,c) = 0.0;
    }
  }
  for (int i = 0; i < n; ++i)
  {
    // Circular indexes
    const int i_minus_1 = (n+i-1)%n;
    const int i_plus_1 = (i+1)%n;
    const float dist_r = 1.0/dist(hull_cloud.at(i), hull_cloud.at(i_minus_1));
    const float dist_f = 1.0/dist(hull_cloud.at(i), hull_cloud.at(i_plus_1));
    const float dist_sum = dist_r + dist_f;
    L(i, i_minus_1) = -dist_r;
    L(i, i) = dist_sum;
    L(i, i_plus_1) = -dist_f;
  }
}

void computeNormalizedInverseDistLaplacian(XYZPointCloud& hull_cloud, Eigen::MatrixXd& L)
{
  const int n = hull_cloud.size();
  for (int r = 0; r < n; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      L(r,c) = 0.0;
    }
  }
  for (int i = 0; i < n; ++i)
  {
    // Circular indexes
    const int i_minus_1 = (n+i-1)%n;
    const int i_plus_1 = (i+1)%n;
    const float dist_r = 1.0/dist(hull_cloud.at(i), hull_cloud.at(i_minus_1));
    const float dist_f = 1.0/dist(hull_cloud.at(i), hull_cloud.at(i_plus_1));
    const float dist_sum = dist_r + dist_f;
    L(i, i_minus_1) = -dist_r/dist_sum;
    L(i, i) = 1.0;
    L(i, i_plus_1) = -dist_f/dist_sum;
  }
}

void computeTutteLaplacian(XYZPointCloud& hull_cloud, Eigen::MatrixXd& L)
{
  const int n = hull_cloud.size();
  for (int r = 0; r < n; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      L(r,c) = 0.0;
    }
  }
  for (int i = 0; i < n; ++i)
  {
    // Circular indexes
    const int i_minus_1 = (n+i-1)%n;
    const int i_plus_1 = (i+1)%n;
    L(i, i_minus_1) = -0.5;
    L(i, i) = 1;
    L(i, i_plus_1) = -0.5;
  }
}

void computeInverseKDistLaplacian(XYZPointCloud& hull_cloud, Eigen::MatrixXd& L, int K)
{
  ROS_INFO_STREAM("Connected to " << K << " neighbors");
  const int n = hull_cloud.size();
  for (int r = 0; r < n; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      L(r,c) = 0.0;
    }
  }
  for (int i = 0; i < n; ++i)
  {
    float dist_sum = 0.0;
    // Circular indexes
    for (int k = 1; k <= K; ++k)
    {
      const int i_minus_k = (n+i-k)%n;
      const int i_plus_k = (i+k)%n;
      const float dist_r = 1.0/dist(hull_cloud.at(i), hull_cloud.at(i_minus_k));
      const float dist_f = 1.0/dist(hull_cloud.at(i), hull_cloud.at(i_plus_k));
      dist_sum += dist_r + dist_f;
      L(i, i_minus_k) = -dist_r;
      L(i, i_plus_k) = -dist_f;
    }
    L(i, i) = dist_sum;
  }
}

};
