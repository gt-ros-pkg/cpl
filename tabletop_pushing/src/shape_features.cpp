#include <sstream>
#include <iostream>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <pcl16_ros/transforms.h>
#include <pcl16/ros/conversions.h>
#include <pcl16/surface/concave_hull.h>
#include <iostream>

#define XY_RES 0.00075
using namespace cpl_visual_features;
using tabletop_pushing::ProtoObject;
namespace tabletop_pushing
{

inline int worldLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}

std::vector<cv::Point2f> getObjectBoundarySamples(ProtoObject& cur_obj)
{
  // Get 2D projection of object
  XYZPointCloud footprint_cloud(cur_obj.cloud);
  for (int i = 0; i < footprint_cloud.size(); ++i)
  {
    footprint_cloud.at(i).z = 0.0;
  }
  // TODO: Examine sensitivity of hull_alpha...
  double hull_alpha = 0.01;
  XYZPointCloud hull_cloud;
  pcl16::ConcaveHull<pcl16::PointXYZ> hull;
  hull.setDimension(2);
  hull.setInputCloud(footprint_cloud.makeShared());
  hull.setAlpha(hull_alpha);
  hull.reconstruct(hull_cloud);

  std::vector<cv::Point2f> samples;
  for (int i = 0; i < hull_cloud.size(); ++i)
  {
    cv::Point2f pt(hull_cloud.at(i).x, hull_cloud.at(i).y);
    samples.push_back(pt);
  }
  return samples;
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

ShapeLocations extractObjectShapeFeatures(ProtoObject& cur_obj)
{
  Samples2f samples = getObjectBoundarySamples(cur_obj);
  int radius_bins = 5;
  int theta_bins = 12;
  cv::Point2f center(cur_obj.centroid[0], cur_obj.centroid[1]);
  ShapeDescriptors descriptors = constructDescriptors(samples, center, radius_bins, theta_bins);
  ShapeLocations locs;
  for (unsigned int i = 0; i < descriptors.size(); ++i)
  {
    geometry_msgs::Point pt;
    pt.x = samples[i].x;
    pt.y = samples[i].y;
    ShapeLocation loc(pt, descriptors[i]);
    locs.push_back(loc);
  }
  return locs;
}

/**
 * Create an (upper-triangular) affinity matrix for a set of ShapeLocations
 *
 * @param locs The vector of ShapeLocation descriptors to compare
 *
 * @return An upper-triangular matrix of all pairwise distances between descriptors
 */
cv::Mat computeShapeFeatureAffinityMatrix(ShapeLocations& locs)
{
  cv::Mat affinity(locs.size(), locs.size(), CV_64FC1, cv::Scalar(0.0));
  double max_affinity = 0.0;
  for (int r = 0; r < affinity.rows; ++r)
  {
    for (int c = r+1; c < affinity.cols; ++c)
    {
      affinity.at<double>(r,c) = shapeFeatureSimilarity(locs[r], locs[c]);
      if (affinity.at<double>(r,c) > max_affinity)
      {
        max_affinity = affinity.at<double>(r,c);
      }
    }
  }
  // cv::imshow("affinity", affinity/max_affinity);
  return affinity;
}

/**
 * Compute the (chi-squared) distance between two ShapeLocation descriptors
 *
 * @param a The first descriptor
 * @param b The second descriptor
 *
 * @return The distance between a and b
 */
double shapeFeatureSimilarity(ShapeLocation& a, ShapeLocation& b)
{
  // compute affinity between shape context i and j
  // using chi-squared test statistic
  double d_affinity = 0;

  double a_sum = 0.0;
  double b_sum = 0.0;
  for (unsigned int k=0; k < a.descriptor_.size(); k++)
  {
    a_sum += a.descriptor_[k];
    b_sum += b.descriptor_[k];
  }
  if (a_sum == 0.0)
  {
    a_sum = 1.0;
  }
  if (b_sum == 0.0)
  {
    b_sum = 1.0;
  }
  for (unsigned int k=0; k < a.descriptor_.size(); k++)
  {
    // NOTE: Normalizing to have L1 of 1 for comparison
    const double a_k = a.descriptor_[k]/a_sum;
    const double b_k = b.descriptor_[k]/b_sum;
    const double a_plus_b = a_k + b_k;
    if (a_plus_b > 0)
    {
      d_affinity += pow(a_k - b_k, 2) / (a_plus_b);
    }
  }
  d_affinity = 1.0 - d_affinity/2;
  return d_affinity;
}

};
