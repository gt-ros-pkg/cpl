#include <sstream>
#include <iostream>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <pcl16_ros/transforms.h>
#include <pcl16/ros/conversions.h>
#include <pcl16/surface/concave_hull.h>
#include <cpl_visual_features/comp_geometry.h>
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

cv::Point worldPtToImgPt(pcl16::PointXYZ world_pt, double min_x, double max_x,
                         double min_y, double max_y)
{
  cv::Point img_pt(worldLocToIdx(world_pt.x, min_x, max_x),
                   worldLocToIdx(world_pt.y, min_y, max_y));
  return img_pt;
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
  computeShapeFeatureAffinityMatrix(locs, use_center);
  return locs;
}

void drawSamplePoints(XYZPointCloud& hull, XYZPointCloud& samples, pcl16::PointXYZ& center_pt,
                      pcl16::PointXYZ& sample_pt, pcl16::PointXYZ& approach_pt,
                      pcl16::PointXYZ e_left, pcl16::PointXYZ e_right,
                      pcl16::PointXYZ c_left, pcl16::PointXYZ c_right,
                      pcl16::PointXYZ i_left, pcl16::PointXYZ i_right)
{
  double max_y = 0.5;
  double min_y = -0.2;
  double max_x = 1.0;
  double min_x = 0.2;
  // TODO: Make function to get cv::Size from (max_x, min_x, max_y, min_y, XY_RES)
  // TODO: Make sure everything is getting drawn
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC3, cv::Scalar(0.0,0.0,0.0));

  for (int i = 0; i < hull.size(); ++i)
  {
    int j = (i+1) % hull.size();
    pcl16::PointXYZ obj_pt0 = hull[i];
    pcl16::PointXYZ obj_pt1 = hull[j];
    cv::Point img_pt0 = worldPtToImgPt(obj_pt0, min_x, max_x, min_y, max_y);
    cv::Point img_pt1 = worldPtToImgPt(obj_pt1, min_x, max_x, min_y, max_y);
    cv::Scalar color(0, 0, 128);
    cv::circle(footprint, img_pt0, 1, color);
    cv::line(footprint, img_pt0, img_pt1, color);
  }
  for (int i = 0; i < samples.size(); ++i)
  {
    cv::Point img_pt = worldPtToImgPt(samples[i], min_x, max_x, min_y, max_y);
    cv::Scalar color(0, 255, 0);
    cv::circle(footprint, img_pt, 3, color);
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

  cv::line(footprint, img_approach_pt, img_center, cv::Scalar(0,255,255));
  cv::line(footprint, e_left_img, e_right_img, cv::Scalar(0,255,255));
  cv::line(footprint, e_left_img, c_left_img, cv::Scalar(255,0,255));
  cv::line(footprint, e_right_img, c_right_img, cv::Scalar(0,255,255));
  cv::circle(footprint, img_center, 3, cv::Scalar(0,255,255));
  cv::circle(footprint, img_approach_pt, 3, cv::Scalar(0,255,255));
  cv::circle(footprint, e_left_img, 3, cv::Scalar(255,0,255));
  cv::circle(footprint, e_right_img, 3, cv::Scalar(0,255,255));

  // Draw sample point last
  cv::line(footprint, img_sample_pt, img_center, cv::Scalar(255,255,255));
  cv::circle(footprint, img_sample_pt, 3, cv::Scalar(255,255,255));
  cv::circle(footprint, i_left_img, 3, cv::Scalar(255,0,255));
  cv::circle(footprint, i_right_img, 3, cv::Scalar(0,255,255));

  cv::imshow("local samples", footprint);
  cv::waitKey();
}

XYZPointCloud getLocalSamples(XYZPointCloud& hull, ProtoObject& cur_obj, pcl16::PointXYZ sample_pt,
                              float sample_spread)
{
  // TODO: This is going to get all points in front of the gripper, not only those on the close boundary
  float radius = sample_spread / 2.0;
  pcl16::PointXYZ center_pt(cur_obj.centroid[0], cur_obj.centroid[1], cur_obj.centroid[2]);
  float center_angle = std::atan2(center_pt.y - sample_pt.y, center_pt.x - sample_pt.x);
  float approach_dist = 0.15;
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
  // ROS_INFO_STREAM("center_pt: " << center_pt);
  // ROS_INFO_STREAM("sample_pt: " << sample_pt);
  // ROS_INFO_STREAM("approach_pt: " << approach_pt);
  // ROS_INFO_STREAM("e_vect: " << e_vect);
  // ROS_INFO_STREAM("e_left: " << e_left);
  // ROS_INFO_STREAM("e_right: " << e_right);

  // Test intersection of gripper end point rays and all line segments on the object boundary
  double min_sample_pt_dist = FLT_MAX;
  int sample_pt_idx = -1;
  pcl16::PointXYZ l_intersection;
  pcl16::PointXYZ r_intersection;
  pcl16::PointXYZ c_intersection;
  double min_l_dist = FLT_MAX;
  double min_r_dist = FLT_MAX;
  double min_c_dist = FLT_MAX;
  int min_l_idx = -1;
  int min_r_idx = -1;
  int min_c_idx = -1;

  for (int i = 0; i < hull.size(); i++)
  {
    int idx0 = i;
    int idx1 = (i+1) % hull.size();
    // TODO: Make sure the lineSegmentIntersection works correctly
    pcl16::PointXYZ intersection;
    // LEFT
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
    // RIGHT
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
    // CENTER
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
    // SAMPLE PT
    double sample_pt_dist = dist(sample_pt, hull[i]);
    if (sample_pt_dist < min_sample_pt_dist)
    {
      min_sample_pt_dist = sample_pt_dist;
      sample_pt_idx = i;
    }
  }

  // Default to smaple_pt if no intersection also
  double sample_pt_dist = dist(approach_pt, sample_pt);
  if (min_c_idx == -1)
  {
    min_c_idx = sample_pt_idx;
  }
  else
  {
    if (sample_pt_dist <= min_c_dist)
    {
      min_c_idx = sample_pt_idx;
    }
  }

  // TODO: Deal with no intersections
  // TODO: Find farthest point left before moving back towards center
  if (min_l_idx == -1)
  {
    ROS_WARN_STREAM("No left intersection");
    min_l_idx = sample_pt_idx;
  }
  if (min_r_idx == -1)
  {
    ROS_WARN_STREAM("No right intersection");
    min_r_idx = sample_pt_idx;
  }
  ROS_INFO_STREAM("min_l_dist is: " << min_l_dist << " at " << min_l_idx);
  ROS_INFO_STREAM("min_r_dist is: " << min_r_dist << " at " << min_r_idx);
  ROS_INFO_STREAM("min_c_dist is: " << min_c_dist << " at " << min_c_idx);
  ROS_INFO_STREAM("sample_pt_dist is : " << sample_pt_dist << " at " << sample_pt_idx);

  std::vector<int> indices;
  indices.push_back(min_l_idx);
  indices.push_back((min_l_idx+1) % hull.size());
  indices.push_back(min_r_idx);
  indices.push_back((min_r_idx+1) % hull.size());

  int start_idx = min_l_idx < min_r_idx ? min_l_idx : min_r_idx;
  int end_idx = min_l_idx < min_r_idx ? min_r_idx : min_l_idx;
  if (min_c_idx >= start_idx && min_c_idx <= end_idx)
  {
    // Good:
    ROS_INFO_STREAM("Walking inside");
  }
  else
  {
    ROS_INFO_STREAM("Walking outside");
    int tmp_idx = start_idx;
    start_idx = end_idx;
    end_idx = tmp_idx;
  }
  ROS_INFO_STREAM("start idx: " << start_idx);
  ROS_INFO_STREAM("end idx: " << end_idx);

  // Walk from one intersection to the other through the centroid
  for (int i = start_idx; i != end_idx; i = (i+1) % hull.size())
  {
    indices.push_back(i);
  }

  // Copy to new cloud and return
  XYZPointCloud local_samples;
  pcl16::copyPointCloud(hull, indices, local_samples);
  drawSamplePoints(hull, local_samples, center_pt, sample_pt, approach_pt, e_left, e_right,
                   c_left, c_right, hull[min_l_idx], hull[min_r_idx]);
  return local_samples;
}

ShapeDescriptor extractLocalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj,
                                          pcl16::PointXYZ sample_pt, float sample_spread)
{
  XYZPointCloud local_samples = getLocalSamples(hull, cur_obj, sample_pt, sample_spread);
  ShapeDescriptor sd;
  return sd;
}

// ShapeDescriptor extractGlobalShapeFeatures(XYZPointCloud& samples_pcl, ProtoObject& cur_obj,
//                                            pcl16::PointXYZ sample_loc, float sample_spread)
// {
// }


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
 *
 * @return The distance between a and b
 */
double shapeFeatureChiSquareDist(ShapeDescriptor& a, ShapeDescriptor& b)
{
  // compute affinity between shape context i and j
  // using chi-squared test statistic
  double d_affinity = 0;

  double a_sum = 0.0;
  double b_sum = 0.0;
  for (unsigned int k=0; k < a.size(); k++)
  {
    a_sum += a[k];
    b_sum += b[k];
  }
  if (a_sum == 0.0)
  {
    a_sum = 1.0;
  }
  if (b_sum == 0.0)
  {
    b_sum = 1.0;
  }
  for (unsigned int k=0; k < a.size(); k++)
  {
    // NOTE: Normalizing to have L1 of 1 for comparison
    const double a_k = a[k]/a_sum;
    const double b_k = b[k]/b_sum;
    const double a_plus_b = a_k + b_k;
    if (a_plus_b > 0)
    {
      d_affinity += pow(a_k - b_k, 2) / (a_plus_b);
    }
  }
  d_affinity = d_affinity/2;
  return d_affinity;
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
  cv::Mat samples(locs.size(), locs[0].descriptor_.size(), CV_32FC1);
  for (int r = 0; r < samples.rows; ++r)
  {
    // NOTE: Normalize features here
    float feature_sum = 0;
    for (int c = 0; c < samples.cols; ++c)
    {
      samples.at<float>(r,c) = locs[r].descriptor_[c];
      feature_sum += samples.at<float>(r,c);
    }
    if (feature_sum == 0)
    {
      continue;
    }
    for (int c = 0; c < samples.cols; ++c)
    {
      samples.at<float>(r,c) /= feature_sum;
      // NOTE: Use Hellinger distance for comparison
      samples.at<float>(r,c) = sqrt(samples.at<float>(r,c));
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
      s[c] = centers_cv.at<float>(r,c);
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
  float feature_sum = 0;
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

};
